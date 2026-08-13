"""
train_tuning_optimized.py
==========================

Versi optimasi dari train_tuning.py.

Perubahan inti vs grid search penuh:

1. RANDOM SEARCH, bukan exhaustive grid.
   Grid asli Pascabayar = 7*5*5*4*4*4 = 11.200 kombinasi. Random search
   dengan sample yang jauh lebih kecil (mis. 150-300) sudah menutupi ruang
   hyperparameter dengan baik, karena tidak semua dimensi (layer_sizes,
   learning_rate, l2_lambda, clip_value, batch_size) punya pengaruh yang
   sama besar -- grid penuh membuang compute merata padahal sebagian besar
   sel tidak informatif (Bergstra & Bengio, 2012).

2. SUCCESSIVE HALVING (ASHA-style).
   `epochs` DIKELUARKAN dari ruang hyperparameter yang di-random-sample --
   epoch sekarang jadi "resource budget" yang dieskalasi bertahap per rung
   (mis. 30 -> 90 -> 270 epoch). Di tiap rung, semua kandidat yang masih
   hidup dilatih dengan budget rung itu, lalu hanya 1/eta kandidat terbaik
   (berdasar val RMSE) yang lanjut ke rung berikutnya. Kandidat yang
   memang jelek dibuang di rung pertama -- SEBELUM menghabiskan compute
   sampai 500 epoch seperti di grid search penuh.

3. EARLY STOPPING berbasis val RMSE + patience.
   Trainer lama hanya berhenti kalau loss NaN/Inf (divergence guard), tapi
   tetap memakai bobot dari epoch TERAKHIR yang bisa saja sudah lewat
   titik terbaiknya (overfit). Di sini bobot terbaik di-checkpoint via
   `copy.deepcopy(model)` tiap kali val RMSE membaik, dan training
   berhenti lebih awal kalau tidak ada perbaikan selama `patience` epoch.

4. PARALELISASI lintas kandidat via ProcessPoolExecutor.
   Tiap evaluasi kombinasi hyperparameter independen satu sama lain
   (embarrassingly parallel) -- training loop sekuensial lama diganti
   dengan pool proses yang memakai semua core CPU yang tersedia.

CATATAN: file ini reuse utilities (metrics, load_data, divergence guard,
predict_mlp/predict_pv) dari train_tuning.py yang sudah ada. Taruh file
ini di folder yang sama dengan train_tuning.py. Kalau nama file/modul
aslinya beda, sesuaikan baris import di bawah.
"""

import copy
import math
import random
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

# Pastikan import src dikenali saat dijalankan langsung
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─── Reuse dari script grid search asli ──────────────────────────────
# Sesuaikan nama modul ini kalau file aslinya bukan train_tuning.py
from src.tuning.train_tuning import (
    compute_rmse, compute_mae, compute_mape, compute_r2, evaluate_metrics,
    is_invalid_number, make_divergent_metrics,
    load_data, predict_mlp, predict_pv,
)

from src.models.pascabayar import PascabayarModel
from src.models.prabayar import PrabayarModel
from src.models.pascabayar_place_value import PascabayarPlaceValueModel
from src.pipeline.preprocessing import inverse_transform_target


# =====================================================================
# RANDOM SEARCH SAMPLING
# =====================================================================

def sample_param_pool(param_grid: dict, n_samples: int, seed: int = 42) -> list[dict]:
    """
    Random sample kombinasi hyperparameter TANPA materialize full cartesian
    product. param_grid TIDAK boleh mengandung key 'epochs' -- epoch
    dikontrol terpisah oleh successive halving sebagai resource budget.
    """
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    seen = set()
    pool = []

    # Kalau ruang kombinasi sebenarnya lebih kecil dari n_samples, jangan
    # infinite-loop -- cap ke ukuran combinatorial space.
    space_size = 1
    for k in keys:
        space_size *= len(param_grid[k])
    n_samples = min(n_samples, space_size)

    while len(pool) < n_samples:
        combo = {k: rng.choice(param_grid[k]) for k in keys}
        # layer_sizes/lists tidak hashable -> pakai repr sebagai dedup key
        combo_key = repr(sorted(combo.items(), key=lambda kv: kv[0]))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        pool.append(combo)

    return pool


# =====================================================================
# GRID DEFINITIONS (tanpa 'epochs' -- itu jadi resource budget halving)
# =====================================================================

def get_search_space_pascabayar(n_features: int) -> dict:
    return {
        "layer_sizes": [
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 128, 1],
            [n_features, 64, 32, 1],
            [n_features, 128, 64, 1],
            [n_features, 128, 64, 32, 1],
            [n_features, 256, 128, 64, 1],
        ],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [1.0, 3.0, 5.0, 10.0],
        "batch_size": [16, 32, 64, 128],
    }


def get_search_space_prabayar(n_features: int) -> dict:
    return {
        "layer_sizes": [
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 128, 1],
            [n_features, 64, 32, 1],
            [n_features, 128, 64, 1],
        ],
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [5.0, 10.0, 15.0],
        "batch_size": [16, 32, 64, 128],
    }


def get_search_space_pv() -> dict:
    return {
        "component_activation": ["relu", "linear"],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [5.0, 10.0, 15.0],
        "batch_size": [16, 32, 64, 128],
    }


# =====================================================================
# TRAINER: MLP DENGAN VAL-BASED EARLY STOPPING
# =====================================================================

def train_mlp_early_stop(
    model,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    learning_rate: float, max_epochs: int, batch_size: int,
    patience: int = 15, check_every: int = 5,
):
    """
    Latih MLP sampai max_epochs, tapi checkpoint bobot terbaik (val RMSE
    scaled terkecil) dan berhenti lebih awal kalau tidak ada perbaikan
    selama `patience` epoch. Mengembalikan MODEL TERBAIK, bukan model
    di epoch terakhir.
    """
    X = np.array(x_train, dtype=np.float32)
    Y = np.array(y_train_scaled, dtype=np.float32).reshape(-1, 1)
    Yv = np.array(y_val_scaled, dtype=np.float64)
    n = X.shape[0]

    best_val = float("inf")
    best_model = None
    best_epoch = 0
    no_improve = 0
    diverged = False
    loss_start = None

    for ep in range(max_epochs):
        idx = np.random.permutation(n)
        X_shuf, Y_shuf = X[idx], Y[idx]

        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            loss = model.train_batch(X_shuf[start:end], Y_shuf[start:end], learning_rate)
            epoch_losses.append(loss)
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if ep == 0:
            loss_start = epoch_loss
        if is_invalid_number(epoch_loss):
            diverged = True
            break

        if ep % check_every == 0 or ep == max_epochs - 1:
            val_preds = predict_mlp(model, x_val).astype(np.float64)
            if not np.all(np.isfinite(val_preds)):
                diverged = True
                break
            val_rmse = compute_rmse(Yv, val_preds)
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_model = copy.deepcopy(model)
                best_epoch = ep
                no_improve = 0
            else:
                no_improve += check_every
                if no_improve >= patience:
                    break

    return {
        "model": best_model if best_model is not None else model,
        "best_val_rmse_scaled": best_val,
        "best_epoch": best_epoch,
        "diverged": diverged,
        "loss_start": loss_start,
    }


def train_pv_early_stop(
    model: PascabayarPlaceValueModel,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    learning_rate: float, max_epochs: int, batch_size: int,
    patience: int = 10, check_every: int = 3,
):
    """Sama seperti train_mlp_early_stop tapi untuk model per-sample (PV)."""
    n = len(x_train)
    indices_all = list(range(n))
    Yv = np.array(y_val_scaled, dtype=np.float64)

    best_val = float("inf")
    best_model = None
    best_epoch = 0
    no_improve = 0
    diverged = False
    loss_start = None

    for ep in range(max_epochs):
        np.random.shuffle(indices_all)
        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in indices_all[start:end]:
                x_i = x_train[i] if isinstance(x_train[i], list) else list(x_train[i])
                y_i = float(y_train_scaled[i])
                loss = model.train_one_sample(x_i, y_i, learning_rate)
                epoch_losses.append(loss)
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if ep == 0:
            loss_start = epoch_loss
        if is_invalid_number(epoch_loss) or not np.isfinite(np.sum(model.w2)) or not np.isfinite(model.b2):
            diverged = True
            break

        if ep % check_every == 0 or ep == max_epochs - 1:
            val_preds = predict_pv(model, x_val).astype(np.float64)
            if not np.all(np.isfinite(val_preds)):
                diverged = True
                break
            val_rmse = compute_rmse(Yv, val_preds)
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_model = copy.deepcopy(model)
                best_epoch = ep
                no_improve = 0
            else:
                no_improve += check_every
                if no_improve >= patience:
                    break

    return {
        "model": best_model if best_model is not None else model,
        "best_val_rmse_scaled": best_val,
        "best_epoch": best_epoch,
        "diverged": diverged,
        "loss_start": loss_start,
    }


# =====================================================================
# WORKER FUNCTIONS (top-level supaya picklable oleh ProcessPoolExecutor)
# =====================================================================

def _eval_mlp_candidate(args):
    (ModelClass, params, epoch_budget, patience,
     x_train, y_train_scaled, x_val, y_val_scaled, y_val_original, y_scaler) = args

    np.random.seed(42)
    model = ModelClass(
        layer_sizes=list(params["layer_sizes"]),
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
    )

    info = train_mlp_early_stop(
        model, x_train, y_train_scaled, x_val, y_val_scaled,
        params["learning_rate"], epoch_budget, params["batch_size"],
        patience=patience,
    )

    if info["diverged"]:
        m = make_divergent_metrics(params, info["loss_start"])
        m["best_epoch"] = info["best_epoch"]
        return m

    preds_scaled = predict_mlp(info["model"], x_val)
    preds_original = np.array([inverse_transform_target(float(p), y_scaler) for p in preds_scaled])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    metrics["diverged"] = False
    metrics["best_epoch"] = info["best_epoch"]
    metrics["epoch_budget"] = epoch_budget
    return metrics


def _eval_pv_candidate(args):
    (params, n_features, epoch_budget, patience,
     x_train, y_train_scaled, x_val, y_val_scaled, y_val_original, y_scaler) = args

    np.random.seed(42)
    random.seed(42)
    model = PascabayarPlaceValueModel(
        input_size=n_features,
        hidden_size=7,
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
        component_activation=params["component_activation"],
    )

    info = train_pv_early_stop(
        model, x_train, y_train_scaled, x_val, y_val_scaled,
        params["learning_rate"], epoch_budget, params["batch_size"],
        patience=patience,
    )

    if info["diverged"]:
        m = make_divergent_metrics(params, info["loss_start"])
        m["best_epoch"] = info["best_epoch"]
        return m

    preds_scaled = predict_pv(info["model"], x_val)
    preds_original = np.array([inverse_transform_target(float(p), y_scaler) for p in preds_scaled])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    metrics["diverged"] = False
    metrics["best_epoch"] = info["best_epoch"]
    metrics["epoch_budget"] = epoch_budget
    return metrics


# =====================================================================
# SUCCESSIVE HALVING DRIVER
# =====================================================================

@dataclass
class HalvingConfig:
    n_initial_candidates: int = 200      # ukuran random search pool awal
    rung_epochs: tuple = (30, 90, 270)   # resource budget tiap rung
    eta: int = 3                         # reduction factor tiap rung
    patience_per_rung: tuple = (10, 15, 25)
    n_workers: int | None = None         # None = pakai semua core CPU
    seed: int = 42


def run_successive_halving(
    eval_fn, worker_arg_builder, param_pool: list[dict], cfg: HalvingConfig,
):
    """
    eval_fn: _eval_mlp_candidate atau _eval_pv_candidate
    worker_arg_builder: fungsi (params, epoch_budget, patience) -> tuple args
                        yang cocok dengan signature eval_fn
    """
    candidates = param_pool
    last_results = []

    for rung_idx, epoch_budget in enumerate(cfg.rung_epochs):
        patience = cfg.patience_per_rung[min(rung_idx, len(cfg.patience_per_rung) - 1)]
        t0 = time.time()
        print(f"\n  Rung {rung_idx + 1}/{len(cfg.rung_epochs)}: "
              f"{len(candidates)} kandidat, budget={epoch_budget} epoch, patience={patience}")

        work_items = [worker_arg_builder(p, epoch_budget, patience) for p in candidates]

        results = []
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            futures = [ex.submit(eval_fn, item) for item in work_items]
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda r: (r["rmse"], r["mae"]))
        n_diverged = sum(1 for r in results if r.get("diverged"))
        elapsed = time.time() - t0

        print(f"    Selesai dalam {elapsed:.1f}s | divergen: {n_diverged}/{len(results)}")
        if results and not results[0].get("diverged"):
            print(f"    Best RMSE rung ini: {results[0]['rmse']:.4f} "
                  f"(best_epoch={results[0].get('best_epoch')})")

        keep_n = max(1, math.ceil(len(results) / cfg.eta))
        candidates = [r["params"] for r in results[:keep_n]]
        last_results = results

    return last_results  # hasil rung terakhir, sudah ranked terbaik->terjelek


# =====================================================================
# RUNNER PER MODEL
# =====================================================================

def tune_pascabayar(x_train, y_train, x_val, y_val, y_val_orig, n_features, y_scaler, cfg: HalvingConfig):
    space = get_search_space_pascabayar(n_features)
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (PascabayarModel, params, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_mlp_candidate, build_args, pool, cfg)


def tune_prabayar(x_train, y_train, x_val, y_val, y_val_orig, n_features, y_scaler, cfg: HalvingConfig):
    space = get_search_space_prabayar(n_features)
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (PrabayarModel, params, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_mlp_candidate, build_args, pool, cfg)


def tune_pv(x_train, y_train, x_val, y_val, y_val_orig, n_features, y_scaler, cfg: HalvingConfig):
    space = get_search_space_pv()
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (params, n_features, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_pv_candidate, build_args, pool, cfg)


# =====================================================================
# MAIN
# =====================================================================

def main():
    cfg = HalvingConfig(
        n_initial_candidates=200,
        rung_epochs=(30, 90, 270),
        eta=3,
        patience_per_rung=(10, 15, 25),
        n_workers=None,  # None -> os.cpu_count()
        seed=42,
    )

    print("\n" + "▓" * 70)
    print("  LOADING DATA: Pascabayar")
    print("▓" * 70)
    (x_tr_p, x_va_p, y_tr_p, y_va_p, y_va_orig_p, n_feat_p, yscaler_p, _) = load_data("pascabayar")
    print(f"  Train: {len(x_tr_p)} | Val: {len(x_va_p)} | Fitur: {n_feat_p}")
    results_pasca = tune_pascabayar(x_tr_p, y_tr_p, x_va_p, y_va_p, y_va_orig_p, n_feat_p, yscaler_p, cfg)

    print("\n" + "▓" * 70)
    print("  LOADING DATA: Prabayar")
    print("▓" * 70)
    (x_tr_r, x_va_r, y_tr_r, y_va_r, y_va_orig_r, n_feat_r, yscaler_r, _) = load_data("prabayar")
    print(f"  Train: {len(x_tr_r)} | Val: {len(x_va_r)} | Fitur: {n_feat_r}")
    results_pra = tune_prabayar(x_tr_r, y_tr_r, x_va_r, y_va_r, y_va_orig_r, n_feat_r, yscaler_r, cfg)

    print("\n" + "▓" * 70)
    print("  TUNING: PascabayarPlaceValueModel (reuse data Pascabayar)")
    print("▓" * 70)
    results_pv = tune_pv(x_tr_p, y_tr_p, x_va_p, y_va_p, y_va_orig_p, n_feat_p, yscaler_p, cfg)

    print("\n" + "▓" * 70)
    print("  RINGKASAN")
    print("▓" * 70)
    for name, res in [("PascabayarModel", results_pasca), ("PrabayarModel", results_pra),
                       ("PascabayarPlaceValueModel", results_pv)]:
        valid = [r for r in res if not r.get("diverged")]
        print(f"\n  {name}: {len(valid)}/{len(res)} konvergen di rung terakhir")
        if valid:
            best = valid[0]
            print(f"    RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  "
                  f"MAPE={best['mape']:.4f}%  R²={best['r2']:.6f}  "
                  f"best_epoch={best.get('best_epoch')}")
            print(f"    Params: {best['params']}")


if __name__ == "__main__":
    main()