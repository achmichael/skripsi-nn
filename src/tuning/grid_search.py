"""
grid_search.py (Successive Halving / Random Search)

Hyperparameter tuning untuk model Prabayar.
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tuning.train_tuning import (
    compute_rmse, compute_mae, compute_mape, compute_r2, evaluate_metrics,
    is_invalid_number, make_divergent_metrics,
    load_data, predict_mlp,
)

from src.models.prabayar import PrabayarModel
from src.pipeline.preprocessing import inverse_transform_target


# =====================================================================
# RANDOM SEARCH SAMPLING
# =====================================================================

def sample_param_pool(param_grid: dict, n_samples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    seen = set()
    pool = []

    space_size = 1
    for k in keys:
        space_size *= len(param_grid[k])
    n_samples = min(n_samples, space_size)

    while len(pool) < n_samples:
        combo = {k: rng.choice(param_grid[k]) for k in keys}
        combo_key = repr(sorted(combo.items(), key=lambda kv: kv[0]))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        pool.append(combo)

    return pool


# =====================================================================
# GRID DEFINITION — PRABAYAR ONLY
# =====================================================================

def get_search_space_prabayar(n_features: int) -> dict:
    return {
        "layer_sizes": [
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 128, 1],
            [n_features, 64, 32, 1],
            [n_features, 64, 48, 1],
            [n_features, 64, 16, 1],
            [n_features, 48, 32, 1],
            [n_features, 48, 16, 1],
            [n_features, 32, 16, 1],
            [n_features, 128, 64, 1],
        ],
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3],
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
            loss = model.train_batch(X_shuf[start:end], None, Y_shuf[start:end], learning_rate)
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


# =====================================================================
# WORKER FUNCTION
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


# =====================================================================
# SUCCESSIVE HALVING DRIVER
# =====================================================================

@dataclass
class HalvingConfig:
    n_initial_candidates: int = 200
    rung_epochs: tuple = (30, 90, 270)
    eta: int = 3
    patience_per_rung: tuple = (10, 15, 25)
    n_workers: int | None = None
    seed: int = 42


def run_successive_halving(
    eval_fn, worker_arg_builder, param_pool: list[dict], cfg: HalvingConfig,
):
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

    return last_results


# =====================================================================
# RUNNER
# =====================================================================

def tune_prabayar(x_train, y_train, x_val, y_val, y_val_orig, n_features, y_scaler, cfg: HalvingConfig):
    space = get_search_space_prabayar(n_features)
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (PrabayarModel, params, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_mlp_candidate, build_args, pool, cfg)


# =====================================================================
# MAIN
# =====================================================================

def main():
    cfg = HalvingConfig(
        n_initial_candidates=200,
        rung_epochs=(30, 90, 270),
        eta=3,
        patience_per_rung=(10, 15, 25),
        n_workers=None,
        seed=42,
    )

    print("\n" + "▓" * 70)
    print("  LOADING DATA: Prabayar")
    print("▓" * 70)
    (x_tr_r, x_va_r, y_tr_r, y_va_r, y_va_orig_r, n_feat_r, yscaler_r, _) = load_data("prabayar")
    print(f"  Train: {len(x_tr_r)} | Val: {len(x_va_r)} | Fitur: {n_feat_r}")
    results_pra = tune_prabayar(x_tr_r, y_tr_r, x_va_r, y_va_r, y_va_orig_r, n_feat_r, yscaler_r, cfg)

    print("\n" + "▓" * 70)
    print("  RINGKASAN")
    print("▓" * 70)
    valid = [r for r in results_pra if not r.get("diverged")]
    print(f"\n  PrabayarModel: {len(valid)}/{len(results_pra)} konvergen di rung terakhir")
    if valid:
        best = valid[0]
        print(f"    RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  "
              f"MAPE={best['mape']:.4f}%  R²={best['r2']:.6f}  "
              f"best_epoch={best.get('best_epoch')}")
        print(f"    Params: {best['params']}")


if __name__ == "__main__":
    main()
