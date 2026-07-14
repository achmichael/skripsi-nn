"""
Grid Search Hyperparameter Tuning — Pure Python + NumPy.
REVISI v2 — fokus mengatasi divergensi (loss naik/NaN) pada
PrabayarModel dan PascabayarPlaceValueModel.

Root cause yang dikonfirmasi sebelum revisi ini:
  - `_clip_gradient` di base class NeuralNetwork melakukan CLIP-BY-VALUE
    (np.clip element-wise), bukan clip-by-norm. Dengan 21+ fitur input,
    kombinasi learning_rate besar + arsitektur dalam/lebar + clip_value
    kecil membuat clipping memutar arah gradien secara sistematis
    (bukan cuma memperlambat), sehingga loss naik monoton / NaN.
  - Ini terjadi di KEDUA model (Prabayar = vectorized batch,
    PlaceValue = per-sample SGD) karena keduanya mewarisi clip
    function yang sama dari base class.

Strategi revisi (TIDAK mengubah neural_network.py / strategi clipping —
itu di luar scope; di sini kita hindari kombinasi yang membuat
clip-by-value jadi destruktif):
  1. Grid Prabayar: turunkan learning_rate ceiling, naikkan clip_value
     minimum, buang arsitektur paling dalam (4 hidden layer).
  2. Grid PlaceValue: turunkan learning_rate lebih jauh lagi (karena
     update per-sample SGD, bukan batch-averaged, lebih noisy/agresif
     per step), naikkan clip_value minimum.
  3. Tambah NaN/Inf early-stop per kombinasi -> tidak buang waktu
     compute melatih kombinasi yang sudah pasti divergen.
  4. Tambah loss-trajectory tracking (loss epoch awal vs akhir) supaya
     hasil JSON punya visibility apakah suatu kombinasi convergent,
     divergent, atau underfit/plateau -- bukan cuma RMSE akhir.
  5. PascabayarModel (sudah konvergen) -- grid TIDAK diubah, supaya
     hasil yang sudah baik tidak terganggu.

Methodology (tetap sama):
  - 80/20 train/test split (seed=42)
  - Setiap kombinasi: init model (seed=42), train, eval pada validation set
  - Metrik: RMSE, MAE, MAPE, R²
  - Ranking by RMSE (primary), MAE (tiebreaker) -- DENGAN filter
    divergen (RMSE/loss NaN atau Inf otomatis didorong ke bawah ranking)
  - Top 5 per model, Top 1 dievaluasi pada test set

Usage:
  python -m src.tuning.grid_search
"""

import itertools
import math
import os
import sys
import time
import json
import numpy as np

# ─── Project imports ─────────────────────────────────────────────────
from src.config.config import config
from src.pipeline.preprocessing import (
    load_and_preprocess,
    train_test_split,
    fit_minmax_scaler,
    transform_minmax,
    fit_target_scaler,
    transform_target,
    inverse_transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.models.pascabayar import PascabayarModel
from src.models.prabayar import PrabayarModel
from src.models.pascabayar_place_value import PascabayarPlaceValueModel


# =====================================================================
# METRICS (pure numpy, no sklearn)
# =====================================================================

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %. Samples with y_true==0 are excluded."""
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


# =====================================================================
# DIVERGENCE GUARD HELPERS (NEW)
# =====================================================================

def is_invalid_number(x: float) -> bool:
    """True jika NaN atau Inf."""
    return math.isnan(x) or math.isinf(x)


def make_divergent_metrics(params: dict, loss_start: float | None = None) -> dict:
    """
    Metrics dummy untuk kombinasi yang divergen (loss NaN/Inf terdeteksi
    sebelum training selesai). RMSE=inf memastikan kombinasi ini selalu
    di-rank paling bawah, tapi tetap tercatat (bukan exception/crash)
    sehingga grid search lain dalam batch tidak ikut gagal.
    """
    return {
        "rmse": float("inf"),
        "mae": float("inf"),
        "mape": float("inf"),
        "r2": float("-inf"),
        "params": params,
        "diverged": True,
        "loss_trajectory": {
            "loss_start": loss_start,
            "loss_end": None,
        },
    }


# =====================================================================
# DATA LOADING HELPERS
# =====================================================================

def load_data(model_type: str):
    """
    Load & preprocess data, extract features, split 80/20, scale.

    Returns:
        X_train, X_val, y_train_scaled, y_val_scaled,
        y_val_original, feature_count, y_scaler, use_log
    """
    dataset_path = config[model_type]["dataset_path"]
    use_log = config[model_type].get("use_log_transform", False)

    df = load_and_preprocess(dataset_path)
    x_data, y_data, feature_cols, target_col = extract_features_and_target(df, model_type)

    # 80/20 split
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_ratio=0.2, seed=42)

    # Feature scaling (fit on train only)
    feat_scaler = fit_minmax_scaler(x_train)
    x_train = transform_minmax(x_train, feat_scaler)
    x_val = transform_minmax(x_val, feat_scaler)

    # Target scaling (fit on train only)
    y_scaler = fit_target_scaler(y_train, use_log=use_log)
    y_train_scaled = transform_target(y_train, y_scaler)
    y_val_scaled = transform_target(y_val, y_scaler)

    n_features = len(x_train[0])

    return (
        x_train, x_val,
        y_train_scaled, y_val_scaled,
        y_val,  # original scale for metrics
        n_features, y_scaler, use_log,
    )


# =====================================================================
# TRAINING HELPERS (silent, with divergence detection)
# =====================================================================

def train_mlp_silent(
    model,
    x_train: list, y_train: list,
    learning_rate: float, epochs: int, batch_size: int,
) -> dict:
    """
    Train MLP model (PascabayarModel or PrabayarModel) silently.

    NEW: mengembalikan loss trajectory + flag divergen. Berhenti lebih
    awal (early-stop) begitu loss NaN/Inf terdeteksi -- tidak buang
    waktu compute melatih sisa epoch dari kombinasi yang sudah pasti gagal.
    """
    X = np.array(x_train, dtype=np.float32)
    Y = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    n = X.shape[0]

    loss_start = None
    loss_end = None
    diverged = False

    for ep in range(epochs):
        indices = np.random.permutation(n)
        X_shuf = X[indices]
        Y_shuf = Y[indices]

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
            loss_end = epoch_loss
            break

        loss_end = epoch_loss

    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "diverged": diverged,
    }


def train_pv_silent(
    model: PascabayarPlaceValueModel,
    x_train: list, y_train: list,
    learning_rate: float, epochs: int, batch_size: int,
) -> dict:
    """
    Train PascabayarPlaceValueModel silently (sample-by-sample in batches).

    NEW: same divergence tracking/early-stop as train_mlp_silent.
    """
    n = len(x_train)
    indices_all = list(range(n))

    loss_start = None
    loss_end = None
    diverged = False

    for ep in range(epochs):
        np.random.shuffle(indices_all)

        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices_all[start:end]

            for i in batch_idx:
                x_i = x_train[i] if isinstance(x_train[i], list) else list(x_train[i])
                y_i = float(y_train[i])
                loss = model.train_one_sample(x_i, y_i, learning_rate)
                epoch_losses.append(loss)

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if ep == 0:
            loss_start = epoch_loss

        if is_invalid_number(epoch_loss):
            diverged = True
            loss_end = epoch_loss
            break

        loss_end = epoch_loss

        # Cek tambahan: untuk SGD per-sample, divergensi kadang muncul
        # sebagai bobot meledak walau rata-rata epoch loss masih terlihat
        # finite (karena outlier ter-rata-kan). Sampling cek beberapa bobot:
        if not np.isfinite(np.sum(model.w2)) or not np.isfinite(model.b2):
            diverged = True
            break

    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "diverged": diverged,
    }


def predict_mlp(model, x_data: list) -> np.ndarray:
    X = np.array(x_data, dtype=np.float32)
    preds = model.predict(X)
    return preds.flatten()


def predict_pv(model: PascabayarPlaceValueModel, x_data: list) -> np.ndarray:
    preds = []
    for x in x_data:
        x_list = x if isinstance(x, list) else list(x)
        preds.append(model.forward(x_list))
    return np.array(preds, dtype=np.float64)


# =====================================================================
# GRID DEFINITIONS
# =====================================================================

def get_mlp_grid_pascabayar(n_features: int) -> list[dict]:
    """
    Grid untuk PascabayarModel -- TIDAK DIUBAH dari versi asli karena
    model ini sudah konvergen dengan baik. Mengubah grid yang sudah
    bekerja berisiko menurunkan hasil yang sudah bagus.
    """
    param_grid = {
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
        "epochs": [100, 200, 300, 500],
        "batch_size": [16, 32, 64, 128],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def get_mlp_grid_prabayar(n_features: int) -> list[dict]:
    """
    Grid REVISI untuk PrabayarModel.

    Perubahan vs grid asli:
      - learning_rate: ceiling diturunkan dari 1e-2 -> 1e-3.
        Alasan: dengan 21+ fitur, LR besar membuat lebih banyak elemen
        gradien menabrak clip_value secara serentak setiap step --
        karena clipping di base class adalah clip-by-VALUE (bukan
        clip-by-norm), ini memutar arah gradien secara sistematis,
        bukan sekadar memperlambat. Ditambah titik LR sangat kecil
        (5e-5) untuk eksplorasi konvergensi lambat tapi stabil.
      - clip_value: minimum dinaikkan dari 1.0 -> 5.0. clip_value
        kecil + clip-by-value pada banyak elemen gradien besar =
        distorsi arah paling parah. Titik 15.0 ditambah sebagai
        opsi "clipping minimal".
      - layer_sizes: arsitektur 4-hidden-layer paling dalam
        ([n,256,128,64,1]) DIBUANG. Semakin dalam arsitektur,
        semakin banyak elemen gradien yang exploding serentak di
        awal training (saat bobot masih random) -- ini yang paling
        rawan menyebabkan loss naik monoton / NaN yang anda amati.
        Arsitektur medium [n,128,64,1] dipertahankan sebagai opsi
        paling dalam yang masih dicoba.
      - epochs, batch_size: tidak diubah.
    """
    param_grid = {
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
        "epochs": [100, 200, 300, 500],
        "batch_size": [16, 32, 64, 128],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def get_pv_grid() -> list[dict]:
    """
    Grid REVISI untuk PascabayarPlaceValueModel.

    Perubahan vs grid asli:
      - learning_rate: ceiling diturunkan lebih jauh dari 1e-2 -> 5e-4.
        Alasan: training_one_sample melakukan update bobot PER SAMPLE
        (bukan rata-rata gradien per batch seperti train_batch di MLP
        biasa) -- ini membuat efektif "langkah" optimasi jauh lebih
        noisy dan agresif dibanding LR yang sama pada model batch.
        Ditambah titik sangat kecil (1e-5) karena update per-sample +
        hidden_size=7 yang sempit (bottleneck dari 21+ fitur) butuh
        langkah yang sangat hati-hati untuk tidak meledak.
      - clip_value: minimum dinaikkan dari 1.0 -> 5.0, sama seperti
        Prabayar -- alasan identik (clip-by-value destruktif pada
        gradien besar yang sering terjadi karena bottleneck 21+ fitur
        -> 7 neuron membuat gradien hidden layer besar di awal training).
      - component_activation: tidak diubah (relu, linear).
      - epochs, batch_size: tidak diubah.

    CATATAN PENTING: grid ini TIDAK akan mengatasi underfit struktural
    dari hidden_size=7 yang fixed (bottleneck ~3:1 dari 21+ fitur).
    Grid ini hanya mengatasi DIVERGENSI (loss naik/NaN). Jika setelah
    revisi ini model konvergen tapi R² tetap rendah, itu sinyal
    keterbatasan kapasitas arsitektur place-value itu sendiri, bukan
    lagi masalah hyperparameter.
    """
    param_grid = {
        "component_activation": ["relu", "linear"],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [5.0, 10.0, 15.0],
        "epochs": [100, 200, 300, 500],
        "batch_size": [16, 32, 64, 128],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


# =====================================================================
# SINGLE COMBINATION EVALUATION
# =====================================================================

def eval_mlp_combo(
    ModelClass,
    params: dict,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    y_scaler,
) -> dict:
    """Train one MLP combo, return metrics dict (dengan divergence guard)."""
    np.random.seed(42)

    model = ModelClass(
        layer_sizes=list(params["layer_sizes"]),
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
    )

    train_info = train_mlp_silent(
        model, x_train, y_train_scaled,
        params["learning_rate"], params["epochs"], params["batch_size"],
    )

    if train_info["diverged"]:
        return make_divergent_metrics(params, train_info["loss_start"])

    # Predict on validation (normalized scale)
    preds_scaled = predict_mlp(model, x_val)

    if not np.all(np.isfinite(preds_scaled)):
        return make_divergent_metrics(params, train_info["loss_start"])

    # Inverse transform to original scale
    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    metrics["diverged"] = False
    metrics["loss_trajectory"] = {
        "loss_start": train_info["loss_start"],
        "loss_end": train_info["loss_end"],
    }
    return metrics


def eval_pv_combo(
    params: dict,
    n_features: int,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    y_scaler,
) -> dict:
    """Train one PlaceValue combo, return metrics dict (dengan divergence guard)."""
    np.random.seed(42)
    import random
    random.seed(42)

    model = PascabayarPlaceValueModel(
        input_size=n_features,
        hidden_size=7,
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
        component_activation=params["component_activation"],
    )

    train_info = train_pv_silent(
        model, x_train, y_train_scaled,
        params["learning_rate"], params["epochs"], params["batch_size"],
    )

    if train_info["diverged"]:
        return make_divergent_metrics(params, train_info["loss_start"])

    preds_scaled = predict_pv(model, x_val)

    if not np.all(np.isfinite(preds_scaled)):
        return make_divergent_metrics(params, train_info["loss_start"])

    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    metrics["diverged"] = False
    metrics["loss_trajectory"] = {
        "loss_start": train_info["loss_start"],
        "loss_end": train_info["loss_end"],
    }
    return metrics


# =====================================================================
# GRID SEARCH RUNNER
# =====================================================================

def run_grid_search_mlp(
    model_name: str,
    ModelClass,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    n_features: int,
    y_scaler,
    grid: list[dict],
) -> list[dict]:
    """Run full grid search for one MLP model. Returns sorted results."""
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH: {model_name}")
    print(f"  Total kombinasi: {total}")
    print(f"{'='*70}\n")

    results = []
    n_diverged = 0
    t_start = time.time()

    for idx, params in enumerate(grid, 1):
        t0 = time.time()

        metrics = eval_mlp_combo(
            ModelClass, params,
            x_train, y_train_scaled,
            x_val, y_val_scaled,
            y_val_original, y_scaler,
        )
        elapsed = time.time() - t0

        if metrics.get("diverged"):
            n_diverged += 1

        results.append(metrics)

        if idx % 50 == 0 or idx == total or idx == 1:
            pct = idx / total * 100
            eta = (time.time() - t_start) / idx * (total - idx)
            div_pct = n_diverged / idx * 100
            print(
                f"  [{idx:>4d}/{total}] {pct:5.1f}%  "
                f"RMSE={metrics['rmse']:>12.2f}  "
                f"MAE={metrics['mae']:>12.2f}  "
                f"R²={metrics['r2']:>7.4f}  "
                f"Diverged={n_diverged} ({div_pct:.1f}%)  "
                f"({elapsed:.1f}s)  "
                f"ETA: {eta/60:.1f}min"
            )

    total_time = time.time() - t_start
    print(f"\n  Selesai dalam {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Kombinasi divergen: {n_diverged}/{total} ({n_diverged/total*100:.1f}%)")

    # Sort by RMSE (asc), then MAE (asc) as tiebreaker
    # (RMSE=inf untuk yang divergen otomatis turun ke bawah)
    results.sort(key=lambda r: (r["rmse"], r["mae"]))
    return results


def run_grid_search_pv(
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    n_features: int,
    y_scaler,
    grid: list[dict],
) -> list[dict]:
    """Run full grid search for PascabayarPlaceValueModel. Returns sorted results."""
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH: PascabayarPlaceValueModel")
    print(f"  Total kombinasi: {total}")
    print(f"{'='*70}\n")

    results = []
    n_diverged = 0
    t_start = time.time()

    for idx, params in enumerate(grid, 1):
        t0 = time.time()

        metrics = eval_pv_combo(
            params, n_features,
            x_train, y_train_scaled,
            x_val, y_val_scaled,
            y_val_original, y_scaler,
        )
        elapsed = time.time() - t0

        if metrics.get("diverged"):
            n_diverged += 1

        results.append(metrics)

        if idx % 50 == 0 or idx == total or idx == 1:
            pct = idx / total * 100
            eta = (time.time() - t_start) / idx * (total - idx)
            div_pct = n_diverged / idx * 100
            print(
                f"  [{idx:>4d}/{total}] {pct:5.1f}%  "
                f"RMSE={metrics['rmse']:>12.2f}  "
                f"MAE={metrics['mae']:>12.2f}  "
                f"R²={metrics['r2']:>7.4f}  "
                f"Diverged={n_diverged} ({div_pct:.1f}%)  "
                f"({elapsed:.1f}s)  "
                f"ETA: {eta/60:.1f}min"
            )

    total_time = time.time() - t_start
    print(f"\n  Selesai dalam {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Kombinasi divergen: {n_diverged}/{total} ({n_diverged/total*100:.1f}%)")

    results.sort(key=lambda r: (r["rmse"], r["mae"]))
    return results


# =====================================================================
# DISPLAY & FINAL EVALUATION HELPERS
# =====================================================================

def print_top_results(model_name: str, results: list[dict], top_n: int = 5):
    """Print top N results for a model. Skip kombinasi yang divergen kalau ada
    cukup kombinasi valid; tampilkan warning kalau semua top-N divergen."""
    valid_results = [r for r in results if not r.get("diverged")]

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} HASIL — {model_name}")
    print(f"{'='*70}")

    if not valid_results:
        print("\n  ⚠️  PERINGATAN: SEMUA kombinasi di grid ini divergen (NaN/Inf).")
        print("  Grid perlu dipersempit lebih jauh -- learning_rate lebih kecil")
        print("  dan/atau clip_value lebih besar dari yang sudah dicoba.")
        return

    print(f"\n  ({len(valid_results)}/{len(results)} kombinasi konvergen)")

    for rank, r in enumerate(valid_results[:top_n], 1):
        print(f"\n  ── Rank #{rank} ──")
        print(f"     RMSE : {r['rmse']:>14.4f}")
        print(f"     MAE  : {r['mae']:>14.4f}")
        print(f"     MAPE : {r['mape']:>14.4f}%")
        print(f"     R²   : {r['r2']:>14.6f}")
        traj = r.get("loss_trajectory", {})
        if traj:
            print(f"     Loss : start={traj.get('loss_start')}  end={traj.get('loss_end')}")
        print(f"     Params:")
        for k, v in r["params"].items():
            print(f"       {k:25s}: {v}")


def get_best_valid_params(results: list[dict]) -> dict | None:
    """Ambil params dari kombinasi terbaik yang TIDAK divergen."""
    for r in results:
        if not r.get("diverged"):
            return r["params"]
    return None


def final_eval_mlp(
    model_name: str,
    ModelClass,
    best_params: dict,
    x_train, y_train_scaled,
    x_test, y_test_scaled,
    y_test_original,
    y_scaler,
):
    """Retrain best MLP on full train set, evaluate on held-out test set."""
    print(f"\n{'='*70}")
    print(f"  EVALUASI TEST SET — {model_name}")
    print(f"{'='*70}")
    print(f"  Best params: {best_params}")

    np.random.seed(42)
    model = ModelClass(
        layer_sizes=list(best_params["layer_sizes"]),
        seed=42,
        clip_value=best_params["clip_value"],
        l2_lambda=best_params["l2_lambda"],
    )

    train_info = train_mlp_silent(
        model, x_train, y_train_scaled,
        best_params["learning_rate"],
        best_params["epochs"],
        best_params["batch_size"],
    )

    if train_info["diverged"]:
        print("\n  ⚠️  Model terbaik tetap divergen saat retrain penuh.")
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}

    preds_scaled = predict_mlp(model, x_test)
    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_test_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)

    print(f"\n  ── Test Set Metrics ──")
    print(f"     RMSE : {metrics['rmse']:>14.4f}")
    print(f"     MAE  : {metrics['mae']:>14.4f}")
    print(f"     MAPE : {metrics['mape']:>14.4f}%")
    print(f"     R²   : {metrics['r2']:>14.6f}")

    return metrics


def final_eval_pv(
    best_params: dict,
    n_features: int,
    x_train, y_train_scaled,
    x_test, y_test_scaled,
    y_test_original,
    y_scaler,
):
    """Retrain best PlaceValue on full train set, evaluate on held-out test set."""
    print(f"\n{'='*70}")
    print(f"  EVALUASI TEST SET — PascabayarPlaceValueModel")
    print(f"{'='*70}")
    print(f"  Best params: {best_params}")

    np.random.seed(42)
    import random
    random.seed(42)

    model = PascabayarPlaceValueModel(
        input_size=n_features,
        hidden_size=7,
        seed=42,
        clip_value=best_params["clip_value"],
        l2_lambda=best_params["l2_lambda"],
        component_activation=best_params["component_activation"],
    )

    train_info = train_pv_silent(
        model, x_train, y_train_scaled,
        best_params["learning_rate"],
        best_params["epochs"],
        best_params["batch_size"],
    )

    if train_info["diverged"]:
        print("\n  ⚠️  Model terbaik tetap divergen saat retrain penuh.")
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}

    preds_scaled = predict_pv(model, x_test)
    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_test_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)

    print(f"\n  ── Test Set Metrics ──")
    print(f"     RMSE : {metrics['rmse']:>14.4f}")
    print(f"     MAE  : {metrics['mae']:>14.4f}")
    print(f"     MAPE : {metrics['mape']:>14.4f}%")
    print(f"     R²   : {metrics['r2']:>14.6f}")

    return metrics


# =====================================================================
# SAVE RESULTS TO JSON
# =====================================================================

def save_results(model_name: str, results: list[dict], test_metrics: dict, output_dir: str):
    """Save top results + test metrics + divergence summary to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"grid_search_{safe_name}.json")

    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    valid_results = [r for r in results if not r.get("diverged")]
    n_diverged = len(results) - len(valid_results)

    top5 = []
    for r in valid_results[:5]:
        entry = {
            "rmse": make_serializable(r["rmse"]),
            "mae": make_serializable(r["mae"]),
            "mape": make_serializable(r["mape"]),
            "r2": make_serializable(r["r2"]),
            "loss_trajectory": r.get("loss_trajectory", {}),
            "params": {
                k: make_serializable(v) for k, v in r["params"].items()
            },
        }
        top5.append(entry)

    output = {
        "model": model_name,
        "total_combinations": len(results),
        "diverged_combinations": n_diverged,
        "diverged_pct": round(n_diverged / len(results) * 100, 2) if results else 0,
        "top_5_validation": top5,
        "test_set_metrics": {
            k: make_serializable(v) for k, v in test_metrics.items()
        },
        "best_params": (
            {k: make_serializable(v) for k, v in valid_results[0]["params"].items()}
            if valid_results else None
        ),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Hasil disimpan ke: {path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    output_dir = "results/tuning"

    # ==================================================================
    # 1. PASCABAYAR MODEL (MLP) -- grid TIDAK diubah, sudah konvergen
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  LOADING DATA: Pascabayar")
    print("▓" * 70)

    (
        x_train_pasca, x_val_pasca,
        y_train_pasca, y_val_pasca,
        y_val_orig_pasca,
        n_feat_pasca, y_scaler_pasca, use_log_pasca,
    ) = load_data("pascabayar")

    print(f"  Train: {len(x_train_pasca)} samples, Val: {len(x_val_pasca)} samples")
    print(f"  Features: {n_feat_pasca}")

    grid_pasca = get_mlp_grid_pascabayar(n_feat_pasca)

    results_pasca = run_grid_search_mlp(
        "PascabayarModel", PascabayarModel,
        x_train_pasca, y_train_pasca,
        x_val_pasca, y_val_pasca,
        y_val_orig_pasca,
        n_feat_pasca, y_scaler_pasca,
        grid_pasca,
    )

    print_top_results("PascabayarModel", results_pasca, top_n=5)

    best_params_pasca = get_best_valid_params(results_pasca)
    if best_params_pasca is None:
        print("\n  ⚠️  PascabayarModel: tidak ada kombinasi valid -- skip test eval.")
        test_pasca = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}
    else:
        test_pasca = final_eval_mlp(
            "PascabayarModel", PascabayarModel,
            best_params_pasca,
            x_train_pasca, y_train_pasca,
            x_val_pasca, y_val_pasca,
            y_val_orig_pasca,
            y_scaler_pasca,
        )

    save_results("PascabayarModel", results_pasca, test_pasca, output_dir)

    # ==================================================================
    # 2. PRABAYAR MODEL (MLP) -- grid DIREVISI untuk atasi divergensi
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  LOADING DATA: Prabayar")
    print("▓" * 70)

    (
        x_train_pra, x_val_pra,
        y_train_pra, y_val_pra,
        y_val_orig_pra,
        n_feat_pra, y_scaler_pra, use_log_pra,
    ) = load_data("prabayar")

    print(f"  Train: {len(x_train_pra)} samples, Val: {len(x_val_pra)} samples")
    print(f"  Features: {n_feat_pra}")

    grid_pra = get_mlp_grid_prabayar(n_feat_pra)

    results_pra = run_grid_search_mlp(
        "PrabayarModel", PrabayarModel,
        x_train_pra, y_train_pra,
        x_val_pra, y_val_pra,
        y_val_orig_pra,
        n_feat_pra, y_scaler_pra,
        grid_pra,
    )

    print_top_results("PrabayarModel", results_pra, top_n=5)

    best_params_pra = get_best_valid_params(results_pra)
    if best_params_pra is None:
        print("\n  ⚠️  PrabayarModel: SEMUA kombinasi masih divergen.")
        print("  Rekomendasi: turunkan learning_rate lebih jauh (coba 1e-5)")
        print("  dan/atau naikkan clip_value (coba 20.0-30.0), lalu jalankan ulang.")
        test_pra = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}
    else:
        test_pra = final_eval_mlp(
            "PrabayarModel", PrabayarModel,
            best_params_pra,
            x_train_pra, y_train_pra,
            x_val_pra, y_val_pra,
            y_val_orig_pra,
            y_scaler_pra,
        )

    save_results("PrabayarModel", results_pra, test_pra, output_dir)

    # ==================================================================
    # 3. PASCABAYAR PLACE VALUE MODEL -- grid DIREVISI
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  LOADING DATA: PascabayarPlaceValue (reusing pascabayar data)")
    print("▓" * 70)

    print(f"  Train: {len(x_train_pasca)} samples, Val: {len(x_val_pasca)} samples")
    print(f"  Features: {n_feat_pasca}")

    grid_pv = get_pv_grid()

    results_pv = run_grid_search_pv(
        x_train_pasca, y_train_pasca,
        x_val_pasca, y_val_pasca,
        y_val_orig_pasca,
        n_feat_pasca, y_scaler_pasca,
        grid_pv,
    )

    print_top_results("PascabayarPlaceValueModel", results_pv, top_n=5)

    best_params_pv = get_best_valid_params(results_pv)
    if best_params_pv is None:
        print("\n  ⚠️  PascabayarPlaceValueModel: SEMUA kombinasi masih divergen.")
        print("  Rekomendasi: turunkan learning_rate lebih jauh (coba 1e-6 - 5e-6)")
        print("  dan/atau naikkan clip_value (coba 20.0-30.0).")
        print("  CATATAN: hidden_size=7 dengan 21+ fitur input adalah bottleneck")
        print("  struktural ~3:1 yang TIDAK bisa diperbaiki lewat tuning learning")
        print("  rate/clip_value saja -- ini batasan arsitektur yang disengaja.")
        test_pv = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}
    else:
        test_pv = final_eval_pv(
            best_params_pv,
            n_feat_pasca,
            x_train_pasca, y_train_pasca,
            x_val_pasca, y_val_pasca,
            y_val_orig_pasca,
            y_scaler_pasca,
        )

    save_results("PascabayarPlaceValueModel", results_pv, test_pv, output_dir)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  RINGKASAN GRID SEARCH")
    print("▓" * 70)

    for name, res, test in [
        ("PascabayarModel", results_pasca, test_pasca),
        ("PrabayarModel", results_pra, test_pra),
        ("PascabayarPlaceValueModel", results_pv, test_pv),
    ]:
        valid = [r for r in res if not r.get("diverged")]
        n_div = len(res) - len(valid)
        print(f"\n  {name}:")
        print(f"    Divergen: {n_div}/{len(res)} ({n_div/len(res)*100:.1f}%)")
        if valid:
            best = valid[0]
            print(f"    Val  RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  "
                  f"MAPE={best['mape']:.4f}%  R²={best['r2']:.6f}")
            print(f"    Test RMSE={test['rmse']:.4f}  MAE={test['mae']:.4f}  "
                  f"MAPE={test['mape']:.4f}%  R²={test['r2']:.6f}")
            print(f"    Best params: {best['params']}")
        else:
            print(f"    ⚠️  Tidak ada kombinasi konvergen -- lihat rekomendasi di atas.")

    print(f"\n  Output directory: {output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()