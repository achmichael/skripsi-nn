"""
Grid Search Hyperparameter Tuning — Pure Python + NumPy.

Tiga model:
  1. PascabayarModel         (MLP standar)
  2. PrabayarModel           (MLP standar)
  3. PascabayarPlaceValueModel (place-value architecture)

Methodology:
  - 80/20 train/test split (seed=42)
  - Setiap kombinasi: init model (seed=42), train, eval pada validation set
  - Metrik: RMSE, MAE, MAPE, R²
  - Ranking by RMSE (primary), MAE (tiebreaker)
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
# TRAINING HELPERS (silent, no prints)
# =====================================================================

def train_mlp_silent(
    model,
    x_train: list, y_train: list,
    learning_rate: float, epochs: int, batch_size: int,
):
    """Train MLP model (PascabayarModel or PrabayarModel) silently."""
    X = np.array(x_train, dtype=np.float32)
    Y = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    n = X.shape[0]

    for ep in range(epochs):
        indices = np.random.permutation(n)
        X_shuf = X[indices]
        Y_shuf = Y[indices]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            model.train_batch(X_shuf[start:end], Y_shuf[start:end], learning_rate)


def train_pv_silent(
    model: PascabayarPlaceValueModel,
    x_train: list, y_train: list,
    learning_rate: float, epochs: int, batch_size: int,
):
    """Train PascabayarPlaceValueModel silently (sample-by-sample in batches)."""
    n = len(x_train)
    indices_all = list(range(n))

    for ep in range(epochs):
        np.random.shuffle(indices_all)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices_all[start:end]

            for i in batch_idx:
                x_i = x_train[i] if isinstance(x_train[i], list) else list(x_train[i])
                y_i = float(y_train[i])
                model.train_one_sample(x_i, y_i, learning_rate)


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

def get_mlp_grid(n_features: int) -> list[dict]:
    """Generate hyperparameter grid for MLP models."""
    param_grid = {
        "layer_sizes": [
            # Shallow Architectures
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 128, 1],
            # Medium Deep Architectures
            [n_features, 64, 32, 1],
            [n_features, 128, 64, 1],
            # Deep Architectures
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


def get_pv_grid() -> list[dict]:
    """Generate hyperparameter grid for PascabayarPlaceValueModel."""
    param_grid = {
        "component_activation": ["relu", "linear"],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [1.0, 3.0, 5.0, 10.0],
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
    """Train one MLP combo, return metrics dict."""
    np.random.seed(42)

    model = ModelClass(
        layer_sizes=list(params["layer_sizes"]),
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
    )

    train_mlp_silent(
        model, x_train, y_train_scaled,
        params["learning_rate"], params["epochs"], params["batch_size"],
    )

    # Predict on validation (normalized scale)
    preds_scaled = predict_mlp(model, x_val)

    # Inverse transform to original scale
    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    return metrics


def eval_pv_combo(
    params: dict,
    n_features: int,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    y_scaler,
) -> dict:
    """Train one PlaceValue combo, return metrics dict."""
    np.random.seed(42)
    import random
    random.seed(42)

    model = PascabayarPlaceValueModel(
        input_size=n_features,
        hidden_size=7,
        num_components=7,
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
        component_activation=params["component_activation"],
    )

    train_pv_silent(
        model, x_train, y_train_scaled,
        params["learning_rate"], params["epochs"], params["batch_size"],
    )

    preds_scaled = predict_pv(model, x_val)

    preds_original = np.array([
        inverse_transform_target(float(p), y_scaler) for p in preds_scaled
    ])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
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
) -> list[dict]:
    """Run full grid search for one MLP model. Returns sorted results."""
    grid = get_mlp_grid(n_features)
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH: {model_name}")
    print(f"  Total kombinasi: {total}")
    print(f"{'='*70}\n")

    results = []
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

        results.append(metrics)

        if idx % 50 == 0 or idx == total or idx == 1:
            pct = idx / total * 100
            eta = (time.time() - t_start) / idx * (total - idx)
            print(
                f"  [{idx:>4d}/{total}] {pct:5.1f}%  "
                f"RMSE={metrics['rmse']:>12.2f}  "
                f"MAE={metrics['mae']:>12.2f}  "
                f"R²={metrics['r2']:>7.4f}  "
                f"({elapsed:.1f}s)  "
                f"ETA: {eta/60:.1f}min"
            )

    total_time = time.time() - t_start
    print(f"\n  Selesai dalam {total_time:.1f}s ({total_time/60:.1f}min)")

    # Sort by RMSE (asc), then MAE (asc) as tiebreaker
    results.sort(key=lambda r: (r["rmse"], r["mae"]))
    return results


def run_grid_search_pv(
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    y_val_original,
    n_features: int,
    y_scaler,
) -> list[dict]:
    """Run full grid search for PascabayarPlaceValueModel. Returns sorted results."""
    grid = get_pv_grid()
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH: PascabayarPlaceValueModel")
    print(f"  Total kombinasi: {total}")
    print(f"{'='*70}\n")

    results = []
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

        results.append(metrics)

        if idx % 50 == 0 or idx == total or idx == 1:
            pct = idx / total * 100
            eta = (time.time() - t_start) / idx * (total - idx)
            print(
                f"  [{idx:>4d}/{total}] {pct:5.1f}%  "
                f"RMSE={metrics['rmse']:>12.2f}  "
                f"MAE={metrics['mae']:>12.2f}  "
                f"R²={metrics['r2']:>7.4f}  "
                f"({elapsed:.1f}s)  "
                f"ETA: {eta/60:.1f}min"
            )

    total_time = time.time() - t_start
    print(f"\n  Selesai dalam {total_time:.1f}s ({total_time/60:.1f}min)")

    results.sort(key=lambda r: (r["rmse"], r["mae"]))
    return results


# =====================================================================
# DISPLAY & FINAL EVALUATION HELPERS
# =====================================================================

def print_top_results(model_name: str, results: list[dict], top_n: int = 5):
    """Print top N results for a model."""
    print(f"\n{'='*70}")
    print(f"  TOP {top_n} HASIL — {model_name}")
    print(f"{'='*70}")

    for rank, r in enumerate(results[:top_n], 1):
        print(f"\n  ── Rank #{rank} ──")
        print(f"     RMSE : {r['rmse']:>14.4f}")
        print(f"     MAE  : {r['mae']:>14.4f}")
        print(f"     MAPE : {r['mape']:>14.4f}%")
        print(f"     R²   : {r['r2']:>14.6f}")
        print(f"     Params:")
        for k, v in r["params"].items():
            print(f"       {k:25s}: {v}")


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

    train_mlp_silent(
        model, x_train, y_train_scaled,
        best_params["learning_rate"],
        best_params["epochs"],
        best_params["batch_size"],
    )

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
        num_components=7,
        seed=42,
        clip_value=best_params["clip_value"],
        l2_lambda=best_params["l2_lambda"],
        component_activation=best_params["component_activation"],
    )

    train_pv_silent(
        model, x_train, y_train_scaled,
        best_params["learning_rate"],
        best_params["epochs"],
        best_params["batch_size"],
    )

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
    """Save top results + test metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"grid_search_{safe_name}.json")

    # Convert layer_sizes (list) and other non-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    top5 = []
    for r in results[:5]:
        entry = {
            "rmse": make_serializable(r["rmse"]),
            "mae": make_serializable(r["mae"]),
            "mape": make_serializable(r["mape"]),
            "r2": make_serializable(r["r2"]),
            "params": {
                k: make_serializable(v) for k, v in r["params"].items()
            },
        }
        top5.append(entry)

    output = {
        "model": model_name,
        "top_5_validation": top5,
        "test_set_metrics": {
            k: make_serializable(v) for k, v in test_metrics.items()
        },
        "best_params": {
            k: make_serializable(v) for k, v in results[0]["params"].items()
        },
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
    # 1. PASCABAYAR MODEL (MLP)
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

    results_pasca = run_grid_search_mlp(
        "PascabayarModel", PascabayarModel,
        x_train_pasca, y_train_pasca,
        x_val_pasca, y_val_pasca,
        y_val_orig_pasca,
        n_feat_pasca, y_scaler_pasca,
    )

    print_top_results("PascabayarModel", results_pasca, top_n=5)

    test_pasca = final_eval_mlp(
        "PascabayarModel", PascabayarModel,
        results_pasca[0]["params"],
        x_train_pasca, y_train_pasca,
        x_val_pasca, y_val_pasca,
        y_val_orig_pasca,
        y_scaler_pasca,
    )

    save_results("PascabayarModel", results_pasca, test_pasca, output_dir)

    # ==================================================================
    # 2. PRABAYAR MODEL (MLP)
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

    results_pra = run_grid_search_mlp(
        "PrabayarModel", PrabayarModel,
        x_train_pra, y_train_pra,
        x_val_pra, y_val_pra,
        y_val_orig_pra,
        n_feat_pra, y_scaler_pra,
    )

    print_top_results("PrabayarModel", results_pra, top_n=5)

    test_pra = final_eval_mlp(
        "PrabayarModel", PrabayarModel,
        results_pra[0]["params"],
        x_train_pra, y_train_pra,
        x_val_pra, y_val_pra,
        y_val_orig_pra,
        y_scaler_pra,
    )

    save_results("PrabayarModel", results_pra, test_pra, output_dir)

    # ==================================================================
    # 3. PASCABAYAR PLACE VALUE MODEL
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  LOADING DATA: PascabayarPlaceValue (reusing pascabayar data)")
    print("▓" * 70)

    # Reuse pascabayar data (same dataset, different architecture)
    print(f"  Train: {len(x_train_pasca)} samples, Val: {len(x_val_pasca)} samples")
    print(f"  Features: {n_feat_pasca}")

    results_pv = run_grid_search_pv(
        x_train_pasca, y_train_pasca,
        x_val_pasca, y_val_pasca,
        y_val_orig_pasca,
        n_feat_pasca, y_scaler_pasca,
    )

    print_top_results("PascabayarPlaceValueModel", results_pv, top_n=5)

    test_pv = final_eval_pv(
        results_pv[0]["params"],
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
        best = res[0]
        print(f"\n  {name}:")
        print(f"    Val  RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  "
              f"MAPE={best['mape']:.4f}%  R²={best['r2']:.6f}")
        print(f"    Test RMSE={test['rmse']:.4f}  MAE={test['mae']:.4f}  "
              f"MAPE={test['mape']:.4f}%  R²={test['r2']:.6f}")
        print(f"    Best params: {best['params']}")

    print(f"\n  Output directory: {output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
