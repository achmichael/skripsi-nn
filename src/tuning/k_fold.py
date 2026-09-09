"""
k_fold.py — K-Fold Cross-Validation

Splits data into k folds, trains on k-1 folds, validates on remaining fold.
Repeats k times, reports average metrics across all folds.
"""

import copy
import math
import random
import time
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tuning.train_tuning import (
    compute_rmse, compute_mae, compute_mape, compute_r2, evaluate_metrics,
    is_invalid_number, make_divergent_metrics,
    predict_mlp,
)
from src.pipeline.preprocessing import (
    load_and_preprocess,
    fit_standard_scaler,
    transform_standard_scaler,
    fit_target_scaler,
    transform_target,
    inverse_transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.models.prabayar import PrabayarModel
from src.config.config import config


# =====================================================================
# K-FOLD SPLIT
# =====================================================================

def k_fold_split(
    x_data: list,
    y_data: list,
    k: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Split data into k folds. Returns list of k dicts, each containing:
        - x_train, y_train
        - x_val, y_val
    """
    n = len(x_data)
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError(f"k={k} exceeds sample count n={n}")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    fold_size = n // k
    remainder = n % k

    folds_indices = []
    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        folds_indices.append(indices[start:end])
        start = end

    splits = []
    for fold_idx in range(k):
        val_indices = folds_indices[fold_idx]
        train_indices = []
        for j in range(k):
            if j != fold_idx:
                train_indices.extend(folds_indices[j])

        splits.append({
            "x_train": [x_data[i] for i in train_indices],
            "y_train": [y_data[i] for i in train_indices],
            "x_val": [x_data[i] for i in val_indices],
            "y_val": [y_data[i] for i in val_indices],
        })

    return splits


# =====================================================================
# TRAIN SINGLE FOLD
# =====================================================================

def train_fold(
    model,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    patience: int = 15,
    check_every: int = 5,
):
    """Train model on one fold with early stopping. Returns best model + info."""
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


# =====================================================================
# LOAD RAW DATA (no split, no scaling)
# =====================================================================

def load_raw_data(model_type: str = "prabayar"):
    """Load and preprocess CSV, extract features. No train/test split, no scaling."""
    cfg = config[model_type]
    df, _ = load_and_preprocess(cfg["dataset_path"])
    x_data, y_data, feat_cols, target_col = extract_features_and_target(df, model_type)
    n_features = len(feat_cols)
    return x_data, y_data, n_features, feat_cols


# =====================================================================
# K-FOLD CROSS-VALIDATION
# =====================================================================

def run_kfold_cv(
    model_type: str = "prabayar",
    k: int = 5,
    max_epochs: int = 270,
    patience: int = 20,
    check_every: int = 5,
    seed: int = 42,
    layer_sizes: list[int] | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    clip_value: float | None = None,
    l2_lambda: float | None = None,
):
    """
    Run k-fold cross-validation. Per fold:
      1. Split into train/val
      2. Fit scaler on train fold only
      3. Scale train+val
      4. Train model with early stopping
      5. Evaluate on val in original scale

    Returns dict with per-fold and average metrics.
    """
    cfg = config[model_type]

    # Defaults from config
    lr = learning_rate if learning_rate is not None else cfg["learning_rate"]
    bs = batch_size if batch_size is not None else cfg["batch_size"]
    cv = clip_value if clip_value is not None else cfg["clip_value"]
    l2 = l2_lambda if l2_lambda is not None else cfg["l2_lambda"]
    use_log = cfg.get("use_log_transform", False)

    print(f"\n{'▓' * 70}")
    print(f"  K-FOLD CROSS-VALIDATION (k={k}) — {model_type}")
    print(f"{'▓' * 70}")

    # 1. Load raw data
    x_data, y_data, n_features, feat_cols = load_raw_data(model_type)
    print(f"  Total samples: {len(x_data)} | Features: {n_features}")

    ls = layer_sizes if layer_sizes is not None else [n_features] + cfg["hidden_layers"] + [1]
    print(f"  Architecture: {ls}")
    print(f"  LR={lr} | batch={bs} | clip={cv} | L2={l2} | epochs={max_epochs}")

    # 2. Create k-fold splits
    splits = k_fold_split(x_data, y_data, k=k, seed=seed)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, split in enumerate(splits):
        t0 = time.time()
        print(f"\n  ── Fold {fold_idx + 1}/{k} "
              f"(train={len(split['x_train'])}, val={len(split['x_val'])})")

        # 3. Fit scalers on TRAIN fold only
        x_scaler = fit_standard_scaler(split["x_train"])
        x_train_scaled = transform_standard_scaler(split["x_train"], x_scaler)
        x_val_scaled = transform_standard_scaler(split["x_val"], x_scaler)

        y_scaler = fit_target_scaler(split["y_train"], use_log=use_log)
        y_train_scaled = transform_target(split["y_train"], y_scaler)
        y_val_scaled = transform_target(split["y_val"], y_scaler)

        # 4. Init fresh model
        np.random.seed(seed)
        model = PrabayarModel(
            layer_sizes=list(ls),
            seed=seed,
            clip_value=cv,
            l2_lambda=l2,
        )

        # 5. Train
        info = train_fold(
            model,
            x_train_scaled, y_train_scaled,
            x_val_scaled, y_val_scaled,
            lr, max_epochs, bs,
            patience=patience,
            check_every=check_every,
        )

        elapsed = time.time() - t0

        if info["diverged"]:
            print(f"     ⚠ DIVERGED at epoch {info['best_epoch']}")
            fold_metrics.append({
                "fold": fold_idx + 1,
                "diverged": True,
                "rmse": float("inf"),
                "mae": float("inf"),
                "mape": float("inf"),
                "r2": -float("inf"),
                "best_epoch": info["best_epoch"],
                "time": elapsed,
            })
            continue

        # 6. Evaluate in original scale
        preds_scaled = predict_mlp(info["model"], x_val_scaled)
        preds_original = np.array([
            inverse_transform_target(float(p), y_scaler) for p in preds_scaled
        ])
        y_true = np.array(split["y_val"], dtype=np.float64)

        metrics = evaluate_metrics(y_true, preds_original)
        metrics["fold"] = fold_idx + 1
        metrics["diverged"] = False
        metrics["best_epoch"] = info["best_epoch"]
        metrics["time"] = elapsed
        fold_metrics.append(metrics)

        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(preds_original.tolist())

        print(f"     RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
              f"MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.6f}  "
              f"epoch={info['best_epoch']}  ({elapsed:.1f}s)")

    # 7. Aggregate
    valid_folds = [m for m in fold_metrics if not m.get("diverged")]
    n_valid = len(valid_folds)

    print(f"\n{'─' * 70}")
    print(f"  SUMMARY: {n_valid}/{k} folds converged")

    if n_valid == 0:
        avg_metrics = {
            "rmse": float("inf"),
            "mae": float("inf"),
            "mape": float("inf"),
            "r2": -float("inf"),
        }
    else:
        avg_metrics = {
            "rmse": float(np.mean([m["rmse"] for m in valid_folds])),
            "mae": float(np.mean([m["mae"] for m in valid_folds])),
            "mape": float(np.mean([m["mape"] for m in valid_folds])),
            "r2": float(np.mean([m["r2"] for m in valid_folds])),
        }
        std_metrics = {
            "rmse_std": float(np.std([m["rmse"] for m in valid_folds])),
            "mae_std": float(np.std([m["mae"] for m in valid_folds])),
            "mape_std": float(np.std([m["mape"] for m in valid_folds])),
            "r2_std": float(np.std([m["r2"] for m in valid_folds])),
        }

        # Overall metrics across all predictions
        overall_metrics = evaluate_metrics(
            np.array(all_y_true), np.array(all_y_pred)
        )

        print(f"\n  Average across folds:")
        print(f"    RMSE  = {avg_metrics['rmse']:.4f} ± {std_metrics['rmse_std']:.4f}")
        print(f"    MAE   = {avg_metrics['mae']:.4f} ± {std_metrics['mae_std']:.4f}")
        print(f"    MAPE  = {avg_metrics['mape']:.2f}% ± {std_metrics['mape_std']:.2f}%")
        print(f"    R²    = {avg_metrics['r2']:.6f} ± {std_metrics['r2_std']:.6f}")

        print(f"\n  Overall (all predictions pooled):")
        print(f"    RMSE  = {overall_metrics['rmse']:.4f}")
        print(f"    MAE   = {overall_metrics['mae']:.4f}")
        print(f"    MAPE  = {overall_metrics['mape']:.2f}%")
        print(f"    R²    = {overall_metrics['r2']:.6f}")

        avg_metrics["std"] = std_metrics
        avg_metrics["overall"] = overall_metrics

    return {
        "k": k,
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "n_converged": n_valid,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    results = run_kfold_cv(
        model_type="prabayar",
        k=5,
        max_epochs=270,
        patience=20,
        check_every=5,
        seed=42,
    )

    print(f"\n{'▓' * 70}")
    print(f"  K-FOLD CV COMPLETE")
    print(f"{'▓' * 70}")


if __name__ == "__main__":
    main()
