"""
Training Module for Capacity-Specific Models

This module handles training of separate models for each electrical capacity category.
"""

import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.preprocessing import (
    load_and_preprocess,
    train_test_split,
    fit_standard_scaler,
    transform_standard_scaler,
    fit_target_scaler,
    transform_target,
    inverse_transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.utils.core import train_model, evaluate_model
from src.models.model_factory import create_model_for_capacity
from src.config.config import config


def save_loss_curve(history: dict, save_path: str, capacity: str):
    """Plot training loss curve untuk capacity-specific model."""
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        print(f"      Warning: No training loss data to plot")
        return

    trained_epochs = len(train_loss)
    epoch_axis = list(range(1, trained_epochs + 1))

    marker_size = max(1.5, 5.0 - trained_epochs * 0.008)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Train loss
    ax.plot(
        epoch_axis, train_loss,
        color="#1565C0", linewidth=1.5,
        marker="o", markersize=marker_size, markerfacecolor="#1565C0",
        label="Train Loss",
        zorder=3,
    )

    # Validation loss
    if val_loss:
        ax.plot(
            epoch_axis, val_loss,
            color="#E53935", linewidth=1.5,
            marker="o", markersize=marker_size, markerfacecolor="#E53935",
            label="Validation Loss",
            zorder=3,
        )

    # Best train loss point
    best_epoch = train_loss.index(min(train_loss)) + 1
    best_val = min(train_loss)
    ax.scatter(
        [best_epoch], [best_val],
        color="#FFB300", s=60, zorder=5,
        label=f"Best Train Loss (epoch {best_epoch}: {best_val:.6f})",
    )

    # Early stopping line
    ax.axvline(
        x=trained_epochs, color="gray", linestyle=":", linewidth=1.2,
        label=f"Early stop (epoch {trained_epochs})",
    )

    ax.set_title(
        f"Training vs Validation Loss — {capacity}VA ({trained_epochs} epochs)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_xlim(1, trained_epochs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"      Loss curve saved: {save_path}")


def save_prediction_scatter(y_actual, y_predicted, save_path: str, capacity: str):
    """Plot scatter prediksi vs aktual untuk capacity-specific model."""
    actual_vals = list(y_actual)
    pred_vals = list(y_predicted)

    # Calculate R²
    mean_actual = sum(actual_vals) / len(actual_vals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_vals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual_vals, pred_vals))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Calculate percentage within ±20% tolerance
    within_20pct = sum(
        1 for a, p in zip(actual_vals, pred_vals)
        if a != 0 and abs(p - a) / abs(a) <= 0.20
    )
    pct_within = within_20pct / len(actual_vals) * 100

    all_vals = actual_vals + pred_vals
    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) * 0.05
    lo = min_val - margin
    hi = max_val + margin

    fig, ax = plt.subplots(figsize=(9, 8))

    # Tolerance zone ±20%
    ref = [lo, hi]
    ax.fill_between(
        ref,
        [v * 0.80 for v in ref],
        [v * 1.20 for v in ref],
        alpha=0.12,
        color="#2196F3",
        label="Tolerance ±20%",
    )

    # Scatter points - color based on tolerance
    colors = [
        "#4CAF50" if abs(a) > 0 and abs(p - a) / abs(a) <= 0.20 else "#F44336"
        for a, p in zip(actual_vals, pred_vals)
    ]
    ax.scatter(actual_vals, pred_vals, c=colors, alpha=0.75, s=35, zorder=3)

    # Ideal line y = x
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)", zorder=4)

    # Dummy scatter for legend
    ax.scatter([], [], c="#4CAF50", s=35, label=f"Within tolerance ({within_20pct}/{len(actual_vals)} points)")
    ax.scatter([], [], c="#F44336", s=35, label="Outside tolerance")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual (hari)", fontsize=12)
    ax.set_ylabel("Predicted (hari)", fontsize=12)
    ax.set_title(f"Prediction vs Actual — {capacity}VA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Statistics box
    stats_text = f"R² = {r2:.4f}\nWithin ±20%: {pct_within:.1f}%"
    ax.text(
        0.98, 0.05, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Scatter plot saved: {save_path}")


def train_single_capacity(capacity: str, verbose: bool = True):
    """
    Train model for a specific capacity category.
    
    Args:
        capacity: Capacity category ("450", "900", "1300", "2200", "3500")
        verbose: Print detailed logs
    
    Returns:
        dict with training results and metrics
    """
    if capacity not in config["capacity_configs"]:
        raise ValueError(f"Capacity '{capacity}' not found in config. Available: {list(config['capacity_configs'].keys())}")
    
    cfg = config["capacity_configs"][capacity]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  TRAINING MODEL FOR {capacity}VA CAPACITY")
        print(f"{'='*70}")
    
    # Check if dataset exists
    if not os.path.exists(cfg["dataset_path"]):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {cfg['dataset_path']}")
    
    # Create output directories
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    os.makedirs(cfg["metrics_dir"], exist_ok=True)
    
    # Load and preprocess data
    if verbose:
        print(f"\n[1/7] Loading data: {cfg['dataset_path']}")
    
    rows, minmax_scaler_params, prob_params = load_and_preprocess(cfg["dataset_path"])
    
    if verbose:
        print(f"      Total data: {len(rows)} baris")
    
    # Extract features and target
    x_data, y_data, feature_columns, target_column = extract_features_and_target(
        df=rows,
        model_type="prabayar",
    )
    input_size = len(feature_columns)
    
    if verbose:
        print(f"      Fitur: {input_size} kolom")
        print(f"      Target: {target_column}")
    
    # Split data
    if verbose:
        print(f"\n[2/7] Splitting data (80/20)")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_data=x_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )
    
    if verbose:
        print(f"      Train: {len(x_train)} samples")
        print(f"      Test:  {len(x_test)} samples")
    
    # Scale features
    if verbose:
        print(f"\n[3/7] Scaling features (Standard Scaler)")
    
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_test_scaled = transform_standard_scaler(x_test, x_scaler)
    
    # Scale target
    y_scaler = fit_target_scaler(y_train, use_log=cfg.get("use_log_transform", False))
    y_train_scaled = transform_target(y_train, y_scaler)
    y_test_scaled = transform_target(y_test, y_scaler)
    
    # Build layer sizes
    layer_sizes = [input_size] + cfg["hidden_layers"] + [1]
    
    if verbose:
        print(f"\n[4/7] Building model")
        print(f"      Architecture: {layer_sizes}")
        print(f"      Learning rate: {cfg['learning_rate']}")
        print(f"      Batch size: {cfg['batch_size']}")
        print(f"      L2 lambda: {cfg['l2_lambda']}")
    
    # Create capacity-specific model
    model = create_model_for_capacity(
        capacity=capacity,
        layer_sizes=layer_sizes,
        seed=42,
        clip_value=cfg["clip_value"],
        l2_lambda=cfg.get("l2_lambda", 0.0),
        asymmetric_alpha=cfg.get("asymmetric_alpha", 0.5),
        l1_lambda_input=cfg.get("l1_lambda_input", 0.0),
    )
    
    if verbose:
        print(f"      {model.get_summary()}")
    
    # Train model
    if verbose:
        print(f"\n[5/7] Training model...")
    
    history = train_model(
        model=model,
        x_train=x_train_scaled,
        y_train=y_train_scaled,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg.get("batch_size", 16),
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        epochs=None,
        x_val=x_test_scaled,
        y_val=y_test_scaled,
        lr_decay=cfg.get("lr_decay", 0.0),
        use_log=cfg.get("use_log_transform", False),
        model_type=f"prabayar_{capacity}",
    )
    
    total_epochs = len(history["train_loss"])
    
    if verbose:
        print(f"      Training selesai. Total epoch: {total_epochs}")
    
    # Evaluate
    if verbose:
        print(f"\n[6/7] Evaluating model...")
    
    evaluation = evaluate_model(
        model=model,
        x_test=x_test_scaled,
        y_test=y_test_scaled,
    )
    
    # Calculate metrics in original scale
    preds_orig = [
        inverse_transform_target(p, y_scaler)
        for p in evaluation["predictions"]
    ]
    errors_orig = [p - a for p, a in zip(preds_orig, y_test)]
    
    mape = sum(
        abs(p - a) / max(abs(a), 1)
        for p, a in zip(preds_orig, y_test)
    ) / len(y_test) * 100
    
    mse_orig = sum(e ** 2 for e in errors_orig) / len(errors_orig)
    rmse_orig = math.sqrt(mse_orig)
    mae_orig = sum(abs(e) for e in errors_orig) / len(errors_orig)
    
    # Calculate R²
    mean_actual = sum(y_test) / len(y_test)
    ss_tot = sum((a - mean_actual) ** 2 for a in y_test)
    ss_res = sum((a - p) ** 2 for a, p in zip(y_test, preds_orig))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    if verbose:
        print(f"\n      Metrics (original scale):")
        print(f"      RMSE: {rmse_orig:,.4f}")
        print(f"      MAE:  {mae_orig:,.4f}")
        print(f"      MAPE: {mape:.2f}%")
        print(f"      R²:   {r2:.4f}")
    
    # Generate plots
    if verbose:
        print(f"\n[7/7] Generating plots...")
    
    # Training loss curve
    loss_curve_path = os.path.join(cfg["metrics_dir"], "training_loss_curve.png")
    save_loss_curve(history, loss_curve_path, capacity)
    
    # Prediction vs Actual scatter
    scatter_path = os.path.join(cfg["metrics_dir"], "prediction_vs_actual.png")
    save_prediction_scatter(y_test, preds_orig, scatter_path, capacity)
    
    # Save metrics
    metrics_data = {
        "capacity": f"{capacity}VA",
        "model_type": f"prabayar_{capacity}",
        "arsitektur": layer_sizes,
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "total_epochs": total_epochs,
        "final_train_loss": history["train_loss"][-1],
        "best_train_loss": min(history["train_loss"]),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "evaluasi_skala_asli": {
            "rmse": rmse_orig,
            "mae": mae_orig,
            "mape": mape,
            "r2": r2,
        },
        "data_info": {
            "total_samples": len(rows),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "n_features": input_size,
        },
        "plots": {
            "loss_curve": loss_curve_path,
            "prediction_scatter": scatter_path,
        }
    }
    
    metrics_path = os.path.join(cfg["metrics_dir"], "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
    
    if verbose:
        print(f"\n      Metrics saved to: {metrics_path}")
    
    # Save model
    metadata = {
        "model_type": f"prabayar_{capacity}",
        "capacity": f"{capacity}VA",
        "feature_columns": feature_columns,
        "target_column": target_column,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "minmax_scaler_params": minmax_scaler_params,
        "prob_params": prob_params,
        "layer_sizes": layer_sizes,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }
    
    model.save(cfg["model_path"], metadata=metadata)
    
    if verbose:
        print(f"      Model saved to: {cfg['model_path']}")
        print(f"\n{'='*70}\n")
    
    return {
        "capacity": capacity,
        "metrics": metrics_data,
        "model_path": cfg["model_path"],
        "metrics_path": metrics_path,
        "history": history,
        "y_test": y_test,
        "y_pred": preds_orig,
    }


def train_all_capacities(capacities: list[str] = None, verbose: bool = True):
    """
    Train models for all capacity categories.
    
    Args:
        capacities: List of capacities to train. If None, train all.
        verbose: Print detailed logs
    
    Returns:
        dict with results for each capacity
    """
    if capacities is None:
        capacities = list(config["capacity_configs"].keys())
    
    results = {}
    summary = []
    
    print(f"\n{'▓'*70}")
    print(f"  TRAINING CAPACITY-SPECIFIC MODELS")
    print(f"  Capacities: {', '.join([f'{c}VA' for c in capacities])}")
    print(f"{'▓'*70}\n")
    
    for capacity in capacities:
        try:
            result = train_single_capacity(capacity, verbose=verbose)
            results[capacity] = result
            
            metrics = result["metrics"]["evaluasi_skala_asli"]
            summary.append({
                "capacity": f"{capacity}VA",
                "samples": result["metrics"]["data_info"]["total_samples"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
            })
            
        except Exception as e:
            print(f"\n[ERROR] Failed to train {capacity}VA model: {e}")
            import traceback
            traceback.print_exc()
            results[capacity] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'▓'*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'▓'*70}\n")
    
    print(f"{'Capacity':<12} {'Samples':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'R²':<10}")
    print(f"{'-'*70}")
    
    for item in summary:
        print(f"{item['capacity']:<12} "
              f"{item['samples']:<10} "
              f"{item['rmse']:<12.4f} "
              f"{item['mae']:<12.4f} "
              f"{item['mape']:<10.2f} "
              f"{item['r2']:<10.4f}")
    
    # Save summary
    summary_path = "results/capacity_models_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "trained_at": "2026-09-19",
            "details": {k: v.get("metrics", {}) for k, v in results.items() if "error" not in v}
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\n{'▓'*70}\n")
    
    return results


if __name__ == "__main__":
    # Train all capacity-specific models
    results = train_all_capacities()
