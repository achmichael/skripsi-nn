import json
import math
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.pipeline.preprocessing import (
    load_and_preprocess,
    train_test_split,
    fit_minmax_scaler,
    transform_minmax,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.utils.core import train_model, evaluate_model
from src.models.pascabayar_place_value import PascabayarPlaceValueModel
from src.config.config import config


def save_loss_curve(history: dict, save_path: str, model_type: str):
    """Plot training loss curve dengan titik per epoch, anotasi best loss, dan val loss."""
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])

    if not train_loss:
        raise ValueError("Train loss kosong. Tidak bisa membuat kurva loss.")

    trained_epochs = len(train_loss)
    epoch_axis = list(range(1, trained_epochs + 1))
    marker_size = max(1.5, 5.0 - trained_epochs * 0.008)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        epoch_axis, train_loss,
        color="#1565C0", linewidth=1.5,
        marker="o", markersize=marker_size, markerfacecolor="#1565C0",
        label="Train Loss",
        zorder=3,
    )

    if val_loss:
        ax.plot(
            epoch_axis, val_loss,
            color="#E53935", linewidth=1.5,
            marker="o", markersize=marker_size, markerfacecolor="#E53935",
            label="Validation Loss",
            zorder=3,
        )

    best_epoch = train_loss.index(min(train_loss)) + 1
    best_val   = min(train_loss)
    ax.scatter(
        [best_epoch], [best_val],
        color="#FFB300", s=60, zorder=5,
        label=f"Best Train Loss (epoch {best_epoch}: {best_val:.6f})",
    )

    ax.axvline(
        x=trained_epochs, color="gray", linestyle=":", linewidth=1.2,
        label=f"Early stop (epoch {trained_epochs})",
    )

    ax.set_title(
        f"Training vs Validation Loss — {model_type.upper()} ({trained_epochs} epoch)",
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


def save_prediction_scatter(y_actual, y_predicted, save_path: str, model_type: str, label: str):
    """Plot scatter prediksi vs aktual dengan satuan ribu Rp, zona toleransi, dan statistik."""
    scale = 1_000
    actual_k  = [v / scale for v in y_actual]
    pred_k    = [v / scale for v in y_predicted]

    mean_actual = sum(actual_k) / len(actual_k)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_k)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual_k, pred_k))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    within_20pct = sum(
        1 for a, p in zip(actual_k, pred_k)
        if a != 0 and abs(p - a) / abs(a) <= 0.20
    )
    pct_within = within_20pct / len(actual_k) * 100 if len(actual_k) > 0 else 0

    all_vals = actual_k + pred_k
    min_val  = min(all_vals)
    max_val  = max(all_vals)
    margin   = (max_val - min_val) * 0.05
    lo = min_val - margin
    hi = max_val + margin

    fig, ax = plt.subplots(figsize=(9, 8))

    ref = [lo, hi]
    ax.fill_between(
        ref,
        [v * 0.80 for v in ref],
        [v * 1.20 for v in ref],
        alpha=0.12,
        color="#2196F3",
        label="Toleransi ±20%",
    )

    colors = [
        "#4CAF50" if abs(a) > 0 and abs(p - a) / abs(a) <= 0.20 else "#F44336"
        for a, p in zip(actual_k, pred_k)
    ]
    ax.scatter(actual_k, pred_k, c=colors, alpha=0.75, s=35, zorder=3)

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)", zorder=4)

    ax.scatter([], [], c="#4CAF50", s=35, label=f"Dalam toleransi ({within_20pct}/{len(actual_k)} titik)")
    ax.scatter([], [], c="#F44336", s=35, label="Di luar toleransi")

    def fmt_ribu(x, _):
        return f"{x:,.0f} rb"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_ribu))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_ribu))
    ax.tick_params(axis='x', rotation=30)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Aktual ({label}) — ribu Rp", fontsize=12)
    ax.set_ylabel(f"Prediksi ({label}) — ribu Rp", fontsize=12)
    ax.set_title(f"Prediksi vs Aktual — {model_type.upper()}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    stats_text = f"R² = {r2:.4f}\nDalam ±20%: {pct_within:.1f}%"
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


def plot_error_distribution(
    y_true: list[float],
    y_pred: list[float],
    mae_rupiah: float,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    errors = y_p - y_t

    n = len(errors)
    if n == 0:
        return

    max_abs_err = np.max(np.abs(errors))
    if max_abs_err < 500_000:
        unit_divider = 1_000.0
        unit_name = "ribu Rp"
    else:
        unit_divider = 1_000_000.0
        unit_name = "juta Rp"

    errors_scaled = errors / unit_divider
    mae_scaled = mae_rupiah / unit_divider
    
    mean_e = np.mean(errors_scaled)
    median_e = np.median(errors_scaled)
    std_e = np.std(errors_scaled)
    min_e = np.min(errors_scaled)
    max_e = np.max(errors_scaled)
    range_e = max_e - min_e

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    ax1.hist(errors_scaled, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    ticks = np.linspace(min_e, max_e, 10)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{v:+.1f}" for v in ticks])
    
    ax1.set_xlabel(f"Error ({unit_name}) [Prediksi - Aktual]", fontsize=12)
    ax1.set_ylabel("Frekuensi", fontsize=12)
    ax1.set_title("Distribusi Error Prediksi", fontsize=14, fontweight='bold')

    ax1.axvspan(-mae_scaled, mae_scaled, color='lightgreen', alpha=0.3, label='±MAE zone')

    offset = max(range_e * 0.005, 1e-5)
    mean_offset = offset if abs(mean_e - 0.0) <= offset * 2 else 0.0
    median_offset = -offset if (abs(median_e - 0.0) <= offset * 2 or abs(median_e - mean_e) <= offset * 2) else 0.0

    ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='Zero Error')
    ax1.axvline(mean_e + mean_offset, color='orange', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(median_e + median_offset, color='yellow', linestyle=':', linewidth=2, label='Median')
    
    ax1.legend(loc='upper left', fontsize=10)

    stats_text = (
        f"n = {n} data\n"
        f"Mean = {mean_e:+.1f} {unit_name}\n"
        f"Median = {median_e:+.1f} {unit_name}\n"
        f"Std Dev = {std_e:.1f} {unit_name}\n"
        f"Range: [{min_e:+.1f}, {max_e:+.1f}] {unit_name}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    buckets = [
        (0, 150_000, "0–150rb"),
        (150_000, 300_000, "150–300rb"),
        (300_000, 500_000, "300–500rb"),
        (500_000, 750_000, "500–750rb"),
        (750_000, float('inf'), "750rb+")
    ]
    
    bucket_labels = []
    bucket_maes = []
    bucket_counts = []
    bucket_colors = []
    
    for low, high, label in buckets:
        mask = (y_t >= low) & (y_t < high)
        count = np.sum(mask)
        
        bucket_labels.append(label)
        bucket_counts.append(count)
        
        if count > 0:
            bucket_errs = errors[mask]
            b_mae_raw = np.mean(np.abs(bucket_errs))
            b_mae_ribu = b_mae_raw / 1000.0
            
            bucket_maes.append(b_mae_ribu)
            
            if b_mae_raw < 100_000:
                bucket_colors.append('green')
            elif b_mae_raw < 200_000:
                bucket_colors.append('yellow')
            else:
                bucket_colors.append('red')
        else:
            bucket_maes.append(0.0)
            bucket_colors.append('gray')
            
    x_pos = np.arange(len(buckets))
    bars = ax2.bar(x_pos, bucket_maes, color=bucket_colors, edgecolor='black', alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bucket_labels, fontsize=10)
    ax2.set_xlabel("Tagihan Aktual", fontsize=12)
    ax2.set_ylabel("MAE (ribu Rp)", fontsize=12)
    ax2.set_title("Mean Absolute Error per Bucket Tagihan Aktual", fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, bucket_counts):
        height = bar.get_height()
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f"n={count}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_training():
    model_type = "pascabayar_place_value"
    cfg = config[model_type]
    
    # Override directories for place value model
    model_path = "results/pascabayar_place_value/models/model_place_value.json"
    metrics_dir = "results/pascabayar_place_value/metrics"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"=== Training PASCABAYAR PLACE VALUE MODEL ===\n")

    # Load & encode
    rows = load_and_preprocess(cfg["dataset_path"])
    print(f"Total data: {len(rows)} baris")

    # Extract features & target
    x_data, y_data, feature_columns, target_column = extract_features_and_target(
        df=rows,
        model_type=model_type,
    )
    input_size = len(feature_columns)
    print(f"Fitur: {input_size} kolom")
    print(f"Target: {target_column}\n")

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data=x_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )

    # Scale features
    x_scaler = fit_minmax_scaler(x_train)
    x_train_scaled = transform_minmax(x_train, x_scaler)
    x_test_scaled = transform_minmax(x_test, x_scaler)

    # Target scaling for Place Value: simply divide by 1,000,000
    y_train_scaled = [y / 1_000_000.0 for y in y_train]
    y_test_scaled = [y / 1_000_000.0 for y in y_test]

    model = PascabayarPlaceValueModel(
        input_size=input_size,
        hidden_size=7,
        num_components=7,
        seed=42,
        clip_value=cfg["clip_value"],
        l2_lambda=cfg.get("l2_lambda", 0.0),
        component_activation="relu"
    )

    print("Mulai training...")
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
    )
    
    total_epochs = len(history["train_loss"])
    print(f"Training selesai. Total epoch aktual: {total_epochs}")

    # Evaluate
    evaluation = evaluate_model(
        model=model,
        x_test=x_test_scaled,
        y_test=y_test_scaled,
    )

    print(f"\nEvaluasi (skala normalisasi / 1.000.000):")
    print(f"  MSE: {evaluation['mse']:.8f}")
    print(f"  MAE: {evaluation['mae']:.8f}")

    # MAPE & RMSE in original scale
    preds_orig = [p * 1_000_000.0 for p in evaluation["predictions"]]
    errors_orig = [p - a for p, a in zip(preds_orig, y_test)]

    mape = sum(
        abs(p - a) / max(abs(a), 1)
        for p, a in zip(preds_orig, y_test)
    ) / len(y_test) * 100

    mse_orig = sum(e ** 2 for e in errors_orig) / len(errors_orig)
    rmse_orig = math.sqrt(mse_orig)
    mae_orig = sum(abs(e) for e in errors_orig) / len(errors_orig)

    print(f"  MAPE: {mape:.2f}%")
    print(f"\nEvaluasi (skala asli Rp):")
    print(f"  MSE: {mse_orig:,.4f}")
    print(f"  RMSE: {rmse_orig:,.4f}")
    print(f"  MAE: {mae_orig:,.4f}")

    print("\n--- Perbandingan Prediksi vs Aktual (15 Data Pertama) ---")
    for i in range(min(15, len(y_test))):
        print(f"Data {i+1:2d} | Aktual: Rp {y_test[i]:>10,.0f} | Prediksi: Rp {preds_orig[i]:>10,.0f} | Selisih: Rp {errors_orig[i]:>10,.0f}")

    # Visualisasi dan Metrik
    save_loss_curve(
        history=history,
        save_path=os.path.join(metrics_dir, "training_loss_curve.png"),
        model_type="pascabayar_place_value",
    )

    save_prediction_scatter(
        y_actual=y_test,
        y_predicted=preds_orig,
        save_path=os.path.join(metrics_dir, "prediction_vs_actual.png"),
        model_type="pascabayar_place_value",
        label=cfg["target_label"],
    )

    plot_error_distribution(
        y_true=y_test,
        y_pred=preds_orig,
        mae_rupiah=mae_orig,
        save_path=os.path.join(metrics_dir, "error_distribution.png"),
        show=False,
    )

    metrics_data = {
        "model_type": "pascabayar_place_value",
        "arsitektur": [input_size, 7, 7],
        "learning_rate": cfg["learning_rate"],
        "total_epochs": total_epochs,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "best_train_loss": min(history["train_loss"]) if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "evaluasi_normalisasi": {
            "mse": evaluation["mse"],
            "mae": evaluation["mae"],
        },
        "evaluasi_skala_asli": {
            "mse": mse_orig,
            "rmse": rmse_orig,
            "mae": mae_orig,
            "mape": mape,
        },
        "data_split": {
            "train": len(x_train),
            "test": len(x_test),
        },
    }
    
    metrics_file = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
    print(f"Metrics JSON disimpan ke: {metrics_file}")

    metadata = {
        "model_type": "pascabayar_place_value",
        "feature_columns": feature_columns,
        "target_column": target_column,
        "x_scaler": x_scaler,
    }

    model.save(model_path, metadata=metadata)
    print(f"\nModel disimpan ke: {model_path}")

if __name__ == "__main__":
    run_training()
