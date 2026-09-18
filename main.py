import json
import math
import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.ticker as mticker

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
from src.utils.core import (
    train_model,
    evaluate_model,
)
from src.models.prabayar import PrabayarModel
from src.models.neural_network import NeuralNetwork
from src.config.config import config

def save_loss_curve(history: dict, save_path: str, model_type: str):
    """Plot training loss curve dengan titik per epoch, anotasi best loss, dan val loss."""
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])

    if not train_loss:
        raise ValueError("Train loss kosong. Tidak bisa membuat kurva loss.")

    trained_epochs = len(train_loss)
    epoch_axis = list(range(1, trained_epochs + 1))

    # Ukuran marker: kecil jika banyak epoch agar tidak berantakan
    marker_size = max(1.5, 5.0 - trained_epochs * 0.008)

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Train loss ---
    ax.plot(
        epoch_axis, train_loss,
        color="#1565C0", linewidth=1.5,
        marker="o", markersize=marker_size, markerfacecolor="#1565C0",
        label="Train Loss",
        zorder=3,
    )

    # --- Validation loss ---
    if val_loss:
        ax.plot(
            epoch_axis, val_loss,
            color="#E53935", linewidth=1.5,
            marker="o", markersize=marker_size, markerfacecolor="#E53935",
            label="Validation Loss",
            zorder=3,
        )

    # --- Titik best train loss ---
    best_epoch = train_loss.index(min(train_loss)) + 1
    best_val   = min(train_loss)
    ax.scatter(
        [best_epoch], [best_val],
        color="#FFB300", s=60, zorder=5,
        label=f"Best Train Loss (epoch {best_epoch}: {best_val:.6f})",
    )

    # --- Garis vertikal early stopping ---
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

    print(f"Loss curve disimpan ke: {save_path} (epoch aktual: {trained_epochs})")

def save_prediction_scatter(y_actual, y_predicted, save_path: str, model_type: str, label: str):
    """Plot scatter prediksi vs aktual dengan zona toleransi dan statistik."""
    actual_vals = list(y_actual)
    pred_vals   = list(y_predicted)

    # Hitung R²
    mean_actual = sum(actual_vals) / len(actual_vals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_vals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual_vals, pred_vals))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Hitung persen prediksi dalam toleransi ±20%
    within_20pct = sum(
        1 for a, p in zip(actual_vals, pred_vals)
        if a != 0 and abs(p - a) / abs(a) <= 0.20
    )
    pct_within = within_20pct / len(actual_vals) * 100

    all_vals = actual_vals + pred_vals
    min_val  = min(all_vals)
    max_val  = max(all_vals)
    margin   = (max_val - min_val) * 0.05
    lo = min_val - margin
    hi = max_val + margin

    fig, ax = plt.subplots(figsize=(9, 8))

    # Zona toleransi ±20%
    ref = [lo, hi]
    ax.fill_between(
        ref,
        [v * 0.80 for v in ref],
        [v * 1.20 for v in ref],
        alpha=0.12,
        color="#2196F3",
        label="Toleransi ±20%",
    )

    # Scatter points — warna berdasarkan masuk/keluar toleransi
    colors = [
        "#4CAF50" if abs(a) > 0 and abs(p - a) / abs(a) <= 0.20 else "#F44336"
        for a, p in zip(actual_vals, pred_vals)
    ]
    ax.scatter(actual_vals, pred_vals, c=colors, alpha=0.75, s=35, zorder=3)

    # Garis ideal y = x
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)", zorder=4)

    # Dummy scatter untuk legend
    ax.scatter([], [], c="#4CAF50", s=35, label=f"Dalam toleransi ({within_20pct}/{len(actual_vals)} titik)")
    ax.scatter([], [], c="#F44336", s=35, label="Di luar toleransi")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Aktual ({label})", fontsize=12)
    ax.set_ylabel(f"Prediksi ({label})", fontsize=12)
    ax.set_title(f"Prediksi vs Aktual — {model_type.upper()}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Kotak statistik
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
    print(f"Scatter plot disimpan ke: {save_path}")


def plot_error_distribution(
    y_true: list[float],
    y_pred: list[float],
    mae_value: float,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Membuat visualisasi distribusi error prediksi Neural Network.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    errors = y_p - y_t

    n = len(errors)
    if n == 0:
        return

    errors_scaled = errors
    mae_scaled = mae_value

    mean_e = np.mean(errors_scaled)
    median_e = np.median(errors_scaled)
    std_e = np.std(errors_scaled)
    min_e = np.min(errors_scaled)
    max_e = np.max(errors_scaled)
    range_e = max_e - min_e

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.hist(errors_scaled, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    ticks = np.linspace(min_e, max_e, 10)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{v:+.1f}" for v in ticks])

    ax1.set_xlabel("Error (hari) [Prediksi - Aktual]", fontsize=12)
    ax1.set_ylabel("Frekuensi", fontsize=12)
    ax1.set_title("Distribusi Error Prediksi", fontsize=14, fontweight='bold')

    ax1.axvspan(-mae_scaled, mae_scaled, color='lightgreen', alpha=0.3, label='±MAE zone')

    offset = max(range_e * 0.005, 1e-5)
    mean_offset = 0.0
    median_offset = 0.0

    if abs(mean_e - 0.0) <= offset * 2:
        mean_offset = offset

    if abs(median_e - 0.0) <= offset * 2 or abs(median_e - mean_e) <= offset * 2:
        median_offset = -offset

    ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='Zero Error')
    ax1.axvline(mean_e + mean_offset, color='orange', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(median_e + median_offset, color='yellow', linestyle=':', linewidth=2, label='Median')

    ax1.legend(loc='upper left', fontsize=10)

    stats_text = (
        f"n = {n} data\n"
        f"Mean = {mean_e:+.1f} hari\n"
        f"Median = {median_e:+.1f} hari\n"
        f"Std Dev = {std_e:.1f} hari\n"
        f"Range: [{min_e:+.1f}, {max_e:+.1f}] hari"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Error distribution disimpan ke: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_metrics_json(metrics: dict, save_path: str):
    """Simpan metrik evaluasi ke JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Metrics JSON disimpan ke: {save_path}")


def run_training():
    model_type = "prabayar"
    cfg = config[model_type]

    if not os.path.exists(cfg["dataset_path"]):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {cfg['dataset_path']}")

    # Buat folder results
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    os.makedirs(cfg["metrics_dir"], exist_ok=True)

    print(f"=== Training model PRABAYAR ===\n")
    print(f"[INFO] Log transform: {cfg.get('use_log_transform', False)}")

    rows, minmax_scaler_params, prob_params = load_and_preprocess(cfg["dataset_path"])
    print(f"Total data: {len(rows)} baris")

    # Extract features & target
    x_data, y_data, feature_columns, target_column = extract_features_and_target(
        df=rows,
        model_type=model_type,
    )
    input_size = len(feature_columns)

    print(f"Fitur Numerik: {input_size} kolom")
    print(f"Target: {target_column}\n")

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data=x_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )

    print(f"Train: {len(x_train)}, Test: {len(x_test)}\n")

    # Scale
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_test_scaled = transform_standard_scaler(x_test, x_scaler)

    y_scaler = fit_target_scaler(y_train, use_log=cfg.get("use_log_transform", False))
    y_train_scaled = transform_target(y_train, y_scaler)
    y_test_scaled = transform_target(y_test, y_scaler)

    # Logging sample data preprocessed
    print("\n--- SAMPLE DATA PREPROCESSED (3 Baris Pertama) ---")
    for i in range(min(3, len(x_train_scaled))):
        print(f"\n[BARIS {i+1}]")
        print("  [FITUR]")
        for j, col_name in enumerate(feature_columns):
            print(f"    {col_name}: {x_train_scaled[i][j]:.4f}")
        print("  [TARGET]")
        print(f"    {target_column}: {y_train_scaled[i]:.4f}")
    print("-" * 50 + "\n")

    # Logging nilai mean dari setiap fitur (untuk Integrated Gradients baseline)
    import numpy as np
    x_train_array = np.array(x_train_scaled)
    feature_means = np.mean(x_train_array, axis=0)
    
    print("\n=== FEATURE MEANS (Training Data - Scaled) ===")
    print("Nilai mean ini dapat digunakan sebagai baseline untuk Integrated Gradients\n")
    for i, col_name in enumerate(feature_columns):
        print(f"  {col_name:<30}: {feature_means[i]:>10.6f}")
    print("\n" + "=" * 70 + "\n")

    # Build layer sizes: [input, ...hidden..., 1]
    layer_sizes = [input_size] + cfg["hidden_layers"] + [1]
    print(f"Arsitektur Dense: {layer_sizes}")

    model = PrabayarModel(
        layer_sizes=layer_sizes,
        seed=42,
        clip_value=cfg["clip_value"],
        l2_lambda=cfg.get("l2_lambda", 0.0),
        asymmetric_alpha=cfg.get("asymmetric_alpha", 0.5),
        l1_lambda_input=cfg.get("l1_lambda_input")
    )

    # Train
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
        use_log=cfg.get("use_log_transform", False),
        model_type=model_type,
    )
    total_epochs = len(history["train_loss"])
    print(f"Training selesai. Total epoch aktual: {total_epochs}")

    # Logging bobot model untuk Integrated Gradients
    print("\n=== MODEL WEIGHTS INFO (untuk Integrated Gradients) ===")
    for i, w in enumerate(model.weights):
        print(f"  Weight Layer {i+1} shape: {w.shape}")
        print(f"    - Mean: {np.mean(w):.6f}")
        print(f"    - Std:  {np.std(w):.6f}")
        print(f"    - Min:  {np.min(w):.6f}")
        print(f"    - Max:  {np.max(w):.6f}")
    print("=" * 70 + "\n")

    contributions = model.get_feature_contributions()

    # Gabungkan nama fitur yang digunakan oleh model dengan skor
    importance = sorted(zip(feature_columns, contributions), key=lambda x: x[1], reverse=True)

    print("Tingkat Kontribusi Fitur:")
    for name, score in importance:
        print(f"{name:<20}: {score * 100:.2f}%")

    # Evaluate
    evaluation = evaluate_model(
        model=model,
        x_test=x_test_scaled,
        y_test=y_test_scaled,
    )

    print(f"\nEvaluasi (skala normalisasi):")
    print(f"  MSE: {evaluation['mse']:.8f}")
    print(f"  MAE: {evaluation['mae']:.8f}")

    # MAPE & RMSE in original scale
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

    print(f"  MAPE: {mape:.2f}%")
    print(f"\nEvaluasi (skala asli):")
    print(f"  MSE: {mse_orig:,.4f}")
    print(f"  RMSE: {rmse_orig:,.4f}")
    print(f"  MAE: {mae_orig:,.4f}")

    # Sample predictions
    print(f"\n--- Perbandingan Prediksi vs Aktual (15 Data Pertama) [{cfg['target_label']}] ---")
    for i in range(min(15, len(x_test_scaled))):
        pred = preds_orig[i]
        actual = y_test[i]
        selisih = errors_orig[i]
        print(f"Data {i+1:2d} | Aktual: {actual:>12,.2f} | Prediksi: {pred:>12,.2f} | Selisih: {selisih:>12,.2f}")

    # === Simpan feature means untuk Integrated Gradients ===
    feature_means_data = {
        "feature_columns": feature_columns,
        "feature_means": feature_means.tolist(),
        "description": "Mean values of each feature from training data (scaled). Use as baseline for Integrated Gradients.",
    }
    feature_means_path = os.path.join(cfg["metrics_dir"], "feature_means.json")
    with open(feature_means_path, "w", encoding="utf-8") as f:
        json.dump(feature_means_data, f, indent=4, ensure_ascii=False)
    print(f"Feature means disimpan ke: {feature_means_path}")

    # === Simpan bobot model untuk Integrated Gradients ===
    model_weights_data = {
        "layer_sizes": layer_sizes,
        "weights": [
            {
                "layer": i + 1,
                "shape": list(w.shape),
                "values": w.tolist(),
                "statistics": {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                }
            }
            for i, w in enumerate(model.weights)
        ],
        "biases": [
            {
                "layer": i + 1,
                "shape": list(b.shape),
                "values": b.tolist(),
            }
            for i, b in enumerate(model.biases)
        ],
        "description": "Model weights and biases for Integrated Gradients computation.",
    }
    weights_path = os.path.join(cfg["metrics_dir"], "model_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(model_weights_data, f, indent=4, ensure_ascii=False)
    print(f"Model weights disimpan ke: {weights_path}\n")

    # === Simpan metrics ===
    metrics_dir = cfg["metrics_dir"]

    # 1. Training loss curve
    save_loss_curve(
        history=history,
        save_path=os.path.join(metrics_dir, "training_loss_curve.png"),
        model_type=model_type,
    )

    # 2. Scatter prediksi vs aktual
    save_prediction_scatter(
        y_actual=y_test,
        y_predicted=preds_orig,
        save_path=os.path.join(metrics_dir, "prediction_vs_actual.png"),
        model_type=model_type,
        label=cfg["target_label"],
    )

    # 3. Error distribution
    plot_error_distribution(
        y_true=y_test,
        y_pred=preds_orig,
        mae_value=mae_orig,
        save_path=os.path.join(metrics_dir, "error_distribution.png"),
        show=False,
    )

    # 4. Metrics JSON
    train_loss_list = history["train_loss"]
    val_loss_list = history["val_loss"]
    metrics_data = {
        "model_type": model_type,
        "arsitektur": layer_sizes,
        "learning_rate": cfg["learning_rate"],
        "total_epochs": total_epochs,
        "final_train_loss": train_loss_list[-1],
        "best_train_loss": min(train_loss_list),
        "final_val_loss": val_loss_list[-1] if val_loss_list else None,
        "best_val_loss": min(val_loss_list) if val_loss_list else None,
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
    save_metrics_json(
        metrics=metrics_data,
        save_path=os.path.join(metrics_dir, "evaluation_metrics.json"),
    )

    # === Simpan model ===
    metadata = {
        "model_type": model_type,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "minmax_scaler_params": minmax_scaler_params,
        "prob_params": prob_params,  # Probability encoding params
        "layer_sizes": layer_sizes,
        "feature_means": feature_means.tolist(),  # Untuk Integrated Gradients baseline
    }

    model.save(cfg["model_path"], metadata=metadata)
    print(f"\nModel disimpan ke: {cfg['model_path']}")


def main():
    run_training()


if __name__ == "__main__":
    main()
