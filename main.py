import json
import math
import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.ticker as mticker

from src.pipeline.preprocessing import (
    load_csv,
    train_test_split,
    fit_minmax_scaler,
    transform_minmax,
    fit_target_scaler,
    transform_target,
    inverse_transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.utils.core import (
    train_model,
    evaluate_model,
)
from src.models.pascabayar import PascabayarModel
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
    """Plot scatter prediksi vs aktual dengan satuan juta Rp, zona toleransi, dan statistik."""
    # Konversi ke juta Rp agar sumbu mudah dibaca
    scale = 1_000_000
    actual_m  = [v / scale for v in y_actual]
    pred_m    = [v / scale for v in y_predicted]

    # Hitung R²
    mean_actual = sum(actual_m) / len(actual_m)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_m)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual_m, pred_m))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Hitung persen prediksi dalam toleransi ±20%
    within_20pct = sum(
        1 for a, p in zip(actual_m, pred_m)
        if a != 0 and abs(p - a) / abs(a) <= 0.20
    )
    pct_within = within_20pct / len(actual_m) * 100

    all_vals = actual_m + pred_m
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
        for a, p in zip(actual_m, pred_m)
    ]
    ax.scatter(actual_m, pred_m, c=colors, alpha=0.75, s=35, zorder=3)

    # Garis ideal y = x
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)", zorder=4)

    # Dummy scatter untuk legend
    ax.scatter([], [], c="#4CAF50", s=35, label=f"Dalam toleransi ({within_20pct}/{len(actual_m)} titik)")
    ax.scatter([], [], c="#F44336", s=35, label="Di luar toleransi")

    # Format sumbu dengan satuan jt Rp
    def fmt_juta(x, _):
        return f"{x:,.0f} jt"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_juta))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_juta))
    ax.tick_params(axis='x', rotation=30)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Aktual ({label}) — juta Rp", fontsize=12)
    ax.set_ylabel(f"Prediksi ({label}) — juta Rp", fontsize=12)
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


def save_error_distribution(errors, save_path: str, model_type: str):
    """Plot distribusi error dalam juta Rp dengan statistik ringkas dan zona under/overprediction."""
    scale = 1_000_000
    errors_m = [e / scale for e in errors]

    n = len(errors_m)
    mean_e  = sum(errors_m) / n
    sorted_e = sorted(errors_m)
    mid = n // 2
    median_e = (sorted_e[mid - 1] + sorted_e[mid]) / 2 if n % 2 == 0 else sorted_e[mid]
    variance = sum((e - mean_e) ** 2 for e in errors_m) / n
    std_e = variance ** 0.5

    n_over  = sum(1 for e in errors_m if e > 0)   # prediksi lebih tinggi
    n_under = sum(1 for e in errors_m if e < 0)   # prediksi lebih rendah
    n_exact = n - n_over - n_under

    fig, ax = plt.subplots(figsize=(11, 6))

    # Zona warna background: merah=underprediction, hijau=overprediction
    x_min = min(errors_m)
    x_max = max(errors_m)
    ax.axvspan(x_min, 0, alpha=0.06, color="#F44336", label="_nolegend_")
    ax.axvspan(0, x_max, alpha=0.06, color="#4CAF50", label="_nolegend_")

    # Histogram
    ax.hist(errors_m, bins=25, color="#5C6BC0", edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)

    # Garis referensi
    ax.axvline(x=0,      color="#212121", linestyle="-",  linewidth=2.0, label="Zero error",       zorder=4)
    ax.axvline(x=mean_e, color="#FF5722", linestyle="--", linewidth=1.8, label=f"Mean = {mean_e:+,.1f} jt",   zorder=4)
    ax.axvline(x=median_e, color="#FFC107", linestyle=":", linewidth=1.8, label=f"Median = {median_e:+,.1f} jt", zorder=4)

    # Label zona di atas grafik
    ax.text(0.02, 0.97, f"Underprediksi\n({n_under} titik, {n_under/n*100:.1f}%)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            color="#C62828", fontweight="bold")
    ax.text(0.98, 0.97, f"Overprediksi\n({n_over} titik, {n_over/n*100:.1f}%)",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            color="#2E7D32", fontweight="bold")

    # Format sumbu X
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f} jt"))
    ax.tick_params(axis='x', rotation=20)

    ax.set_xlabel("Error = Prediksi − Aktual (juta Rp)", fontsize=12)
    ax.set_ylabel("Frekuensi", fontsize=12)
    ax.set_title(
        f"Distribusi Error Prediksi — {model_type.upper()}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Kotak statistik
    stats_text = (
        f"n = {n} data\n"
        f"Std Dev = {std_e:,.1f} jt\n"
        f"Min = {min(errors_m):+,.1f} jt\n"
        f"Max = {max(errors_m):+,.1f} jt"
    )
    ax.text(
        0.98, 0.60, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error distribution disimpan ke: {save_path}")


def save_metrics_json(metrics: dict, save_path: str):
    """Simpan metrik evaluasi ke JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Metrics JSON disimpan ke: {save_path}")


def run_training(model_type: str):
    cfg = config[model_type]

    if not os.path.exists(cfg["dataset_path"]):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {cfg['dataset_path']}")

    # Buat folder results
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    os.makedirs(cfg["metrics_dir"], exist_ok=True)

    print(f"=== Training model {model_type.upper()} ===\n")

    # Load & encode
    rows = load_csv(cfg["dataset_path"])
    print(f"Total data: {len(rows)} baris")

    # Extract features & target
    x_data, y_data, feature_columns, target_column = extract_features_and_target(
        rows=rows,
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
    print(f"Train: {len(x_train)}, Test: {len(x_test)}\n")

    # Scale
    x_scaler = fit_minmax_scaler(x_train)
    x_train_scaled = transform_minmax(x_train, x_scaler)
    x_test_scaled = transform_minmax(x_test, x_scaler)

    y_scaler = fit_target_scaler(y_train)
    y_train_scaled = transform_target(y_train, y_scaler)
    y_test_scaled = transform_target(y_test, y_scaler)

    # Build layer sizes: [input, ...hidden..., 1]
    layer_sizes = [input_size] + cfg["hidden_layers"] + [1]
    print(f"Arsitektur: {layer_sizes}")

    # Build model berdasarkan model_type
    if model_type == "pascabayar":
        model: NeuralNetwork = PascabayarModel(
            layer_sizes=layer_sizes,
            seed=42,
            clip_value=cfg["clip_value"],
        )
    else:
        model = PrabayarModel(
            layer_sizes=layer_sizes,
            seed=42,
            clip_value=cfg["clip_value"],
        )

    # Train
    print("Mulai training...")
    history = train_model(
        model=model,
        x_train=x_train_scaled,
        y_train=y_train_scaled,
        learning_rate=cfg["learning_rate"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        epochs=None,
        x_val=x_test_scaled,
        y_val=y_test_scaled,
    )
    total_epochs = len(history["train_loss"])
    print(f"Training selesai. Total epoch aktual: {total_epochs}")

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
    print(f"\nContoh prediksi ({cfg['target_label']}):")
    for i in range(min(5, len(x_test_scaled))):
        pred = preds_orig[i]
        actual = y_test[i]
        ratio = pred / actual if actual != 0 else float('inf')
        print(
            f"  Data {i + 1}: Prediksi = {pred:,.2f}, "
            f"Aktual = {actual:,.2f}, "
            f"Rasio = {ratio:.2f}"
        )

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
    save_error_distribution(
        errors=errors_orig,
        save_path=os.path.join(metrics_dir, "error_distribution.png"),
        model_type=model_type,
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
        "layer_sizes": layer_sizes,
    }

    model.save(cfg["model_path"], metadata=metadata)
    print(f"\nModel disimpan ke: {cfg['model_path']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <prabayar|pascabayar|all>")
        sys.exit(1)

    choice = sys.argv[1].lower()

    if choice == "all":
        for model_type in config:
            run_training(model_type)
            print("\n" + "=" * 60 + "\n")
    elif choice in config:
        run_training(choice)
    else:
        print(f"Model type tidak dikenal: {choice}")
        print("Pilih: prabayar, pascabayar, atau all")
        sys.exit(1)


if __name__ == "__main__":
    main()
