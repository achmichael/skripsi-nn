import json
import math
import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

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
    """Plot dan simpan training loss curve berisi train loss dan validation loss."""
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        raise ValueError("Train loss kosong. Tidak bisa membuat kurva loss.")

    trained_epochs = len(train_loss)
    epoch_axis = list(range(1, trained_epochs + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(
        epoch_axis,
        train_loss,
        linewidth=1.5,
        label="Train Loss",
    )

    if val_loss:
        plt.plot(
            epoch_axis,
            val_loss,
            linewidth=1.5,
            label="Validation Loss",
        )

    plt.title(
        f"Training vs Validation Loss - {model_type.upper()} ({trained_epochs} epoch)",
        fontsize=14,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.xlim(1, trained_epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Loss curve disimpan ke: {save_path} (epoch aktual: {trained_epochs})")
    
def save_prediction_scatter(y_actual, y_predicted, save_path: str, model_type: str, label: str):
    """Plot scatter prediksi vs aktual."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_predicted, alpha=0.5, s=20, color="#4CAF50")
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
    plt.title(f"Prediksi vs Aktual - {model_type.upper()}", fontsize=14)
    plt.xlabel(f"Aktual ({label})", fontsize=12)
    plt.ylabel(f"Prediksi ({label})", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot disimpan ke: {save_path}")


def save_error_distribution(errors, save_path: str, model_type: str):
    """Plot distribusi error."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, color="#FF9800", edgecolor="black", alpha=0.7)
    plt.title(f"Distribusi Error - {model_type.upper()}", fontsize=14)
    plt.xlabel("Error (Prediksi - Aktual)", fontsize=12)
    plt.ylabel("Frekuensi", fontsize=12)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    plt.grid(True, alpha=0.3)
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
    )
    print(f"Training selesai. Total epoch aktual: {len(history)}")

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
    metrics_data = {
        "model_type": model_type,
        "arsitektur": layer_sizes,
        "learning_rate": cfg["learning_rate"],
        "total_epochs": len(history),
        "final_loss": history[-1],
        "best_loss": min(history),
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
