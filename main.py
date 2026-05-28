import os
import sys

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
    NeuralNetwork,
    train_model,
    evaluate_model,
)


CONFIGS = {
    "prabayar": {
        "dataset_path": "data/prabayar.csv",
        "model_path": "model_prabayar.json",
        "hidden_size": 32,
        "learning_rate": 0.001,
        "target_label": "durasi token (hari)",
    },
    "pascabayar": {
        "dataset_path": "data/pascabayar.csv",
        "model_path": "model_pascabayar.json",
        "hidden_size": 32,
        "learning_rate": 0.001,
        "target_label": "estimasi biaya (Rp)",
    },
}


def run_training(model_type: str):
    cfg = CONFIGS[model_type]

    if not os.path.exists(cfg["dataset_path"]):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {cfg['dataset_path']}")

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

    # Build model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
    )

    # Train (no epoch limit, early stopping patience=5)
    print("Mulai training...")
    train_model(
        model=model,
        x_train=x_train_scaled,
        y_train=y_train_scaled,
        learning_rate=cfg["learning_rate"],
        patience=5,
        epochs=None,
    )

    # Evaluate
    evaluation = evaluate_model(
        model=model,
        x_test=x_test_scaled,
        y_test=y_test_scaled,
    )

    print(f"\nEvaluasi (skala normalisasi):")
    print(f"  MSE: {evaluation['mse']:.8f}")
    print(f"  MAE: {evaluation['mae']:.8f}")

    # Sample predictions
    print(f"\nContoh prediksi ({cfg['target_label']}):")
    for i in range(min(5, len(x_test_scaled))):
        pred_scaled = model.predict(x_test_scaled[i])
        pred_original = inverse_transform_target(pred_scaled, y_scaler)
        actual = y_test[i]
        print(f"  Data {i + 1}: Prediksi = {pred_original:.2f}, Aktual = {actual:.2f}")

    # Save
    metadata = {
        "model_type": model_type,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_size": input_size,
        "hidden_size": cfg["hidden_size"],
    }

    model.save(cfg["model_path"], metadata=metadata)
    print(f"\nModel disimpan ke: {cfg['model_path']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <prabayar|pascabayar|all>")
        sys.exit(1)

    choice = sys.argv[1].lower()

    if choice == "all":
        for model_type in CONFIGS:
            run_training(model_type)
            print("\n" + "=" * 60 + "\n")
    elif choice in CONFIGS:
        run_training(choice)
    else:
        print(f"Model type tidak dikenal: {choice}")
        print("Pilih: prabayar, pascabayar, atau all")
        sys.exit(1)


if __name__ == "__main__":
    main()
