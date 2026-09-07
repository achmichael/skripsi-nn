import sys

from src.pipeline.preprocessing import (
    transform_minmax,
    inverse_transform_target,
)
from src.models.prabayar import PrabayarModel


MODEL_PATH = "results/prabayar/models/model_prabayar.json"
TARGET_LABEL = "durasi token (hari)"


def main():
    model, metadata = PrabayarModel.load(MODEL_PATH)

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    feature_columns = metadata["feature_columns"]
    input_size = len(feature_columns)

    print("=== Prediksi PRABAYAR ===")
    print(f"Masukkan nilai untuk {input_size} fitur:\n")

    input_values = []

    for i in range(input_size):
        feature_name = feature_columns[i]
        value = float(input(f"  {feature_name}: "))
        input_values.append(value)

    x_scaled = transform_minmax([input_values], x_scaler)[0]

    prediction_scaled = model.predict(x_scaled)

    prediction_original = inverse_transform_target(
        prediction_scaled,
        y_scaler,
    )

    print(f"\nPrediksi {TARGET_LABEL}: {prediction_original:,.2f}")


if __name__ == "__main__":
    main()
