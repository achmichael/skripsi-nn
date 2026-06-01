import sys

from src.pipeline.preprocessing import (
    transform_minmax,
    inverse_transform_target,
)
from src.models.pascabayar import PascabayarModel
from src.models.prabayar import PrabayarModel


MODEL_PATHS = {
    "prabayar": "results/prabayar/models/model_prabayar.json",
    "pascabayar": "results/pascabayar/models/model_pascabayar.json",
}

TARGET_LABELS = {
    "prabayar": "durasi token (hari)",
    "pascabayar": "estimasi biaya listrik (Rp)",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <prabayar|pascabayar>")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if model_type not in MODEL_PATHS:
        print(f"Model type tidak dikenal: {model_type}")
        print("Pilih: prabayar atau pascabayar")
        sys.exit(1)

    loaders = {
        "pascabayar": PascabayarModel,
        "prabayar": PrabayarModel,
    }
    model, metadata = loaders[model_type].load(MODEL_PATHS[model_type])

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    feature_columns = metadata["feature_columns"]
    input_size = len(feature_columns)

    print(f"=== Prediksi {model_type.upper()} ===")
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

    label = TARGET_LABELS[model_type]
    print(f"\nPrediksi {label}: {prediction_original:,.2f}")


if __name__ == "__main__":
    main()
