from src.config.config import config


def extract_features_and_target(
    rows: list[dict],
    model_type: str,
) -> tuple[list[list[float]], list[float], list[str], str]:
    """
    Extract features and target from rows based on model_type ('prabayar' or 'pascabayar').
    Uses config to determine which columns are features and which is target.
    """
    if not rows:
        raise ValueError("Data kosong.")

    if model_type not in config["features"]:
        raise ValueError(f"Model type '{model_type}' tidak dikenal. Gunakan 'prabayar' atau 'pascabayar'.")

    feature_columns = config["features"][model_type]
    target_column = config["target"][model_type]

    # Validate columns exist
    available_columns = set(rows[0].keys())
    missing = [col for col in feature_columns if col not in available_columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan di dataset: {missing}")
    if target_column not in available_columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

    x_data = []
    y_data = []

    for row in rows:
        features = []
        for col in feature_columns:
            val = row[col]
            features.append(float(val))

        x_data.append(features)
        y_data.append(float(row[target_column]))

    return x_data, y_data, feature_columns, target_column
