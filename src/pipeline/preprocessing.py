import csv
import random
from src.config.config import config


def load_csv(path: str) -> list[dict]:
    """Load CSV and encode categorical columns to numeric using config mappings."""
    rows = []
    ordinal = config["ordinal_encoding"]
    numeric_special = config["numeric_special"]

    with open(path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cleaned_row = {}

            for key, value in row.items():
                cleaned_key = key.strip()
                raw = value.strip() if value else ""

                # Check ordinal encoding first
                if cleaned_key in ordinal:
                    mapping = ordinal[cleaned_key]
                    if raw in mapping:
                        cleaned_row[cleaned_key] = float(mapping[raw])
                    else:
                        cleaned_row[cleaned_key] = 0.0
                    continue

                # Check numeric special cases (e.g. "Tidak tahu" in Daya_Listrik_Rumah_VA)
                if cleaned_key in numeric_special and raw in numeric_special[cleaned_key]:
                    cleaned_row[cleaned_key] = float(numeric_special[cleaned_key][raw])
                    continue

                # Try parse as float
                if raw == "":
                    cleaned_row[cleaned_key] = 0.0
                else:
                    try:
                        cleaned_row[cleaned_key] = float(raw.replace(",", "."))
                    except ValueError:
                        # Non-numeric, non-encoded column (e.g. Timestamp, Nama) -> keep as string
                        cleaned_row[cleaned_key] = raw

            rows.append(cleaned_row)

    return rows


def train_test_split(
    x_data: list[list[float]],
    y_data: list[float],
    test_ratio: float = 0.2,
    seed: int = 42,
):
    combined = list(zip(x_data, y_data))

    random.seed(seed)
    random.shuffle(combined)

    test_size = int(len(combined) * test_ratio)

    test_data = combined[:test_size]
    train_data = combined[test_size:]

    x_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]

    x_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    return x_train, x_test, y_train, y_test


def fit_minmax_scaler(x_data: list[list[float]]) -> dict:
    total_features = len(x_data[0])

    minimums = []
    maximums = []

    for feature_index in range(total_features):
        column_values = [row[feature_index] for row in x_data]

        minimums.append(min(column_values))
        maximums.append(max(column_values))

    return {
        "min": minimums,
        "max": maximums,
    }


def transform_minmax(x_data: list[list[float]], scaler: dict) -> list[list[float]]:
    scaled_data = []

    for row in x_data:
        scaled_row = []

        for i, value in enumerate(row):
            min_value = scaler["min"][i]
            max_value = scaler["max"][i]

            if max_value == min_value:
                scaled_value = 0.0
            else:
                scaled_value = (value - min_value) / (max_value - min_value)

            scaled_row.append(scaled_value)

        scaled_data.append(scaled_row)

    return scaled_data


def fit_target_scaler(y_data: list[float]) -> dict:
    return {
        "min": min(y_data),
        "max": max(y_data),
    }


def transform_target(y_data: list[float], scaler: dict) -> list[float]:
    min_value = scaler["min"]
    max_value = scaler["max"]

    scaled = []

    for value in y_data:
        if max_value == min_value:
            scaled.append(0.0)
        else:
            scaled.append((value - min_value) / (max_value - min_value))

    return scaled


def inverse_transform_target(value: float, scaler: dict) -> float:
    return value * (scaler["max"] - scaler["min"]) + scaler["min"]
