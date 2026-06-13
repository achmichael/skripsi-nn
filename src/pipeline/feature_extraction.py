"""
Feature extraction — mengambil fitur dan target dari DataFrame yang sudah dipreprocess.
"""

import pandas as pd
from src.config.config import config


def extract_features_and_target(
    df: pd.DataFrame,
    model_type: str,
) -> tuple[list[list[float]], list[float], list[str], str]:
    """
    Extract features and target dari DataFrame berdasarkan model_type.

    Args:
        df        : DataFrame hasil preprocessing.
        model_type: 'prabayar' atau 'pascabayar'.

    Returns:
        (x_data, y_data, feature_columns, target_column)
    """
    if df.empty:
        raise ValueError("Data kosong.")

    if model_type not in config["features"]:
        raise ValueError(f"Model type '{model_type}' tidak dikenal. Gunakan 'prabayar' atau 'pascabayar'.")

    feature_columns = config["features"][model_type]
    target_column = config[model_type]["target"]

    # Validate columns exist
    available_columns = set(df.columns)
    missing = [col for col in feature_columns if col not in available_columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan di dataset: {missing}")
    if target_column not in available_columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

    x_df = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    if x_df.isna().any().any():
        bad_cols = x_df.columns[x_df.isna().any()].tolist()
        raise ValueError(f"Non-numeric values found in columns: {bad_cols}")
        
    x_data = x_df.values.tolist()
    y_data = df[target_column].values.tolist()

    return x_data, y_data, feature_columns, target_column
