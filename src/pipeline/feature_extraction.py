"""
Feature extraction — mengambil fitur dan target dari DataFrame yang sudah dipreprocess.
"""

import pandas as pd
from src.config.config import config


def extract_features_and_target(
    df: pd.DataFrame,
    model_type: str,
) -> tuple[list[list[float]], list[dict[str, int]], list[float], list[str], list[dict], str]:
    """
    Extract features and target dari DataFrame berdasarkan model_type.

    Args:
        df        : DataFrame hasil preprocessing.
        model_type: 'prabayar' atau 'pascabayar'.

    Returns:
        (x_data, x_cat_data, y_data, feature_columns, embedding_configs, target_column)
    """
    if df.empty:
        raise ValueError("Data kosong.")

    if model_type not in config["features"]:
        raise ValueError(f"Model type '{model_type}' tidak dikenal. Gunakan 'prabayar' atau 'pascabayar'.")

    feature_columns = config["features"][model_type]
    embedding_configs = config.get("embedding_features", {}).get(model_type, [])
    target_column = config[model_type]["target"]

    # Ambil daftar nama kolom embedding
    embedding_columns = [cfg["name"] for cfg in embedding_configs]

    # Validate columns exist
    available_columns = set(df.columns)
    missing = [col for col in feature_columns if col not in available_columns]
    if missing:
        raise ValueError(f"Kolom fitur numerik tidak ditemukan di dataset: {missing}")
        
    missing_emb = [col for col in embedding_columns if col not in available_columns]
    if missing_emb:
        raise ValueError(f"Kolom fitur embedding tidak ditemukan di dataset: {missing_emb}")
        
    if target_column not in available_columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

    # 1. Ekstrak Numerik
    x_df = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    if x_df.isna().any().any():
        bad_cols = x_df.columns[x_df.isna().any()].tolist()
        raise ValueError(f"Non-numeric values found in columns: {bad_cols}")
    x_data = x_df.values.tolist()

    # 2. Ekstrak Kategorikal untuk Embedding (list of dict per sample)
    x_cat_data = []
    if embedding_columns:
        # Konversi ke integer agar aman untuk indexing list/array
        cat_df = df[embedding_columns].fillna(0).astype(int)
        # Ubah tiap baris menjadi dictionary {nama_fitur: nilai_integer}
        x_cat_data = cat_df.to_dict(orient='records')
    else:
        # Jika tidak ada embedding, return list of empty dicts
        x_cat_data = [{} for _ in range(len(x_data))]
        
    # 3. Ekstrak Target
    y_data = df[target_column].values.tolist()

    return x_data, x_cat_data, y_data, feature_columns, embedding_configs, target_column

