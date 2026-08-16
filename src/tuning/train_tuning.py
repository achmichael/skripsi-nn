"""
train_tuning.py

File pendukung berisi fungsi utilitas dan pipeline evaluasi untuk digunakan
oleh script hyperparameter tuning (seperti grid_search.py).
"""

import math
import numpy as np

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
from src.config.config import config


# =====================================================================
# METRICS EVALUATION
# =====================================================================

def is_invalid_number(val: float) -> bool:
    return math.isnan(val) or math.isinf(val)


def compute_rmse(y_true, y_pred) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    return math.sqrt(mse) if not is_invalid_number(mse) and mse > 0 else float("inf")


def compute_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true, y_pred) -> float:
    # Hindari division by zero
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def compute_r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def evaluate_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


def make_divergent_metrics(params: dict, start_loss: float) -> dict:
    return {
        "rmse": float("inf"),
        "mae": float("inf"),
        "mape": float("inf"),
        "r2": -float("inf"),
        "diverged": True,
        "loss_start": start_loss,
        "params": params,
    }


# =====================================================================
# PREDICTION WRAPPERS
# =====================================================================

def predict_mlp(model, x_data):
    """Prediksi batch menggunakan model MLP standar."""
    X = np.array(x_data, dtype=np.float32)
    return model.predict(X).flatten()


def predict_pv(model, x_data):
    """Prediksi per sampel menggunakan PascabayarPlaceValueModel."""
    preds = []
    for row in x_data:
        x_i = row if isinstance(row, list) else row.tolist()
        preds.append(model.predict(x_i))
    return np.array(preds)


# =====================================================================
# DATA LOADING PIPELINE
# =====================================================================

def load_data(model_type: str):
    """
    Load CSV, jalankan pipeline preprocessing, split train/test (sebagai val),
    lalu lakukan standard scaling fitur dan target scaling.
    Mengembalikan data siap latih dan label aktual untuk metrik.
    """
    cfg = config[model_type]
    
    # 1. Load dan preprocess CSV
    df, _ = load_and_preprocess(cfg["dataset_path"])
    
    # 2. Ekstrak target dan fitur (sesuai spesifikasi config)
    x_data, y_data, feat_cols, target_col = extract_features_and_target(df, model_type)
    n_features = len(feat_cols)
    
    # 3. Split 80/20. Pada tuning, test set digunakan sebagai validation set.
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_ratio=0.2, seed=42
    )
    
    # 4. Standard scaler pada fitur (z-score)
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_val_scaled = transform_standard_scaler(x_val, x_scaler)
    
    # 5. Target scaler (menggunakan log transform jika dikonfigurasi)
    use_log = cfg.get("use_log_transform", False)
    y_scaler = fit_target_scaler(y_train, use_log=use_log)
    y_train_scaled = transform_target(y_train, y_scaler)
    y_val_scaled = transform_target(y_val, y_scaler)
    
    # Validation data (original scale) untuk metrik (RMSE, MAE, R2 asli)
    y_val_original = y_val
    
    return (
        x_train_scaled, 
        x_val_scaled, 
        y_train_scaled, 
        y_val_scaled, 
        y_val_original, 
        n_features, 
        y_scaler, 
        feat_cols
    )
