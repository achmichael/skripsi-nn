import math
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.pipeline.preprocessing import (
    load_and_preprocess,
    train_test_split,
    fit_standard_scaler,
    transform_standard_scaler,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.config.config import config

def benchmark_model():
    model_type = "prabayar"
    print(f"=== Benchmarking PRABAYAR dengan scikit-learn ===")
    cfg = config[model_type]

    # 1. Load dan Preprocess Data
    print("Memuat dataset dan melakukan preprocessing...")
    df, _, _ = load_and_preprocess(cfg["dataset_path"])

    # 2. Ekstrak Fitur dan Target
    x_data, x_cat_data, y_data, feature_cols, embedding_configs, target_col = extract_features_and_target(df, model_type)

    # 3. Split Data
    x_train, x_cat_train, x_test, x_cat_test, y_train, y_test = train_test_split(
        x_data=x_data,
        x_cat_data=x_cat_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )

    # 4. Scaling Fitur
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_test_scaled = transform_standard_scaler(x_test, x_scaler)

    y_train_scaled = y_train
    y_test_scaled = y_test

    # 5. Konfigurasi MLPRegressor
    hidden_layer_sizes = tuple(cfg.get("hidden_layers", (32, 32)))

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=cfg.get("l2_lambda", 0.0001),
        batch_size=cfg.get("batch_size", 16),
        learning_rate_init=cfg.get("learning_rate", 0.001),
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=cfg.get("patience", 10),
        random_state=42,
    )

    # 6. Training
    print(f"Training MLPRegressor (Arsitektur: {hidden_layer_sizes})...")
    mlp.fit(x_train_scaled, y_train_scaled)

    # 7. Prediksi
    preds_orig = mlp.predict(x_test_scaled)

    # 8. Metrik Evaluasi
    mse = mean_squared_error(y_test, preds_orig)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds_orig)
    r2 = r2_score(y_test, preds_orig)

    mape = sum(
        abs(p - a) / max(abs(a), 1)
        for p, a in zip(preds_orig, y_test)
    ) / len(y_test) * 100

    print(f"\nHasil Evaluasi scikit-learn (PRABAYAR):")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R2   : {r2:.4f}")

    print("\n  [Sample 15 Prediksi Teratas]")
    for i in range(min(15, len(y_test))):
        print(f"  Data {i+1:2d} | Aktual: {y_test[i]:>6.2f} hari | Prediksi: {preds_orig[i]:>6.2f} hari")

    print("-" * 60 + "\n")

if __name__ == "__main__":
    benchmark_model()
