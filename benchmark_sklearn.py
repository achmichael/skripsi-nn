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

def benchmark_model(model_type):
    print(f"=== Benchmarking {model_type.upper()} dengan scikit-learn ===")
    cfg = config[model_type]
    
    # 1. Load dan Preprocess Data
    print("Memuat dataset dan melakukan preprocessing...")
    df = load_and_preprocess(cfg["dataset_path"])
    
    # 2. Ekstrak Fitur dan Target
    x_data, y_data, feature_cols, target_col = extract_features_and_target(df, model_type)
    
    # 3. Split Data (Sama dengan pembagian pada model native)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data=x_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )
    
    # 4. Scaling Fitur (Menggunakan StandardScaler)
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_test_scaled = transform_standard_scaler(x_test, x_scaler)
    
    # 5. Scaling Target 
    # Untuk pascabayar nilainya sangat besar (Rupiah), kita scale down agar model scikit-learn konvergen dengan baik,
    # seperti yang dilakukan pada model native.
    if model_type == "pascabayar":
        y_train_scaled = [y / 1_000_000.0 for y in y_train]
        y_test_scaled = [y / 1_000_000.0 for y in y_test]
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test
        
    # 6. Konfigurasi MLPRegressor scikit-learn
    hidden_layer_sizes = tuple(cfg.get("hidden_layers", (32, 32)))
    
    # Mengambil hyperparameter dari config agar seimbang/apple-to-apple sebisa mungkin
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
    
    # 7. Training Model
    print(f"Training MLPRegressor (Arsitektur: {hidden_layer_sizes})...")
    mlp.fit(x_train_scaled, y_train_scaled)
    
    # 8. Prediksi pada data test
    preds_scaled = mlp.predict(x_test_scaled)
    
    # Kembalikan ke skala asli jika sebelumnya di-scale
    if model_type == "pascabayar":
        preds_orig = [p * 1_000_000.0 for p in preds_scaled]
    else:
        preds_orig = preds_scaled
        
    # 9. Hitung Metrik Evaluasi
    mse = mean_squared_error(y_test, preds_orig)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds_orig)
    r2 = r2_score(y_test, preds_orig)
    
    # Menghitung MAPE secara manual untuk skala asli (hindari pembagian nol)
    mape = sum(
        abs(p - a) / max(abs(a), 1)
        for p, a in zip(preds_orig, y_test)
    ) / len(y_test) * 100
    
    print(f"\nHasil Evaluasi scikit-learn ({model_type.upper()}):")
    if model_type == "pascabayar":
        print(f"  MSE  : Rp {mse:,.4f}")
        print(f"  RMSE : Rp {rmse:,.4f}")
        print(f"  MAE  : Rp {mae:,.4f}")
    else:
        print(f"  MSE  : {mse:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R2   : {r2:.4f}")
    
    # Optional: Tampilkan 15 data perbandingan
    print("\n  [Sample 15 Prediksi Teratas]")
    for i in range(min(15, len(y_test))):
        if model_type == "pascabayar":
            print(f"  Data {i+1:2d} | Aktual: Rp {y_test[i]:>10,.0f} | Prediksi: Rp {preds_orig[i]:>10,.0f}")
        else:
            print(f"  Data {i+1:2d} | Aktual: {y_test[i]:>6.2f} hari | Prediksi: {preds_orig[i]:>6.2f} hari")
            
    print("-" * 60 + "\n")

if __name__ == "__main__":
    benchmark_model("prabayar")
    benchmark_model("pascabayar")
