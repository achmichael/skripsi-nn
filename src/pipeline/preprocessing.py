"""
Preprocessing Pipeline — Manual implementation tanpa scikit-learn.

Pipeline:
  1. Noise removal: "Tidak tahu" / "Tidak diisi" → NaN, lalu imputasi modus manual.
  2. Cyclical encoding: Bulan_Tagihan → sin/cos (periode 12).
  3. One-Hot Encoding manual: Sumber_Angka_Tagihan, Tagihan_Relatif_Stabil.
  4. Ordinal encoding rapat (0-based, no gap): kolom frekuensi/ukuran.
  5. Min-Max scaling manual [0,1] pada kolom ordinal.

Juga menyediakan: train_test_split, fit/transform MinMax untuk fitur & target.
"""

import math
import random

import numpy as np
import pandas as pd

from src.config.config import config


# =====================================================================
# MAPPING DICTIONARIES
# =====================================================================

# Mappings are now loaded from src/config/config.py

# Kolom ordinal yang akan di-MinMax scale (Step 5)
ORDINAL_COLS_TO_SCALE = list(config["ordinal_maps"].keys())


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def manual_mode(series: pd.Series):
    """
    Hitung modus secara manual (kategori paling sering muncul).
    Mengabaikan NaN. Jika tie, ambil yang pertama ditemukan.
    """
    counts: dict = {}
    for val in series:
        if pd.isna(val):
            continue
        counts[val] = counts.get(val, 0) + 1
    if not counts:
        return np.nan
    # Return key with max count
    return max(counts, key=counts.get)


def fit_minmax(series: pd.Series) -> tuple[float, float]:
    """
    Computes min and max for scaling from training data.
    """
    x_min = float(series.min())
    x_max = float(series.max())
    return x_min, x_max

def apply_minmax(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """
    Applies min-max scaling using pre-computed parameters.
    """
    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - min_val) / (max_val - min_val)

def manual_minmax(series: pd.Series) -> pd.Series:
    """
    Min-Max scaling manual: X_scaled = (X - X_min) / (X_max - X_min).
    Jika X_max == X_min → return 0.0 (constant column).
    """
    x_min, x_max = fit_minmax(series)
    return apply_minmax(series, x_min, x_max)


# =====================================================================
# MAIN PREPROCESSING
# =====================================================================

def preprocess(df: pd.DataFrame, scaler_params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline. Menerima DataFrame mentah dari CSV,
    mengembalikan DataFrame bersih siap untuk feature extraction.

    Contoh penggunaan:
      # Training:
      df_train, train_scaler = preprocess(df_train, scaler_params=None)
      
      # Inference/Test:
      df_test, _ = preprocess(df_test, scaler_params=train_scaler)

    Steps:
      1. Missing value: "Tidak tahu"/"Tidak diisi" → NaN → imputasi modus.
      2. Cyclical encoding Bulan_Tagihan → sin/cos.
      3. One-Hot Encoding Sumber_Angka_Tagihan & Tagihan_Relatif_Stabil.
      4. Ordinal encoding rapat (0-based, no gap).
      5. Min-Max scaling pada kolom ordinal.
      6. Feature Engineering.
      7. Min-Max scaling pada numeric features dengan scaler parameter.
    """
    df = df.copy()

    # =================================================================
    # STEP 1: Noise Removal & Manual Mode Imputation
    # =================================================================

    # 1a. "Tidak tahu" → NaN
    for col in config["tidak_tahu_cols"]:
        if col in df.columns:
            df[col] = df[col].replace("Tidak tahu", np.nan)

    # 1b. "Tidak diisi" → NaN
    for col in config["tidak_diisi_cols"]:
        if col in df.columns:
            df[col] = df[col].replace("Tidak diisi", np.nan)

    # 1c. Imputasi NaN dengan modus manual
    all_impute_cols = config["tidak_tahu_cols"] + config["tidak_diisi_cols"]
    for col in all_impute_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = manual_mode(df[col])
            df[col] = df[col].fillna(mode_val)

    # =================================================================
    # STEP 2: Cyclical Encoding — Bulan_Tagihan
    # =================================================================
    if "Bulan_Tagihan" in df.columns:
        # Map nama bulan → angka 1-12
        bulan_num = df["Bulan_Tagihan"].map(config["bulan_to_num"])

        # Cyclical transform: sin & cos dengan periode 12
        df["Bulan_Tagihan_sin"] = np.sin(2 * np.pi * bulan_num / 12)
        df["Bulan_Tagihan_cos"] = np.cos(2 * np.pi * bulan_num / 12)

        # Hapus kolom asli
        df = df.drop(columns=["Bulan_Tagihan"])

    # =================================================================
    # STEP 3: Manual One-Hot Encoding (Nominal)
    # =================================================================
    for col in config["one_hot_cols"]:
        if col not in df.columns:
            continue

        # BUG 1 Fix: Use hardcoded deterministic categories
        categories = config["ohe_fixed_categories"].get(col, sorted(df[col].dropna().unique()))

        # Buat binary columns manual
        for cat in categories:
            col_name = f"{col}__{cat}"
            df[col_name] = (df[col] == cat).astype(int)

        # Hapus kolom asli
        df = df.drop(columns=[col])

    # =================================================================
    # STEP 4: Ordinal Encoding Rapat (0-based, consecutive)
    # =================================================================
    for col, mapping in config["ordinal_maps"].items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # =================================================================
    # STEP 5: Manual Min-Max Scaling pada Kolom Ordinal
    # =================================================================
    for col in ORDINAL_COLS_TO_SCALE:
        if col in df.columns:
            df[col] = manual_minmax(df[col].astype(float))

    # =================================================================
    # STEP 6: Feature Engineering — Physical Approximations & Interactions
    # =================================================================
    
    # Fungsi pembantu untuk menghitung tarif eksak PLN
    def hitung_tarif(row):
        daya = row.get("Daya_Listrik_Rumah_VA", 900)
        # BUG 3 Fix: Compare raw string instead of float mapping
        raw_subsidi = row.get("Status_Subsidi_Listrik", "")
        is_subsidi = str(raw_subsidi).strip().lower() == "subsidi"
        
        if daya <= 450: return 415.0
        if daya == 900: return 605.0 if is_subsidi else 1352.0
        if daya in [1300, 2200]: return 1444.70
        if daya > 2200: return 1699.53
        return 1444.70 # Default

    if "Daya_Listrik_Rumah_VA" in df.columns and "Status_Subsidi_Listrik" in df.columns:
        df["Tarif_PLN_Eksak_Rp"] = df.apply(hitung_tarif, axis=1)
    else:
        # Fallback if columns are somehow missing
        df["Tarif_PLN_Eksak_Rp"] = df.get("Estimasi_Tarif_Per_kWh_Rp", 1444.70)

    # 1. PRABAYAR: Estimasi Fisika Durasi Hari
    #    (Nominal - Admin) / 1.05 PPJ / Tarif / Total kWh harian
    if "Nominal_Token_Terakhir_Rp" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Estimasi_kWh_Didapat"] = ((df["Nominal_Token_Terakhir_Rp"] - 2500) / 1.05) / df["Tarif_PLN_Eksak_Rp"]
        df["Estimasi_Fisika_Durasi_Hari"] = df["Estimasi_kWh_Didapat"] / (df["Total_Energi_Semua_kWhPerHari"] + 0.1)
    else:
        df["Estimasi_Fisika_Durasi_Hari"] = 0.0

    # If someone refills N times per month, each token lasts ~30/N days
    # Clip to [1, 60] to avoid division by zero and unrealistic outliers
    if "Frekuensi_Isi_Token_Per_Bulan" in df.columns:
        freq = df["Frekuensi_Isi_Token_Per_Bulan"].replace(0, np.nan)
        df["Durasi_Dari_Frekuensi"] = (30.0 / freq).clip(1, 60)
        df["Durasi_Dari_Frekuensi"] = df["Durasi_Dari_Frekuensi"].fillna(30.0)

    # Nominal token Rp → kWh → days at current consumption rate
    # Using survey-based tariff estimate (may differ from physics tariff)
    if ("Nominal_Token_Terakhir_Rp" in df.columns and 
        "Estimasi_Tarif_Per_kWh_Rp" in df.columns and
        "Total_Energi_Semua_kWhPerHari" in df.columns):
        kwh_beli = df["Nominal_Token_Terakhir_Rp"] / (
            df["Estimasi_Tarif_Per_kWh_Rp"].replace(0, 1444.70)
        )
        df["Rasio_Token_vs_Energi"] = kwh_beli / (
            df["Total_Energi_Semua_kWhPerHari"] + 0.1
        )

    if "Nominal_Token_Terakhir_Rp" in df.columns:
        def kategorikan_nominal(x):
            if x <= 20_000:   return 0  # very short expected duration
            if x <= 50_000:   return 1  # short
            if x <= 100_000:  return 2  # medium
            if x <= 200_000:  return 3  # long
            return 4                     # very long
        df["Token_Nominal_Kategori"] = df["Nominal_Token_Terakhir_Rp"].apply(
            kategorikan_nominal
        )

    if ("Total_Energi_Semua_kWhPerHari" in df.columns and 
        "Nominal_Token_Terakhir_Rp" in df.columns):
        df["Energi_Per_Nominal"] = (
            df["Total_Energi_Semua_kWhPerHari"] /
            (df["Nominal_Token_Terakhir_Rp"] / 1000.0 + 0.01)
        )

    if ("Estimasi_Fisika_Durasi_Hari" in df.columns and 
        "Durasi_Dari_Frekuensi" in df.columns):
        df["Fisika_vs_Frekuensi_Gap"] = (
            df["Estimasi_Fisika_Durasi_Hari"] - df["Durasi_Dari_Frekuensi"]
        )
        df["Rasio_Fisika_vs_Frekuensi"] = df["Estimasi_Fisika_Durasi_Hari"] / (df["Durasi_Dari_Frekuensi"] + 1e-8)
        df["Rasio_Fisika_vs_Frekuensi"] = df["Rasio_Fisika_vs_Frekuensi"].clip(0.1, 10.0)

    # 2. PASCABAYAR: Estimasi Fisika Tagihan Rp
    #    Total kWh bulanan * Tarif * 1.05 PPJ + 2500 Admin
    if "Total_Energi_Semua_kWhPerBulan" in df.columns:
        df["Estimasi_Fisika_Tagihan_Rp"] = (df["Total_Energi_Semua_kWhPerBulan"] * df["Tarif_PLN_Eksak_Rp"] * 1.05) + 2500
    else:
        df["Estimasi_Fisika_Tagihan_Rp"] = 0.0

    # Estimasi biaya bulanan = tarif × kWh/bulan (sudah ada di CSV)
    # Jika tidak ada, hitung manual (fallback lama)
    if "Estimasi_Biaya_Energi_Bulanan_Rp" not in df.columns:
        if "Estimasi_Tarif_Per_kWh_Rp" in df.columns and "Total_Energi_Semua_kWhPerBulan" in df.columns:
            df["Estimasi_Biaya_Energi_Bulanan_Rp"] = (
                df["Estimasi_Tarif_Per_kWh_Rp"] * df["Total_Energi_Semua_kWhPerBulan"]
            )

    # Daya × total energi harian — proxy kapasitas pemakaian
    if "Daya_Listrik_Rumah_VA" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Daya_x_TotalEnergi"] = (
            df["Daya_Listrik_Rumah_VA"] * df["Total_Energi_Semua_kWhPerHari"]
        )

    # =================================================================
    # STEP 7: Numeric Feature Scaling
    # =================================================================
    numeric_cols_to_scale = list(dict.fromkeys(
        config.get("numeric_cols", []) + [
            "Estimasi_Fisika_Tagihan_Rp", "Tarif_PLN_Eksak_Rp", "Daya_x_TotalEnergi",
            "Estimasi_kWh_Didapat", "Estimasi_Fisika_Durasi_Hari",
            "Total_Energi_Alat_Lain_kWhPerHari",
        ]
    ))
    
    out_scaler_params = {} if scaler_params is None else scaler_params.copy()
    
    for col in numeric_cols_to_scale:
        if col in df.columns:
            if scaler_params is None:
                min_val, max_val = fit_minmax(df[col].astype(float))
                out_scaler_params[col] = {"min": min_val, "max": max_val}
            else:
                if col in scaler_params:
                    min_val = scaler_params[col]["min"]
                    max_val = scaler_params[col]["max"]
                else:
                    min_val, max_val = 0.0, 1.0
                    
            df[col] = apply_minmax(df[col].astype(float), min_val, max_val)

    return df, out_scaler_params


# =====================================================================
# LOAD CSV + PREPROCESS (pengganti load_csv lama)
# =====================================================================

def load_and_preprocess(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Baca CSV, lalu jalankan full preprocessing pipeline.
    Menangani juga kolom numerik khusus (Daya_Listrik_Rumah_VA).

    Returns:
        (df_preprocessed, minmax_scaler_params)
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Handle special numeric values sebelum preprocessing
    if "Daya_Listrik_Rumah_VA" in df.columns:
        df["Daya_Listrik_Rumah_VA"] = df["Daya_Listrik_Rumah_VA"].replace({
            "Tidak tahu": 900,
            "> 5500": 7700,
        })
        df["Daya_Listrik_Rumah_VA"] = pd.to_numeric(
            df["Daya_Listrik_Rumah_VA"], errors="coerce"
        )

    # Binary encode Status_Subsidi_Listrik
    if "Status_Subsidi_Listrik" in df.columns:
        df["Status_Subsidi_Listrik"] = df["Status_Subsidi_Listrik"].map({
            "Subsidi": 0,
            "Non Subsidi": 1,
        }).astype(float)

    # Binary encode Alat_Lain_Ada
    if "Alat_Lain_Ada" in df.columns:
        df["Alat_Lain_Ada"] = df["Alat_Lain_Ada"].map({
            "Tidak": 0,
            "Ya": 1,
        }).astype(float)

    df, minmax_scaler_params = preprocess(df)

    return df, minmax_scaler_params


# =====================================================================
# TRAIN/TEST SPLIT
# =====================================================================

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


# =====================================================================
# FIT / TRANSFORM MIN-MAX SCALER (untuk fitur numerik global)
# =====================================================================

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


# =====================================================================
# FIT / TRANSFORM STANDARD SCALER (untuk fitur numerik global)
# =====================================================================

def fit_standard_scaler(x_data: list[list[float]]) -> dict:
    """Fit Standard Scaler (Z-score normalization)."""
    total_features = len(x_data[0])
    means = []
    stds = []

    for feature_index in range(total_features):
        column_values = [row[feature_index] for row in x_data]
        mean = float(np.mean(column_values))
        std = float(np.std(column_values))
        if std == 0.0:
            std = 1.0  # Prevent division by zero
        means.append(mean)
        stds.append(std)

    return {
        "mean": means,
        "std": stds,
    }


def transform_standard_scaler(x_data: list[list[float]], scaler: dict) -> list[list[float]]:
    """Transform data menggunakan StandardScaler (Z-score)."""
    scaled_data = []

    for row in x_data:
        scaled_row = []
        for i, value in enumerate(row):
            mean_value = scaler["mean"][i]
            std_value = scaler["std"][i]
            scaled_value = (value - mean_value) / std_value
            scaled_row.append(scaled_value)
        scaled_data.append(scaled_row)

    return scaled_data



# =====================================================================
# FIT / TRANSFORM TARGET SCALER
# =====================================================================

def fit_target_scaler(y_data: list[float], use_log: bool = False) -> dict:
    """Fit MinMax scaler on target values, optionally with log-transform."""
    if use_log:
        y_transformed = [float(np.log1p(y)) for y in y_data]
    else:
        y_transformed = y_data

    return {
        "min": min(y_transformed),
        "max": max(y_transformed),
        "use_log": use_log,
    }


def transform_target(y_data: list[float], scaler: dict) -> list[float]:
    """Transform target values using fitted scaler (with optional log-transform)."""
    use_log = scaler.get("use_log", False)
    min_value = scaler["min"]
    max_value = scaler["max"]

    scaled = []

    for value in y_data:
        v = float(np.log1p(value)) if use_log else value

        if max_value == min_value:
            scaled.append(0.0)
        else:
            scaled.append((v - min_value) / (max_value - min_value))

    return scaled


def inverse_transform_target(value: float, scaler: dict) -> float:
    """Inverse transform: normalized [0,1] → original scale (with optional exp)."""
    raw = value * (scaler["max"] - scaler["min"]) + scaler["min"]
    # print("value", value)
    # print("scaler max", scaler["max"])
    # print("scaler min", scaler["min"])
    # print("raw", raw)
    if scaler.get("use_log", False):
        raw = min(raw, 709.0)
        return float(np.expm1(raw))

    return raw
