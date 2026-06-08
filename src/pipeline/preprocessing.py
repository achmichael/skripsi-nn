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


# =====================================================================
# MAPPING DICTIONARIES
# =====================================================================

# --- Kolom yang "Tidak tahu" → NaN lalu imputasi modus ---
TIDAK_TAHU_COLS = [
    "Kulkas_Kategori",
    "AC_PK_Kategori",
    "Bulan_Tagihan",
    "Tagihan_Relatif_Stabil",
]

# --- Kolom yang "Tidak diisi" → NaN lalu imputasi modus ---
TIDAK_DIISI_COLS = [
    "Alat_Lain_1_Kategori",
    "Alat_Lain_2_Kategori",
    "Alat_Lain_3_Kategori",
]

# --- Step 2: Bulan_Tagihan label → angka (sebelum sin/cos) ---
BULAN_TO_NUM = {
    "Januari": 1,  "Februari": 2,  "Maret": 3,
    "April": 4,     "Mei": 5,       "Juni": 6,
    "Juli": 7,      "Agustus": 8,   "September": 9,
    "Oktober": 10,  "November": 11, "Desember": 12,
}

# --- Step 3: One-Hot Encoding (nominal, tanpa urutan) ---
ONE_HOT_COLS = [
    "Sumber_Angka_Tagihan",
    "Tagihan_Relatif_Stabil",
]

# --- Step 4: Ordinal encoding rapat (0-based, consecutive) ---
# "Tidak tahu" dan "Tidak diisi" sudah di-imputasi, jadi tidak masuk mapping.
ORDINAL_MAPS = {
    "Kulkas_Kategori": {
        "Tidak ada": 0,
        "Kecil / 1 pintu": 1,
        "Sedang / 2 pintu": 2,
        "Besar / side by side": 3,
    },
    "TV_Kategori": {
        "Tidak ada / tidak digunakan": 0,
        "Jarang, kurang dari 2 jam per hari": 1,
        "Sedang, sekitar 2-5 jam per hari": 2,
        "Sering, sekitar 6-10 jam per hari": 3,
        "Sangat sering, lebih dari 10 jam per hari": 4,
    },
    "AC_Kategori": {
        "Tidak ada / tidak digunakan": 0,
        "Jarang, kurang dari 2 jam per hari": 1,
        "Sedang, sekitar 2-5 jam per hari": 2,
        "Sering, sekitar 6-10 jam per hari": 3,
        "Sangat sering, lebih dari 10 jam per hari": 4,
    },
    "Kipas_Kategori": {
        "Tidak ada / tidak digunakan": 0,
        "Jarang, kurang dari 2 jam per hari": 1,
        "Sedang, sekitar 2-5 jam per hari": 2,
        "Sering, sekitar 6-10 jam per hari": 3,
        "Sangat sering, lebih dari 10 jam per hari": 4,
    },
    "RiceCooker_Kategori": {
        "Tidak ada / tidak digunakan": 0,
        "Jarang, kurang dari 2 jam per hari": 1,
        "Sedang, sekitar 2-5 jam per hari": 2,
        "Sering, sekitar 6-10 jam per hari": 3,
        "Sangat sering, lebih dari 10 jam per hari": 4,
    },
    "MesinCuci_Kategori": {
        "Tidak ada / tidak digunakan": 0,
        "Jarang, 1-2 kali per minggu": 1,
        "Sedang, 3-4 kali per minggu": 2,
        "Sering, 5-6 kali per minggu": 3,
        "Sangat sering, hampir setiap hari": 4,
    },
    "AC_PK_Kategori": {
        "Tidak ada AC": 0,
        "1/2 PK": 1,
        "3/4 PK": 2,
        "1 PK": 3,
        "1.5 PK": 4,
        "2 PK atau lebih": 5,
    },
    "Alat_Lain_1_Kategori": {
        "Jarang, 1-2 kali per minggu": 0,
        "Sedang, 3-4 kali per minggu": 1,
        "Sering, hampir setiap hari": 2,
        "Sangat sering, setiap hari dan cukup lama": 3,
    },
    "Alat_Lain_2_Kategori": {
        "Jarang, 1-2 kali per minggu": 0,
        "Sedang, 3-4 kali per minggu": 1,
        "Sering, hampir setiap hari": 2,
        "Sangat sering, setiap hari dan cukup lama": 3,
    },
    "Alat_Lain_3_Kategori": {
        "Jarang, 1-2 kali per minggu": 0,
        "Sedang, 3-4 kali per minggu": 1,
        "Sering, hampir setiap hari": 2,
        "Sangat sering, setiap hari dan cukup lama": 3,
    },
}

# Kolom ordinal yang akan di-MinMax scale (Step 5)
ORDINAL_COLS_TO_SCALE = list(ORDINAL_MAPS.keys())


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


def manual_minmax(series: pd.Series) -> pd.Series:
    """
    Min-Max scaling manual: X_scaled = (X - X_min) / (X_max - X_min).
    Jika X_max == X_min → return 0.0 (constant column).
    """
    x_min = series.min()
    x_max = series.max()
    if x_max == x_min:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - x_min) / (x_max - x_min)


# =====================================================================
# MAIN PREPROCESSING
# =====================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Menerima DataFrame mentah dari CSV,
    mengembalikan DataFrame bersih siap untuk feature extraction.

    Steps:
      1. Missing value: "Tidak tahu"/"Tidak diisi" → NaN → imputasi modus.
      2. Cyclical encoding Bulan_Tagihan → sin/cos.
      3. One-Hot Encoding Sumber_Angka_Tagihan & Tagihan_Relatif_Stabil.
      4. Ordinal encoding rapat (0-based, no gap).
      5. Min-Max scaling pada kolom ordinal.
    """
    df = df.copy()

    # =================================================================
    # STEP 1: Noise Removal & Manual Mode Imputation
    # =================================================================

    # 1a. "Tidak tahu" → NaN
    for col in TIDAK_TAHU_COLS:
        if col in df.columns:
            df[col] = df[col].replace("Tidak tahu", np.nan)

    # 1b. "Tidak diisi" → NaN
    for col in TIDAK_DIISI_COLS:
        if col in df.columns:
            df[col] = df[col].replace("Tidak diisi", np.nan)

    # 1c. Imputasi NaN dengan modus manual
    all_impute_cols = TIDAK_TAHU_COLS + TIDAK_DIISI_COLS
    for col in all_impute_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = manual_mode(df[col])
            df[col] = df[col].fillna(mode_val)

    # =================================================================
    # STEP 2: Cyclical Encoding — Bulan_Tagihan
    # =================================================================
    if "Bulan_Tagihan" in df.columns:
        # Map nama bulan → angka 1-12
        bulan_num = df["Bulan_Tagihan"].map(BULAN_TO_NUM)

        # Cyclical transform: sin & cos dengan periode 12
        df["Bulan_Tagihan_sin"] = np.sin(2 * np.pi * bulan_num / 12)
        df["Bulan_Tagihan_cos"] = np.cos(2 * np.pi * bulan_num / 12)

        # Hapus kolom asli
        df = df.drop(columns=["Bulan_Tagihan"])

    # =================================================================
    # STEP 3: Manual One-Hot Encoding (Nominal)
    # =================================================================
    for col in ONE_HOT_COLS:
        if col not in df.columns:
            continue

        # Dapatkan unique categories (sorted agar konsisten)
        categories = sorted(df[col].dropna().unique())

        # Buat binary columns manual
        for cat in categories:
            col_name = f"{col}__{cat}"
            df[col_name] = (df[col] == cat).astype(int)

        # Hapus kolom asli
        df = df.drop(columns=[col])

    # =================================================================
    # STEP 4: Ordinal Encoding Rapat (0-based, consecutive)
    # =================================================================
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # =================================================================
    # STEP 5: Manual Min-Max Scaling pada Kolom Ordinal
    # =================================================================
    for col in ORDINAL_COLS_TO_SCALE:
        if col in df.columns:
            df[col] = manual_minmax(df[col].astype(float))

    # =================================================================
    # STEP 6: Feature Engineering — Interaction Features
    # =================================================================
    # Estimasi biaya bulanan = tarif × kWh/bulan (sudah ada di CSV)
    # Jika tidak ada, hitung manual
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

    return df


# =====================================================================
# LOAD CSV + PREPROCESS (pengganti load_csv lama)
# =====================================================================

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Baca CSV, lalu jalankan full preprocessing pipeline.
    Menangani juga kolom numerik khusus (Daya_Listrik_Rumah_VA).
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

    # Run main preprocessing
    df = preprocess(df)

    return df


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
# FIT / TRANSFORM TARGET SCALER
# =====================================================================

def fit_target_scaler(y_data: list[float], use_log: bool = False) -> dict:
    """Fit MinMax scaler on target values, optionally with log-transform."""
    if use_log:
        y_transformed = [math.log(y) for y in y_data]
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
        v = math.log(value) if use_log else value

        if max_value == min_value:
            scaled.append(0.0)
        else:
            scaled.append((v - min_value) / (max_value - min_value))

    return scaled


def inverse_transform_target(value: float, scaler: dict) -> float:
    """Inverse transform: normalized [0,1] → original scale (with optional exp)."""
    raw = value * (scaler["max"] - scaler["min"]) + scaler["min"]

    if scaler.get("use_log", False):
        return math.exp(raw)

    return raw
