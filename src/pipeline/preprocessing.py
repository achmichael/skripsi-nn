"""
Preprocessing Pipeline — Manual implementation tanpa scikit-learn.

Pipeline:
  1. Noise removal: "Tidak tahu" / "Tidak diisi" → NaN, lalu imputasi modus manual.
  2. One-Hot Encoding untuk kolom nominal.
  2b. Probability Encoding: nilai biner (0/1) diganti probabilitas kemunculan kategori.
  3. Ordinal encoding rapat (0-based, no gap): kolom frekuensi/ukuran.
  4. Feature Engineering (prabayar).
  5. Standard scaling pada fitur numerik.

Juga menyediakan: train_test_split, fit/transform MinMax & Standard Scaler untuk fitur & target.
"""

import math
import random

import numpy as np
import pandas as pd

from src.config.config import config

# =====================================================================
# MAPPING DICTIONARIES
# =====================================================================

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
    return max(counts, key=counts.get)


def fit_minmax(series: pd.Series) -> tuple[float, float]:
    x_min = float(series.min())
    x_max = float(series.max())
    return x_min, x_max

def apply_minmax(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - min_val) / (max_val - min_val)

def manual_minmax(series: pd.Series) -> pd.Series:
    x_min, x_max = fit_minmax(series)
    return apply_minmax(series, x_min, x_max)


# =====================================================================
# PROBABILITY ENCODING
# =====================================================================

def fit_probability_encoder(df: pd.DataFrame) -> dict:
    """
    Hitung probabilitas kemunculan setiap kategori dari data training
    untuk kolom binary mapping dan one-hot encoding.

    Returns:
        Dictionary berisi probabilitas per kolom per kategori.
        {
            "binary": {
                "Status_Subsidi_Listrik": {"Subsidi": 0.365, "Non Subsidi": 0.635},
                ...
            },
            "one_hot": {
                "Alat_Lain_1_Jenis": {"Setrika": 0.068, "Dispenser": 0.046, ...},
                ...
            }
        }
    """
    prob_config = config.get("probability_encoding_cols", {})
    prob_params: dict = {"binary": {}, "one_hot": {}}

    # Binary columns — hitung P(setiap nilai)
    for col in prob_config.get("binary", []):
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=True)
        total = counts.sum()
        if total == 0:
            continue
        prob_params["binary"][col] = {
            str(val): float(cnt / total) for val, cnt in counts.items()
        }

    # One-hot columns — hitung P(setiap kategori)
    for col in prob_config.get("one_hot", []):
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=True)
        total = counts.sum()
        if total == 0:
            continue
        prob_params["one_hot"][col] = {
            str(val): float(cnt / total) for val, cnt in counts.items()
        }

    return prob_params


def apply_probability_binary(
    df: pd.DataFrame,
    prob_params: dict,
) -> pd.DataFrame:
    """
    Terapkan probability encoding pada kolom binary.
    Kolom sudah di-map ke 0/1 sebelumnya.
    Ganti nilai 1 dengan P(kategori aktif), 0 dengan P(kategori tidak aktif).

    Mapping yang digunakan:
      - Status_Subsidi_Listrik: Subsidi→0, Non Subsidi→1
      - Alat_Lain_Ada: Tidak→0, Ya→1

    Setelah probability encoding:
      - Nilai 0 (kategori pertama)  → P(kategori pertama)
      - Nilai 1 (kategori kedua)    → P(kategori kedua)
    """
    binary_probs = prob_params.get("binary", {})

    # Mapping dari kolom ke {original_label: encoded_value}
    binary_label_maps = {
        "Status_Subsidi_Listrik": {"Subsidi": 0, "Non Subsidi": 1},
        "Alat_Lain_Ada": {"Tidak": 0, "Ya": 1},
    }

    for col, cat_probs in binary_probs.items():
        if col not in df.columns:
            continue

        label_map = binary_label_maps.get(col, {})
        # Bangun mapping: encoded_value → probability
        encoded_to_prob = {}
        for label, encoded_val in label_map.items():
            prob = cat_probs.get(label, 0.0)
            encoded_to_prob[float(encoded_val)] = prob

        df[col] = df[col].map(encoded_to_prob).fillna(df[col])

    return df


def apply_probability_one_hot(
    df: pd.DataFrame,
    prob_params: dict,
    ohe_fixed_categories: dict,
) -> pd.DataFrame:
    """
    Terapkan probability encoding pada kolom one-hot.
    Setelah OHE, setiap kolom `{col}_{cat}` bernilai 0 atau 1.
    Ganti nilai 1 dengan P(kategori tersebut dari data training).
    Nilai 0 tetap 0 (kategori tidak aktif).
    """
    ohe_probs = prob_params.get("one_hot", {})

    for col, cat_probs in ohe_probs.items():
        categories = ohe_fixed_categories.get(col, [])
        for cat in categories:
            col_name = f"{col}_{cat}"
            if col_name not in df.columns:
                continue
            prob = cat_probs.get(cat, 0.0)
            # Dimana nilai == 1, ganti dengan probabilitas
            df[col_name] = df[col_name].apply(
                lambda x, p=prob: p if x == 1 else 0.0
            )

    return df


# =====================================================================
# MAIN PREPROCESSING
# =====================================================================

def preprocess(df: pd.DataFrame, scaler_params: dict | None = None, prob_params: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """
    Full preprocessing pipeline untuk model prabayar.
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
    # STEP 2: One-Hot Encoding (jika ada)
    # =================================================================
    for col in config["one_hot_cols"]:
        if col not in df.columns:
            continue

        categories = config["ohe_fixed_categories"].get(col, sorted(df[col].dropna().unique()))

        for cat in categories:
            col_name = f"{col}_{cat}"
            df[col_name] = (df[col] == cat).astype(int)

        df = df.drop(columns=[col])

    # =================================================================
    # STEP 2b: Probability Encoding
    # =================================================================
    # Jika prob_params belum ada (training), fit dari data saat ini
    # Jika sudah ada (inference), gunakan yang sudah di-fit
    if prob_params is None:
        # Ini akan di-fit ulang di load_and_preprocess sebelum OHE
        # Tapi sebagai fallback, kita skip di sini
        out_prob_params = {}
    else:
        out_prob_params = prob_params
        # Apply probability encoding pada kolom binary
        df = apply_probability_binary(df, prob_params)
        # Apply probability encoding pada kolom one-hot
        df = apply_probability_one_hot(
            df, prob_params, config.get("ohe_fixed_categories", {})
        )

    # =================================================================
    # STEP 3: Ordinal Encoding Rapat (0-based, consecutive)
    # =================================================================
    for col, mapping in config["ordinal_maps"].items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # =================================================================
    # STEP 4: Feature Engineering — Prabayar
    # =================================================================

    def hitung_tarif(row):
        daya = row.get("Daya_Listrik_Rumah_VA", 900)
        raw_subsidi = row.get("Status_Subsidi_Listrik", "")
        is_subsidi = str(raw_subsidi).strip().lower() == "subsidi"

        if daya <= 450: return 415.0
        if daya == 900: return 605.0 if is_subsidi else 1352.0
        if daya in [1300, 2200]: return 1444.70
        if daya > 2200: return 1699.53
        return 1444.70

    if "Daya_Listrik_Rumah_VA" in df.columns:
        df["Tarif_PLN_Eksak_Rp"] = df.apply(hitung_tarif, axis=1)
    else:
        df["Tarif_PLN_Eksak_Rp"] = 1444.70

    # Estimasi Fisika Durasi Hari (prabayar)
    if "Nominal_Token_Terakhir_Rp" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Estimasi_kWh_Didapat"] = ((df["Nominal_Token_Terakhir_Rp"] - 2500) / 1.05) / df["Tarif_PLN_Eksak_Rp"]
        df["Estimasi_Fisika_Durasi_Hari"] = df["Estimasi_kWh_Didapat"] / (df["Total_Energi_Semua_kWhPerHari"] + 0.1)
    else:
        df["Estimasi_Fisika_Durasi_Hari"] = 0.0

    if "Frekuensi_Isi_Token_Per_Bulan" in df.columns:
        freq = df["Frekuensi_Isi_Token_Per_Bulan"].replace(0, np.nan)
        df["Durasi_Dari_Frekuensi"] = (30.0 / freq).clip(1, 60)
        df["Durasi_Dari_Frekuensi"] = df["Durasi_Dari_Frekuensi"].fillna(30.0)

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
            if x <= 20_000:   return 0
            if x <= 50_000:   return 1
            if x <= 100_000:  return 2
            if x <= 200_000:  return 3
            return 4
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

    # Estimasi biaya bulanan (fallback)
    if "Estimasi_Biaya_Energi_Bulanan_Rp" not in df.columns:
        if "Estimasi_Tarif_Per_kWh_Rp" in df.columns and "Total_Energi_Semua_kWhPerBulan" in df.columns:
            df["Estimasi_Biaya_Energi_Bulanan_Rp"] = (
                df["Estimasi_Tarif_Per_kWh_Rp"] * df["Total_Energi_Semua_kWhPerBulan"]
            )

    # Daya × total energi harian
    if "Daya_Listrik_Rumah_VA" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Daya_x_TotalEnergi"] = (
            df["Daya_Listrik_Rumah_VA"] * df["Total_Energi_Semua_kWhPerHari"]
        )

    out_scaler_params = {} if scaler_params is None else scaler_params.copy()

    print('dataframe', df['Estimasi_Tarif_Per_kWh_Rp'] if 'Estimasi_Tarif_Per_kWh_Rp' in df.columns else 'N/A')
    return df, out_scaler_params, out_prob_params


# =====================================================================
# LOAD CSV + PREPROCESS
# =====================================================================

def load_and_preprocess(path: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Baca CSV, lalu jalankan full preprocessing pipeline.
    Returns: (df, minmax_scaler_params, prob_params)
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Fit probability encoder SEBELUM binary mapping & OHE
    # karena butuh nilai kategorikal asli untuk hitung probabilitas
    prob_params = fit_probability_encoder(df)

    print("\n=== PROBABILITY ENCODING PARAMS ===")
    for enc_type, cols in prob_params.items():
        for col, probs in cols.items():
            print(f"  [{enc_type}] {col}:")
            for cat, p in probs.items():
                print(f"    {cat}: {p:.6f}")
    print("=" * 50 + "\n")

    # Handle special numeric values
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

    df, minmax_scaler_params, prob_params = preprocess(df, prob_params=prob_params)

    return df, minmax_scaler_params, prob_params


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
# FIT / TRANSFORM MIN-MAX SCALER
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
# FIT / TRANSFORM STANDARD SCALER
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
            std = 1.0
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
    """Transform target values using fitted scaler."""
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
    """Inverse transform: normalized [0,1] → original scale."""
    raw = value * (scaler["max"] - scaler["min"]) + scaler["min"]

    if scaler.get("use_log", False):
        raw = min(raw, 709.0)
        final_val = float(np.expm1(raw))
        return final_val

    return raw
