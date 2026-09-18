#!/usr/bin/env python3
"""
Script untuk validasi kelengkapan dan konsistensi fitur di config.py

Usage:
    python scripts/validate_config_features.py

Output:
    - Laporan validasi ke console
    - File report di docs/config_validation_report.txt
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from config.config import config


class ConfigValidator:
    def __init__(self):
        self.dataset_path = project_root / "data" / "prabayar.csv"
        self.df = pd.read_csv(self.dataset_path)
        self.config = config

        # Kolom yang bukan fitur
        self.excluded_cols = {
            'Timestamp', 'Nama/Inisial', 'Kota/Kabupaten',
            'Jenis_Listrik', 'Token_Habis_Dalam_Hari'
        }

        self.errors = []
        self.warnings = []
        self.info = []

    def get_dataset_features(self):
        """Get all feature columns from dataset"""
        all_cols = set(self.df.columns)
        return all_cols - self.excluded_cols

    def validate_features_list(self):
        """Validate features['prabayar'] completeness"""
        print("\n" + "="*80)
        print("1. VALIDASI features['prabayar']")
        print("="*80)

        dataset_features = self.get_dataset_features()
        config_features = set(self.config['features']['prabayar'])
        engineered_features = {
            'Daya_x_TotalEnergi', 'Durasi_Dari_Frekuensi',
            'Energi_Per_Nominal', 'Estimasi_Fisika_Durasi_Hari',
            'Estimasi_kWh_Didapat', 'Fisika_vs_Frekuensi_Gap',
            'Rasio_Fisika_vs_Frekuensi', 'Rasio_Token_vs_Energi',
            'Tarif_PLN_Eksak_Rp', 'Token_Nominal_Kategori'
        }

        # Features from dataset only (not engineered)
        config_from_dataset = config_features - engineered_features

        missing = dataset_features - config_features
        extra = config_from_dataset - dataset_features

        if missing:
            self.errors.append(f"features['prabayar']: {len(missing)} fitur missing dari dataset")
            print(f"\n❌ MISSING ({len(missing)} fitur):")
            for f in sorted(missing):
                print(f"   - {f}")
        else:
            print("\n✅ Semua fitur dari dataset sudah terdaftar")

        if extra:
            self.warnings.append(f"features['prabayar']: {len(extra)} fitur tidak ada di dataset")
            print(f"\n⚠️  EXTRA ({len(extra)} fitur - seharusnya engineered):")
            for f in sorted(extra):
                print(f"   - {f}")

        print(f"\n📊 Total features['prabayar']: {len(config_features)}")
        print(f"   - Dari dataset: {len(config_from_dataset)}")
        print(f"   - Engineered: {len(engineered_features)}")

    def validate_numeric_cols(self):
        """Validate numeric_cols completeness"""
        print("\n" + "="*80)
        print("2. VALIDASI numeric_cols")
        print("="*80)

        config_numeric = set(self.config['numeric_cols'])

        # Find all numeric columns in dataset
        dataset_numeric = set()
        for col in self.df.columns:
            if col in self.excluded_cols:
                continue
            if self.df[col].dtype in ['int64', 'float64']:
                dataset_numeric.add(col)

        # Also check features that are in features list
        features_list = set(self.config['features']['prabayar'])

        missing_numeric = []
        for col in dataset_numeric:
            if col not in config_numeric and col in features_list:
                # Check if it's ordinal or one-hot instead
                is_ordinal = col in self.config['ordinal_encoding']
                is_onehot = col in self.config['one_hot_nominal']
                if not is_ordinal and not is_onehot:
                    missing_numeric.append(col)

        if missing_numeric:
            self.errors.append(f"numeric_cols: {len(missing_numeric)} fitur numeric missing")
            print(f"\n❌ MISSING NUMERIC ({len(missing_numeric)} fitur):")
            for f in sorted(missing_numeric):
                print(f"   - {f} (dtype: {self.df[f].dtype})")
        else:
            print("\n✅ Semua fitur numeric dari features sudah terdaftar")

        # Check for features in features list but not in numeric_cols
        in_features_not_numeric = []
        for col in features_list:
            if col not in self.excluded_cols and col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    if col not in config_numeric:
                        is_ordinal = col in self.config['ordinal_encoding']
                        if not is_ordinal:
                            in_features_not_numeric.append(col)

        if in_features_not_numeric:
            self.warnings.append(f"numeric_cols: {len(in_features_not_numeric)} fitur di features tapi belum di numeric_cols")
            print(f"\n⚠️  DI FEATURES TAPI BELUM DI NUMERIC_COLS ({len(in_features_not_numeric)}):")
            for f in sorted(in_features_not_numeric):
                print(f"   - {f}")

        print(f"\n📊 Total numeric_cols: {len(config_numeric)}")

    def validate_ordinal_encoding(self):
        """Validate ordinal_encoding completeness"""
        print("\n" + "="*80)
        print("3. VALIDASI ordinal_encoding")
        print("="*80)

        ordinal_cols = set(self.config['ordinal_encoding'].keys())

        # Find categorical columns in dataset
        dataset_categorical = set()
        for col in self.df.columns:
            if col in self.excluded_cols:
                continue
            if self.df[col].dtype == 'object' or str(self.df[col].dtype) == 'string':
                dataset_categorical.add(col)

        # Check for obvious ordinal columns missing
        known_ordinal = {
            'AC_PK_Kategori',  # Should be ordinal
            'Status_Subsidi_Listrik',  # Could be ordinal
        }

        missing_ordinal = []
        for col in known_ordinal:
            if col in self.df.columns and col not in ordinal_cols:
                missing_ordinal.append(col)

        if missing_ordinal:
            self.errors.append(f"ordinal_encoding: {len(missing_ordinal)} fitur ordinal missing")
            print(f"\n❌ MISSING ORDINAL ({len(missing_ordinal)} fitur):")
            for f in sorted(missing_ordinal):
                print(f"   - {f}")
                print(f"     Values: {self.df[f].unique()[:5]}")
        else:
            print("\n✅ Semua fitur ordinal yang diketahui sudah terdaftar")

        print(f"\n📊 Total ordinal_encoding: {len(ordinal_cols)}")

    def validate_ordinal_consistency(self):
        """Validate consistency between ordinal_encoding and ordinal_maps"""
        print("\n" + "="*80)
        print("4. VALIDASI KONSISTENSI ordinal_encoding vs ordinal_maps")
        print("="*80)

        ordinal_enc = set(self.config['ordinal_encoding'].keys())
        ordinal_maps = set(self.config['ordinal_maps'].keys())

        # Check for differences
        in_enc_not_maps = ordinal_enc - ordinal_maps
        in_maps_not_enc = ordinal_maps - ordinal_enc

        if in_enc_not_maps:
            self.warnings.append(f"Inkonsistensi: {len(in_enc_not_maps)} fitur di ordinal_encoding tapi tidak di ordinal_maps")
            print(f"\n⚠️  DI ordinal_encoding TAPI TIDAK DI ordinal_maps:")
            for f in sorted(in_enc_not_maps):
                print(f"   - {f}")

        if in_maps_not_enc:
            self.warnings.append(f"Inkonsistensi: {len(in_maps_not_enc)} fitur di ordinal_maps tapi tidak di ordinal_encoding")
            print(f"\n⚠️  DI ordinal_maps TAPI TIDAK DI ordinal_encoding:")
            for f in sorted(in_maps_not_enc):
                print(f"   - {f}")

        # Check for Alat_Lain_X_Jenis in ordinal_maps (should be in one_hot only)
        alat_lain_jenis = [f for f in ordinal_maps if '_Jenis' in f and 'Alat_Lain' in f]
        if alat_lain_jenis:
            self.errors.append(f"ordinal_maps: {len(alat_lain_jenis)} fitur Jenis seharusnya one-hot, bukan ordinal")
            print(f"\n❌ SALAH KATEGORI (seharusnya one-hot, bukan ordinal):")
            for f in sorted(alat_lain_jenis):
                print(f"   - {f}")

        # Check for value consistency
        print(f"\n📊 Checking value consistency...")
        inconsistent = []
        for col in ordinal_enc & ordinal_maps:
            enc_values = set(self.config['ordinal_encoding'][col].keys())
            map_values = set(self.config['ordinal_maps'][col].keys())
            if enc_values != map_values:
                inconsistent.append(col)

        if inconsistent:
            self.warnings.append(f"Inkonsistensi nilai: {len(inconsistent)} fitur memiliki mapping berbeda")
            print(f"\n⚠️  MAPPING BERBEDA ({len(inconsistent)} fitur):")
            for f in sorted(inconsistent):
                enc_vals = set(self.config['ordinal_encoding'][f].keys())
                map_vals = set(self.config['ordinal_maps'][f].keys())
                print(f"   - {f}")
                print(f"     ordinal_encoding: {enc_vals - map_vals}")
                print(f"     ordinal_maps: {map_vals - enc_vals}")
        else:
            print("✅ Semua mapping konsisten")

    def validate_pattern_consistency(self):
        """Validate pattern consistency across appliance categories"""
        print("\n" + "="*80)
        print("5. VALIDASI KONSISTENSI POLA FITUR")
        print("="*80)

        # Define expected pattern for main appliances
        main_appliances = ['Kulkas', 'TV', 'AC', 'Kipas', 'RiceCooker', 'MesinCuci']
        pattern_main = ['Jumlah', 'Kategori', 'EstimasiWattPerUnit',
                        'EstimasiJamPerHari', 'Energi_kWhPerHari']

        # Define expected pattern for Alat_Lain
        alat_lain = ['Alat_Lain_1', 'Alat_Lain_2', 'Alat_Lain_3']
        pattern_alat = ['Jenis', 'Jumlah', 'Kategori', 'EstimasiWatt',
                       'EstimasiJamPerHari', 'Energi_kWhPerHari']

        print("\n📊 Checking main appliances pattern...")
        config_features = set(self.config['features']['prabayar'])
        config_numeric = set(self.config['numeric_cols'])

        for appliance in main_appliances:
            missing = []
            for attr in pattern_main:
                col = f"{appliance}_{attr}"
                if col in self.df.columns and col not in config_features:
                    missing.append(col)
            if missing:
                print(f"   ⚠️  {appliance}: {len(missing)} fitur missing")

        print("\n📊 Checking Alat_Lain pattern...")
        alat_lain_issues = []
        for alat in alat_lain:
            missing_features = []
            missing_numeric = []
            for attr in pattern_alat:
                col = f"{alat}_{attr}"
                if col in self.df.columns:
                    if col not in config_features:
                        missing_features.append(col)
                    # Check numeric
                    if attr in ['Jumlah', 'EstimasiWatt', 'EstimasiJamPerHari', 'Energi_kWhPerHari']:
                        if col not in config_numeric:
                            missing_numeric.append(col)

            if missing_features or missing_numeric:
                alat_lain_issues.append(alat)
                print(f"\n   ⚠️  {alat}:")
                if missing_features:
                    print(f"      Missing dari features: {len(missing_features)}")
                    for f in missing_features:
                        print(f"        - {f}")
                if missing_numeric:
                    print(f"      Missing dari numeric_cols: {len(missing_numeric)}")
                    for f in missing_numeric:
                        print(f"        - {f}")

        if not alat_lain_issues:
            print("   ✅ Semua Alat_Lain lengkap")
        else:
            self.errors.append(f"Pola Alat_Lain: {len(alat_lain_issues)}/3 tidak lengkap")

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*80)
        print("RINGKASAN VALIDASI")
        print("="*80)

        print(f"\n❌ ERRORS: {len(self.errors)}")
        for err in self.errors:
            print(f"   - {err}")

        print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
        for warn in self.warnings:
            print(f"   - {warn}")

        if len(self.errors) == 0:
            print("\n" + "="*80)
            print("✅ VALIDASI SUKSES! Config sudah lengkap dan konsisten.")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print("❌ VALIDASI GAGAL! Perbaiki error di atas.")
            print("="*80)
            return False

    def save_report(self, filename="docs/config_validation_report.txt"):
        """Save report to file"""
        report_path = project_root / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONFIG VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Config: src/config/config.py\n\n")

            f.write(f"ERRORS: {len(self.errors)}\n")
            for err in self.errors:
                f.write(f"  - {err}\n")

            f.write(f"\nWARNINGS: {len(self.warnings)}\n")
            for warn in self.warnings:
                f.write(f"  - {warn}\n")

            if len(self.errors) == 0:
                f.write("\n" + "="*80 + "\n")
                f.write("✅ VALIDATION PASSED\n")
                f.write("="*80 + "\n")

        print(f"\n📄 Report saved to: {report_path}")

    def run(self):
        """Run all validations"""
        print("="*80)
        print("CONFIG FEATURE VALIDATOR")
        print("="*80)
        print(f"Dataset: {self.dataset_path}")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Config features: {len(self.config['features']['prabayar'])}")

        self.validate_features_list()
        self.validate_numeric_cols()
        self.validate_ordinal_encoding()
        self.validate_ordinal_consistency()
        self.validate_pattern_consistency()

        result = self.generate_report()
        self.save_report()

        return result


if __name__ == "__main__":
    validator = ConfigValidator()
    success = validator.run()
    sys.exit(0 if success else 1)
