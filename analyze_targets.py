import pandas as pd
import os
from src.config.config import config

def analyze_targets():
    print("=== Analisis Target Model Prabayar ===")
    prabayar_path = config["prabayar"]["dataset_path"]
    prabayar_target = config["prabayar"]["target"]
    
    if os.path.exists(prabayar_path):
        df_prabayar = pd.read_csv(prabayar_path)
        if prabayar_target in df_prabayar.columns:
            target_data = df_prabayar[prabayar_target]
            print(f"Dataset: {prabayar_path}")
            print(f"Kolom Target: {prabayar_target}")
            print(f"Min  : {target_data.min():.2f}")
            print(f"Max  : {target_data.max():.2f}")
            print(f"Mean : {target_data.mean():.2f}")
        else:
            print(f"Kolom '{prabayar_target}' tidak ditemukan di dataset prabayar.")
    else:
        print(f"File {prabayar_path} tidak ditemukan.")

if __name__ == "__main__":
    analyze_targets()
