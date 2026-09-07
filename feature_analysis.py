"""
Feature Analysis untuk Feature Selection (Prabayar Only).
Menghitung Permutation Importance tiap fitur terhadap model yang sudah dilatih.

Usage: python feature_analysis.py
"""

import json
import os
import sys

from src.pipeline.preprocessing import (
    load_and_preprocess,
    train_test_split,
    fit_minmax_scaler,
    transform_minmax,
    fit_target_scaler,
    transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.config.config import config
from src.models.neural_network import NeuralNetwork


def permutation_importance_mse(
    model: NeuralNetwork,
    x_test: list[list[float]],
    x_cat_test: list[dict[str, int]],
    y_test: list[float],
    feature_names: list[str],
    cat_feature_names: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[float]]:
    """Permutation importance: shuffle tiap fitur (numerik dan kategorikal), ukur kenaikan MSE."""
    import random
    import numpy as np
    import copy
    rng = random.Random(seed)

    x_test_np = np.array(x_test, dtype=np.float32)
    y_test_np = np.array(y_test, dtype=np.float32).reshape(-1, 1)

    # baseline MSE
    preds_base = model.predict(x_test_np, x_cat_test)
    base_mse = float(np.mean((preds_base - y_test_np) ** 2) / 2.0)

    all_names = []
    all_importances = []

    # 1. Permutasi fitur numerik
    n_num_features = len(x_test[0])
    for fi in range(n_num_features):
        deltas = []
        for _ in range(n_repeats):
            col = [row[fi] for row in x_test]
            rng.shuffle(col)
            x_perm = [row[:] for row in x_test]
            for i, row in enumerate(x_perm):
                row[fi] = col[i]

            x_perm_np = np.array(x_perm, dtype=np.float32)
            preds_perm = model.predict(x_perm_np, x_cat_test)
            perm_mse = float(np.mean((preds_perm - y_test_np) ** 2) / 2.0)
            deltas.append(perm_mse - base_mse)

        all_names.append(feature_names[fi])
        all_importances.append(sum(deltas) / len(deltas))

    # 2. Permutasi fitur kategorikal (Embedding)
    if x_cat_test and cat_feature_names:
        for cat_name in cat_feature_names:
            deltas = []
            for _ in range(n_repeats):
                col = [row.get(cat_name, 0) for row in x_cat_test]
                rng.shuffle(col)
                
                x_cat_perm = copy.deepcopy(x_cat_test)
                for i, row in enumerate(x_cat_perm):
                    row[cat_name] = col[i]
                
                preds_perm = model.predict(x_test_np, x_cat_perm)
                perm_mse = float(np.mean((preds_perm - y_test_np) ** 2) / 2.0)
                deltas.append(perm_mse - base_mse)

            all_names.append(cat_name)
            all_importances.append(sum(deltas) / len(deltas))

    return all_names, all_importances


def analyze():
    model_type = "prabayar"
    cfg = config[model_type]

    print(f"\n{'='*70}")
    print(f"  FEATURE ANALYSIS — PRABAYAR")
    print(f"{'='*70}\n")

    # Load data
    rows, _ = load_and_preprocess(cfg["dataset_path"])
    x_data, x_cat_data, y_data, feature_columns, embedding_configs, target_column = extract_features_and_target(rows, model_type)
    
    total_embedding_dim = sum(e_cfg["dim"] for e_cfg in embedding_configs)
    
    print(f"Dataset: {len(rows)} baris, {len(feature_columns)} fitur numerik, {len(embedding_configs)} fitur kategori (dim={total_embedding_dim})")
    print(f"Target: {target_column}\n")

    # Split
    x_train, x_cat_train, x_test, x_cat_test, y_train, y_test = train_test_split(x_data, x_cat_data, y_data, test_ratio=0.2, seed=42)

    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        print(f"[INFO] Model belum ada di {model_path}, tidak bisa menghitung permutation importance.")
        return

    print(f"{'─'*70}")
    print(f"  PERMUTATION IMPORTANCE (model: {model_path})")
    print(f"{'─'*70}")

    from src.models.prabayar import PrabayarModel
    model, _ = PrabayarModel.load(model_path)

    # Scale data
    x_scaler = fit_minmax_scaler(x_train)
    x_test_scaled = transform_minmax(x_test, x_scaler)
    y_scaler = fit_target_scaler(y_train, use_log=cfg.get("use_log_transform", False))
    y_test_scaled = transform_target(y_test, y_scaler)

    print("Menghitung permutation importance (5 repeats)...")
    
    cat_feature_names = [cfg["name"] for cfg in embedding_configs]
    all_names, imp = permutation_importance_mse(
        model, 
        x_test_scaled, 
        x_cat_test, 
        y_test_scaled, 
        feature_columns,
        cat_feature_names,
        n_repeats=5
    )

    imp_results = list(zip(all_names, imp))
    imp_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':>4} | {'Fitur':<52} | {'ΔMSE':>14}")
    print("─" * 75)
    try:
        max_val = float(max([float(i[1].item()) if hasattr(i[1], 'item') else float(i[1]) for i in imp_results]))
    except Exception:
        max_val = 1.0
    for rank, (fname, delta) in enumerate(imp_results, 1):
        try:
            delta_val = float(delta.item()) if hasattr(delta, 'item') else float(delta)
        except Exception:
            delta_val = 0.0
        bar = "█" * max(1, int(delta_val / max_val * 30)) if max_val > 0 and delta_val > 0 else ""
        print(f"{rank:>4} | {fname:<52} | {delta_val:>14.8f} {bar}")

    # Simpan hasil ke JSON
    output_path = "results/prabayar/feature_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "model_type": model_type,
        "target": target_column,
        "n_samples": len(rows),
        "n_features": len(feature_columns),
        "permutation_importance": [
            {"feature": fname, "delta_mse": (float(delta.item()) if hasattr(delta, 'item') else float(delta))}
            for fname, delta in imp_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nHasil disimpan ke: {output_path}")


def main():
    analyze()


if __name__ == "__main__":
    main()
