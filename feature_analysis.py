"""
Feature Analysis untuk Feature Selection.
Menghitung:
1. Korelasi Pearson setiap fitur terhadap target
2. Variance setiap fitur
3. Mean & Std tiap fitur
4. Permutation Importance (jika model sudah ada)

Usage: python feature_analysis.py <prabayar|pascabayar|all>
"""

import json
import math
import os
import sys

from src.pipeline.preprocessing import (
    load_csv,
    train_test_split,
    fit_minmax_scaler,
    transform_minmax,
    fit_target_scaler,
    transform_target,
    inverse_transform_target,
)
from src.pipeline.feature_extraction import extract_features_and_target
from src.config.config import config
from src.models.neural_network import NeuralNetwork


def pearson_correlation(x_col: list[float], y_col: list[float]) -> float:
    """Hitung korelasi Pearson antara dua kolom."""
    n = len(x_col)
    if n == 0:
        return 0.0
    mean_x = sum(x_col) / n
    mean_y = sum(y_col) / n
    
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_col, y_col))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_col))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_col))
    
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def compute_variance(col: list[float]) -> float:
    n = len(col)
    if n == 0:
        return 0.0
    mean = sum(col) / n
    return sum((x - mean) ** 2 for x in col) / n


def compute_stats(col: list[float]) -> dict:
    n = len(col)
    mean = sum(col) / n
    var = sum((x - mean) ** 2 for x in col) / n
    std = math.sqrt(var)
    return {
        "mean": mean,
        "std": std,
        "min": min(col),
        "max": max(col),
        "variance": var,
        "unique_count": len(set(col)),
    }


def permutation_importance_mse(
    model: NeuralNetwork,
    x_test: list[list[float]],
    y_test: list[float],
    n_repeats: int = 5,
    seed: int = 42,
) -> list[float]:
    """Permutation importance: shuffle tiap fitur, ukur kenaikan MSE."""
    import random
    rng = random.Random(seed)

    # baseline MSE
    preds_base = [model.forward(x) for x in x_test]
    base_mse = sum((p - a) ** 2 for p, a in zip(preds_base, y_test)) / len(y_test)

    n_features = len(x_test[0])
    importances = []

    for fi in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            # copy & shuffle column fi
            col = [row[fi] for row in x_test]
            rng.shuffle(col)
            x_perm = [row[:] for row in x_test]
            for i, row in enumerate(x_perm):
                row[fi] = col[i]

            preds_perm = [model.forward(x) for x in x_perm]
            perm_mse = sum((p - a) ** 2 for p, a in zip(preds_perm, y_test)) / len(y_test)
            deltas.append(perm_mse - base_mse)

        importances.append(sum(deltas) / len(deltas))

    return importances


def analyze(model_type: str):
    cfg = config[model_type]

    print(f"\n{'='*70}")
    print(f"  FEATURE ANALYSIS — {model_type.upper()}")
    print(f"{'='*70}\n")

    # Load data
    rows = load_csv(cfg["dataset_path"])
    x_data, y_data, feature_columns, target_column = extract_features_and_target(rows, model_type)
    print(f"Dataset: {len(rows)} baris, {len(feature_columns)} fitur")
    print(f"Target: {target_column}\n")

    # Split (same seed as training)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_ratio=0.2, seed=42)

    # ============================================================
    # 1. STATISTIK DESKRIPTIF & KORELASI (skala asli)
    # ============================================================
    print(f"{'─'*70}")
    print(f"  1. STATISTIK DESKRIPTIF & KORELASI PEARSON (skala asli)")
    print(f"{'─'*70}")
    
    header = f"{'No':>3} | {'Fitur':<52} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12} | {'Var':>14} | {'Unique':>6} | {'Corr(target)':>13}"
    print(header)
    print("─" * len(header))

    all_results = []
    for fi, fname in enumerate(feature_columns):
        col = [row[fi] for row in x_data]
        stats = compute_stats(col)
        corr = pearson_correlation(col, y_data)
        
        all_results.append({
            "index": fi,
            "feature": fname,
            **stats,
            "correlation": corr,
            "abs_correlation": abs(corr),
        })

        print(
            f"{fi+1:>3} | {fname:<52} | {stats['mean']:>12.4f} | {stats['std']:>12.4f} | "
            f"{stats['min']:>12.4f} | {stats['max']:>12.4f} | {stats['variance']:>14.4f} | "
            f"{stats['unique_count']:>6} | {corr:>+13.4f}"
        )

    # ============================================================
    # 2. RANKING KORELASI (|corr| terbesar → terkecil)
    # ============================================================
    print(f"\n{'─'*70}")
    print(f"  2. RANKING KORELASI (|corr| descending)")
    print(f"{'─'*70}")
    
    sorted_by_corr = sorted(all_results, key=lambda r: r["abs_correlation"], reverse=True)
    print(f"{'Rank':>4} | {'Fitur':<52} | {'Corr':>10} | {'|Corr|':>10}")
    print("─" * 82)
    for rank, r in enumerate(sorted_by_corr, 1):
        print(f"{rank:>4} | {r['feature']:<52} | {r['correlation']:>+10.4f} | {r['abs_correlation']:>10.4f}")

    # ============================================================
    # 3. FITUR DENGAN VARIANCE RENDAH (kandidat drop)
    # ============================================================
    print(f"\n{'─'*70}")
    print(f"  3. FITUR VARIANCE RENDAH (kemungkinan tidak informatif)")
    print(f"{'─'*70}")
    
    sorted_by_var = sorted(all_results, key=lambda r: r["variance"])
    print(f"{'No':>3} | {'Fitur':<52} | {'Variance':>14} | {'Unique':>6} | {'|Corr|':>10}")
    print("─" * 92)
    for r in sorted_by_var:
        flag = " ⚠️" if r["variance"] < 0.01 or r["unique_count"] <= 2 else ""
        print(
            f"{r['index']+1:>3} | {r['feature']:<52} | {r['variance']:>14.6f} | "
            f"{r['unique_count']:>6} | {r['abs_correlation']:>10.4f}{flag}"
        )

    # ============================================================
    # 4. KORELASI ANTAR FITUR (deteksi multikolinearitas)
    # ============================================================
    print(f"\n{'─'*70}")
    print(f"  4. PASANGAN FITUR DENGAN KORELASI TINGGI (|r| > 0.85)")
    print(f"{'─'*70}")

    high_corr_pairs = []
    n_feat = len(feature_columns)
    for i in range(n_feat):
        col_i = [row[i] for row in x_data]
        for j in range(i + 1, n_feat):
            col_j = [row[j] for row in x_data]
            r = pearson_correlation(col_i, col_j)
            if abs(r) > 0.85:
                high_corr_pairs.append((feature_columns[i], feature_columns[j], r))

    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"{'Fitur A':<45} | {'Fitur B':<45} | {'Corr':>8}")
        print("─" * 103)
        for a, b, r in high_corr_pairs:
            print(f"{a:<45} | {b:<45} | {r:>+8.4f}")
    else:
        print("Tidak ada pasangan fitur dengan |r| > 0.85")

    # ============================================================
    # 5. PERMUTATION IMPORTANCE (jika model tersedia)
    # ============================================================
    model_path = cfg["model_path"]
    if os.path.exists(model_path):
        print(f"\n{'─'*70}")
        print(f"  5. PERMUTATION IMPORTANCE (model: {model_path})")
        print(f"{'─'*70}")

        if model_type == "prabayar":
            from src.models.prabayar import PrabayarModel
            model, _ = PrabayarModel.load(model_path)
        else:
            from src.models.pascabayar import PascabayarModel
            model, _ = PascabayarModel.load(model_path)

        # Scale data
        x_scaler = fit_minmax_scaler(x_train)
        x_test_scaled = transform_minmax(x_test, x_scaler)
        y_scaler = fit_target_scaler(y_train, use_log=cfg.get("use_log_transform", False))
        y_test_scaled = transform_target(y_test, y_scaler)

        print("Menghitung permutation importance (5 repeats)...")
        imp = permutation_importance_mse(model, x_test_scaled, y_test_scaled, n_repeats=5)

        imp_results = list(zip(feature_columns, imp))
        imp_results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'Rank':>4} | {'Fitur':<52} | {'ΔMSE':>14}")
        print("─" * 75)
        for rank, (fname, delta) in enumerate(imp_results, 1):
            bar = "█" * max(1, int(delta / max(i[1] for i in imp_results) * 30)) if max(i[1] for i in imp_results) > 0 and delta > 0 else ""
            print(f"{rank:>4} | {fname:<52} | {delta:>14.8f} {bar}")
    else:
        print(f"\n[INFO] Model belum ada di {model_path}, skip permutation importance.")

    # ============================================================
    # SIMPAN HASIL KE JSON
    # ============================================================
    output_path = f"results/{model_type}/feature_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "model_type": model_type,
        "target": target_column,
        "n_samples": len(rows),
        "n_features": len(feature_columns),
        "features": all_results,
        "high_corr_pairs": [
            {"feature_a": a, "feature_b": b, "correlation": r}
            for a, b, r in high_corr_pairs
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nHasil disimpan ke: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python feature_analysis.py <prabayar|pascabayar|all>")
        sys.exit(1)

    choice = sys.argv[1].lower()
    if choice == "all":
        for mt in ["prabayar", "pascabayar"]:
            analyze(mt)
    elif choice in config:
        analyze(choice)
    else:
        print(f"Unknown: {choice}. Use: prabayar, pascabayar, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
