#!/usr/bin/env python3
"""
Hyperparameter Tuning for Capacity-Specific Models

Script untuk mencari konfigurasi terbaik untuk setiap capacity model
menggunakan Successive Halving Grid Search.

Usage:
    # Tune single capacity
    python tune_capacity_models.py --capacity 900

    # Tune all capacities
    python tune_capacity_models.py --all

    # Quick test (fewer candidates)
    python tune_capacity_models.py --capacity 450 --quick

    # Custom configuration
    python tune_capacity_models.py --capacity 1300 --n-candidates 100 --epochs 20,60,180
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tuning.grid_search import (
    HalvingConfig,
    tune_capacity_model,
    tune_prabayar,
)
from src.tuning.train_tuning import load_data


def save_tuning_results(capacity: str, results: list, output_dir: str = "results/tuning"):
    """
    Save tuning results to JSON file.
    
    Args:
        capacity: Capacity category
        results: Tuning results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tuning_{capacity}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare results for JSON serialization
    output = {
        "capacity": capacity,
        "timestamp": timestamp,
        "n_candidates": len(results),
        "n_converged": sum(1 for r in results if not r.get("diverged")),
        "results": results[:10],  # Save top 10 only
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  💾 Results saved to: {filepath}")
    
    # Also save best config separately
    valid = [r for r in results if not r.get("diverged")]
    if valid:
        best = valid[0]
        best_filename = f"best_config_{capacity}.json"
        best_filepath = os.path.join(output_dir, best_filename)
        
        best_config = {
            "capacity": capacity,
            "timestamp": timestamp,
            "metrics": {
                "rmse": best["rmse"],
                "mae": best["mae"],
                "mape": best["mape"],
                "r2": best["r2"],
                "best_epoch": best.get("best_epoch"),
            },
            "params": best["params"],
            "recommended_config": {
                "layer_sizes": best["params"]["layer_sizes"],
                "learning_rate": best["params"]["learning_rate"],
                "l2_lambda": best["params"]["l2_lambda"],
                "clip_value": best["params"]["clip_value"],
                "batch_size": best["params"]["batch_size"],
            }
        }
        
        with open(best_filepath, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Best config saved to: {best_filepath}")


def print_capacity_summary(capacity: str, results: list, n_samples: int):
    """Print summary for a capacity."""
    valid = [r for r in results if not r.get("diverged")]
    
    print(f"\n{'='*70}")
    print(f"  CAPACITY: {capacity}VA")
    print(f"{'='*70}")
    print(f"  Dataset: {n_samples} training samples")
    print(f"  Converged: {len(valid)}/{len(results)} candidates")
    
    if valid:
        best = valid[0]
        print(f"\n  🏆 Best Configuration:")
        print(f"  {'─'*70}")
        print(f"    Metrics:")
        print(f"      RMSE  = {best['rmse']:.4f}")
        print(f"      MAE   = {best['mae']:.4f}")
        print(f"      MAPE  = {best['mape']:.4f}%")
        print(f"      R²    = {best['r2']:.6f}")
        print(f"      Epoch = {best.get('best_epoch')}")
        print(f"\n    Hyperparameters:")
        print(f"      Architecture:   {best['params']['layer_sizes']}")
        print(f"      Learning Rate:  {best['params']['learning_rate']}")
        print(f"      L2 Lambda:      {best['params']['l2_lambda']}")
        print(f"      Clip Value:     {best['params']['clip_value']}")
        print(f"      Batch Size:     {best['params']['batch_size']}")
        
        # Show top 3
        if len(valid) > 1:
            print(f"\n    Top 3 Candidates:")
            for i, r in enumerate(valid[:3], 1):
                print(f"      {i}. RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, "
                      f"MAPE={r['mape']:.4f}%, R²={r['r2']:.6f}")
    else:
        print(f"\n  ⚠️  No converged candidates!")
        print(f"     Try:")
        print(f"       - Increase regularization (l2_lambda)")
        print(f"       - Decrease learning rate")
        print(f"       - Simplify architecture")
        print(f"       - Increase gradient clipping")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for capacity-specific models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune 900VA model
  python tune_capacity_models.py --capacity 900

  # Tune all capacities sequentially
  python tune_capacity_models.py --all

  # Quick test with fewer candidates
  python tune_capacity_models.py --capacity 450 --quick

  # Custom configuration
  python tune_capacity_models.py --capacity 1300 --n-candidates 150 --epochs 20,60,180
        """
    )
    
    # Capacity selection
    capacity_group = parser.add_mutually_exclusive_group(required=True)
    capacity_group.add_argument(
        "--capacity",
        type=str,
        choices=["450", "900", "1300", "2200", "3500"],
        help="Specific capacity to tune"
    )
    capacity_group.add_argument(
        "--all",
        action="store_true",
        help="Tune all capacity models"
    )
    
    # Tuning configuration
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of initial candidates (default: auto based on dataset size)"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Comma-separated epoch budgets (default: auto based on dataset size)"
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=3,
        help="Halving factor (default: 3)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (50 candidates, shorter epochs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tuning",
        help="Output directory for results (default: results/tuning)"
    )
    
    args = parser.parse_args()
    
    # Determine capacities to tune
    if args.all:
        capacities = ["450", "900", "1300", "2200", "3500"]
    else:
        capacities = [args.capacity]
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  HYPERPARAMETER TUNING - CAPACITY-SPECIFIC MODELS".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    all_results = {}
    
    for capacity in capacities:
        dataset_name = f"prabayar_{capacity}"
        
        print(f"\n{'▓'*70}")
        print(f"  LOADING DATA: {capacity}VA")
        print(f"{'▓'*70}")
        
        try:
            (x_tr, x_va, y_tr, y_va, y_va_orig, n_feat, yscaler, _) = load_data(dataset_name)
            n_samples = len(x_tr)
            print(f"  Train: {n_samples} | Val: {len(x_va)} | Features: {n_feat}")
            
            # Auto-configure based on dataset size
            if args.quick:
                n_candidates = 50
                rung_epochs = (15, 45, 90)
            elif args.n_candidates:
                n_candidates = args.n_candidates
                if args.epochs:
                    rung_epochs = tuple(int(x) for x in args.epochs.split(","))
                else:
                    rung_epochs = (30, 90, 270)
            else:
                # Auto-configure based on dataset size
                if n_samples < 20:
                    n_candidates = 80
                    rung_epochs = (20, 60, 150)
                elif n_samples < 100:
                    n_candidates = 120
                    rung_epochs = (25, 75, 200)
                elif n_samples < 200:
                    n_candidates = 150
                    rung_epochs = (30, 90, 250)
                else:
                    n_candidates = 200
                    rung_epochs = (30, 90, 270)
            
            patience_per_rung = tuple(min(20, ep // 3) for ep in rung_epochs)
            
            cfg = HalvingConfig(
                n_initial_candidates=n_candidates,
                rung_epochs=rung_epochs,
                eta=args.eta,
                patience_per_rung=patience_per_rung,
                n_workers=args.workers,
                seed=args.seed,
            )
            
            print(f"\n  Configuration:")
            print(f"    Initial candidates: {cfg.n_initial_candidates}")
            print(f"    Rung epochs:        {cfg.rung_epochs}")
            print(f"    Eta:                {cfg.eta}")
            print(f"    Patience:           {cfg.patience_per_rung}")
            print(f"    Workers:            {cfg.n_workers or 'auto'}")
            print(f"    Seed:               {cfg.seed}")
            
            import time
            t_start = time.time()
            
            print(f"\n  Starting tuning for {capacity}VA...")
            results = tune_capacity_model(
                capacity, x_tr, y_tr, x_va, y_va, y_va_orig, 
                n_feat, yscaler, cfg
            )
            
            elapsed = time.time() - t_start
            print(f"\n  ✅ Tuning completed in {elapsed/60:.1f} minutes")
            
            all_results[capacity] = results
            
            # Print summary
            print_capacity_summary(capacity, results, n_samples)
            
            # Save results
            save_tuning_results(capacity, results, args.output_dir)
            
        except Exception as e:
            print(f"\n  ❌ Error tuning {capacity}VA: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'═'*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*70}")
    
    for capacity, results in all_results.items():
        valid = [r for r in results if not r.get("diverged")]
        status = "✅" if valid else "⚠️"
        if valid:
            best = valid[0]
            print(f"  {status} {capacity}VA: RMSE={best['rmse']:.4f}, "
                  f"MAE={best['mae']:.4f}, R²={best['r2']:.6f}")
        else:
            print(f"  {status} {capacity}VA: No converged candidates")
    
    print(f"\n{'═'*70}")
    print(f"  💾 Results saved to: {args.output_dir}/")
    print(f"  📊 Use best_config_*.json files to update your configs")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
