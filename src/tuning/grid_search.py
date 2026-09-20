"""
grid_search.py (Successive Halving / Random Search)

Hyperparameter tuning untuk model Prabayar dan Capacity-specific models.
Mendukung tuning untuk 450VA, 900VA, 1300VA, 2200VA, 3500VA.
"""

import copy
import math
import random
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tuning.train_tuning import (
    compute_rmse, compute_mae, compute_mape, compute_r2, evaluate_metrics,
    is_invalid_number, make_divergent_metrics,
    load_data, predict_mlp,
)

from src.models.prabayar import PrabayarModel
from src.models.model_factory import get_model_class_for_capacity, get_available_capacities
from src.pipeline.preprocessing import inverse_transform_target


# =====================================================================
# RANDOM SEARCH SAMPLING
# =====================================================================

def sample_param_pool(param_grid: dict, n_samples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    seen = set()
    pool = []

    space_size = 1
    for k in keys:
        space_size *= len(param_grid[k])
    n_samples = min(n_samples, space_size)

    while len(pool) < n_samples:
        combo = {k: rng.choice(param_grid[k]) for k in keys}
        combo_key = repr(sorted(combo.items(), key=lambda kv: kv[0]))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        pool.append(combo)

    return pool


# =====================================================================
# GRID DEFINITION — PRABAYAR & CAPACITY MODELS
# =====================================================================

def get_search_space_prabayar(n_features: int) -> dict:
    """Search space for unified Prabayar model."""
    return {
        "layer_sizes": [
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 128, 1],
            [n_features, 64, 32, 1],
            [n_features, 64, 48, 1],
            [n_features, 64, 16, 1],
            [n_features, 48, 32, 1],
            [n_features, 48, 16, 1],
            [n_features, 32, 16, 1],
            [n_features, 128, 64, 1],
        ],
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "clip_value": [5.0, 10.0, 15.0],
        "batch_size": [16, 32, 64, 128],
    }


def get_search_space_capacity(n_features: int, capacity: str, n_samples: int) -> dict:
    """
    Search space for capacity-specific models.
    Adapted based on dataset size to prevent overfitting.
    
    Args:
        n_features: Number of input features
        capacity: Capacity category ("450", "900", "1300", "2200", "3500")
        n_samples: Number of training samples
    
    Returns:
        Parameter search space dictionary
    """
    # Adjust architecture complexity based on dataset size
    # Rule of thumb: Total params < n_samples / 10
    
    if n_samples < 20:  # Tiny dataset (3500VA with ~11 samples)
        layer_sizes = [
            [n_features, 8, 1],
            [n_features, 16, 1],
            [n_features, 12, 1],
            [n_features, 16, 8, 1],
        ]
        learning_rates = [1e-4, 5e-4, 1e-3]
        l2_lambdas = [1e-3, 1e-2, 5e-2]  # Strong regularization
        batch_sizes = [4, 8]
        clip_values = [5.0, 10.0]
    
    elif n_samples < 100:  # Small dataset (450VA with ~80 samples, 2200VA with ~141)
        layer_sizes = [
            [n_features, 16, 1],
            [n_features, 32, 1],
            [n_features, 24, 1],
            [n_features, 32, 16, 1],
            [n_features, 24, 12, 1],
            [n_features, 16, 8, 1],
        ]
        learning_rates = [1e-4, 5e-4, 1e-3]
        l2_lambdas = [1e-4, 1e-3, 1e-2]
        batch_sizes = [8, 16, 32]
        clip_values = [5.0, 10.0, 15.0]
    
    elif n_samples < 200:  # Medium dataset (1300VA with ~149)
        layer_sizes = [
            [n_features, 32, 1],
            [n_features, 64, 1],
            [n_features, 48, 1],
            [n_features, 64, 32, 1],
            [n_features, 48, 24, 1],
            [n_features, 32, 16, 1],
            [n_features, 64, 16, 1],
        ]
        learning_rates = [5e-5, 1e-4, 5e-4, 1e-3]
        l2_lambdas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        batch_sizes = [16, 32, 64]
        clip_values = [5.0, 10.0, 15.0]
    
    else:  # Large dataset (900VA with ~259 samples)
        layer_sizes = [
            [n_features, 64, 1],
            [n_features, 128, 1],
            [n_features, 96, 1],
            [n_features, 128, 64, 1],
            [n_features, 96, 48, 1],
            [n_features, 64, 32, 1],
            [n_features, 128, 32, 1],
            [n_features, 96, 32, 1],
        ]
        learning_rates = [5e-5, 1e-4, 5e-4, 1e-3]
        l2_lambdas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        batch_sizes = [16, 32, 64, 128]
        clip_values = [5.0, 10.0, 15.0]
    
    return {
        "layer_sizes": layer_sizes,
        "learning_rate": learning_rates,
        "l2_lambda": l2_lambdas,
        "clip_value": clip_values,
        "batch_size": batch_sizes,
    }


# =====================================================================
# TRAINER: MLP DENGAN VAL-BASED EARLY STOPPING
# =====================================================================

def train_mlp_early_stop(
    model,
    x_train, y_train_scaled,
    x_val, y_val_scaled,
    learning_rate: float, max_epochs: int, batch_size: int,
    patience: int = 15, check_every: int = 5,
):
    X = np.array(x_train, dtype=np.float32)
    Y = np.array(y_train_scaled, dtype=np.float32).reshape(-1, 1)
    Yv = np.array(y_val_scaled, dtype=np.float64)
    n = X.shape[0]

    best_val = float("inf")
    best_model = None
    best_epoch = 0
    no_improve = 0
    diverged = False
    loss_start = None

    for ep in range(max_epochs):
        idx = np.random.permutation(n)
        X_shuf, Y_shuf = X[idx], Y[idx]

        epoch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            loss = model.train_batch(X_shuf[start:end], Y_shuf[start:end], learning_rate)
            epoch_losses.append(loss)
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if ep == 0:
            loss_start = epoch_loss
        if is_invalid_number(epoch_loss):
            diverged = True
            break

        if ep % check_every == 0 or ep == max_epochs - 1:
            val_preds = predict_mlp(model, x_val).astype(np.float64)
            if not np.all(np.isfinite(val_preds)):
                diverged = True
                break
            val_rmse = compute_rmse(Yv, val_preds)
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_model = copy.deepcopy(model)
                best_epoch = ep
                no_improve = 0
            else:
                no_improve += check_every
                if no_improve >= patience:
                    break

    return {
        "model": best_model if best_model is not None else model,
        "best_val_rmse_scaled": best_val,
        "best_epoch": best_epoch,
        "diverged": diverged,
        "loss_start": loss_start,
    }


# =====================================================================
# WORKER FUNCTION
# =====================================================================

def _eval_mlp_candidate(args):
    (ModelClass, params, epoch_budget, patience,
     x_train, y_train_scaled, x_val, y_val_scaled, y_val_original, y_scaler) = args

    np.random.seed(42)
    model = ModelClass(
        layer_sizes=list(params["layer_sizes"]),
        seed=42,
        clip_value=params["clip_value"],
        l2_lambda=params["l2_lambda"],
    )

    info = train_mlp_early_stop(
        model, x_train, y_train_scaled, x_val, y_val_scaled,
        params["learning_rate"], epoch_budget, params["batch_size"],
        patience=patience,
    )

    if info["diverged"]:
        m = make_divergent_metrics(params, info["loss_start"])
        m["best_epoch"] = info["best_epoch"]
        return m

    preds_scaled = predict_mlp(info["model"], x_val)
    preds_original = np.array([inverse_transform_target(float(p), y_scaler) for p in preds_scaled])
    y_true = np.array(y_val_original, dtype=np.float64)

    metrics = evaluate_metrics(y_true, preds_original)
    metrics["params"] = params
    metrics["diverged"] = False
    metrics["best_epoch"] = info["best_epoch"]
    metrics["epoch_budget"] = epoch_budget
    return metrics


# =====================================================================
# SUCCESSIVE HALVING DRIVER
# =====================================================================

@dataclass
class HalvingConfig:
    n_initial_candidates: int = 200
    rung_epochs: tuple = (30, 90, 270)
    eta: int = 3
    patience_per_rung: tuple = (10, 15, 25)
    n_workers: int | None = None
    seed: int = 42


def run_successive_halving(
    eval_fn, worker_arg_builder, param_pool: list[dict], cfg: HalvingConfig,
):
    candidates = param_pool
    last_results = []

    for rung_idx, epoch_budget in enumerate(cfg.rung_epochs):
        patience = cfg.patience_per_rung[min(rung_idx, len(cfg.patience_per_rung) - 1)]
        t0 = time.time()
        print(f"\n  Rung {rung_idx + 1}/{len(cfg.rung_epochs)}: "
              f"{len(candidates)} kandidat, budget={epoch_budget} epoch, patience={patience}")

        work_items = [worker_arg_builder(p, epoch_budget, patience) for p in candidates]

        results = []
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            futures = [ex.submit(eval_fn, item) for item in work_items]
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda r: (r["rmse"], r["mae"]))
        n_diverged = sum(1 for r in results if r.get("diverged"))
        elapsed = time.time() - t0

        print(f"    Selesai dalam {elapsed:.1f}s | divergen: {n_diverged}/{len(results)}")
        if results and not results[0].get("diverged"):
            print(f"    Best RMSE rung ini: {results[0]['rmse']:.4f} "
                  f"(best_epoch={results[0].get('best_epoch')})")

        keep_n = max(1, math.ceil(len(results) / cfg.eta))
        candidates = [r["params"] for r in results[:keep_n]]
        last_results = results

    return last_results


# =====================================================================
# RUNNER
# =====================================================================

def tune_prabayar(x_train, y_train, x_val, y_val, y_val_orig, n_features, y_scaler, cfg: HalvingConfig):
    """Tune unified Prabayar model."""
    space = get_search_space_prabayar(n_features)
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (PrabayarModel, params, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_mlp_candidate, build_args, pool, cfg)


def tune_capacity_model(
    capacity: str,
    x_train, y_train, 
    x_val, y_val, y_val_orig,
    n_features, y_scaler,
    cfg: HalvingConfig
):
    """
    Tune capacity-specific model with adaptive search space.
    
    Args:
        capacity: Capacity category ("450", "900", "1300", "2200", "3500")
        x_train, y_train: Training data
        x_val, y_val, y_val_orig: Validation data
        n_features: Number of features
        y_scaler: Target scaler
        cfg: Halving configuration
    
    Returns:
        List of results sorted by performance
    """
    ModelClass = get_model_class_for_capacity(capacity)
    n_samples = len(x_train)
    
    space = get_search_space_capacity(n_features, capacity, n_samples)
    pool = sample_param_pool(space, cfg.n_initial_candidates, seed=cfg.seed)

    def build_args(params, budget, patience):
        return (ModelClass, params, budget, patience,
                x_train, y_train, x_val, y_val, y_val_orig, y_scaler)

    return run_successive_halving(_eval_mlp_candidate, build_args, pool, cfg)


# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid search tuning for Prabayar models")
    parser.add_argument(
        "--capacity",
        type=str,
        choices=["base", "450", "900", "1300", "2200", "3500", "all"],
        default="base",
        help="Capacity to tune (base=unified model, or specific capacity, or all)"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=200,
        help="Number of initial candidates (default: 200)"
    )
    parser.add_argument(
        "--rung-epochs",
        type=str,
        default="30,90,270",
        help="Comma-separated epoch budgets per rung (default: 30,90,270)"
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
    
    args = parser.parse_args()
    
    rung_epochs = tuple(int(x) for x in args.rung_epochs.split(","))
    patience_per_rung = tuple(min(15, ep // 3) for ep in rung_epochs)
    
    cfg = HalvingConfig(
        n_initial_candidates=args.n_candidates,
        rung_epochs=rung_epochs,
        eta=args.eta,
        patience_per_rung=patience_per_rung,
        n_workers=args.workers,
        seed=args.seed,
    )
    
    print("\n" + "═" * 70)
    print("  HYPERPARAMETER TUNING - PRABAYAR MODELS")
    print("═" * 70)
    print(f"  Configuration:")
    print(f"    - Initial candidates: {cfg.n_initial_candidates}")
    print(f"    - Rung epochs: {cfg.rung_epochs}")
    print(f"    - Eta (halving factor): {cfg.eta}")
    print(f"    - Patience per rung: {cfg.patience_per_rung}")
    print(f"    - Workers: {cfg.n_workers or 'auto'}")
    print(f"    - Seed: {cfg.seed}")
    print("═" * 70)
    
    all_results = {}
    
    # Determine which capacities to tune
    if args.capacity == "all":
        capacities = ["450", "900", "1300", "2200", "3500"]
    elif args.capacity == "base":
        capacities = ["base"]
    else:
        capacities = [args.capacity]
    
    for capacity in capacities:
        dataset_name = "prabayar" if capacity == "base" else f"prabayar_{capacity}"
        
        print(f"\n" + "▓" * 70)
        print(f"  LOADING DATA: {dataset_name.upper()}")
        print("▓" * 70)
        
        try:
            (x_tr, x_va, y_tr, y_va, y_va_orig, n_feat, yscaler, _) = load_data(dataset_name)
            print(f"  Train: {len(x_tr)} | Val: {len(x_va)} | Fitur: {n_feat}")
            
            print(f"\n  Starting tuning for {dataset_name}...")
            t_start = time.time()
            
            if capacity == "base":
                results = tune_prabayar(x_tr, y_tr, x_va, y_va, y_va_orig, n_feat, yscaler, cfg)
            else:
                results = tune_capacity_model(capacity, x_tr, y_tr, x_va, y_va, y_va_orig, n_feat, yscaler, cfg)
            
            elapsed = time.time() - t_start
            print(f"\n  Tuning completed in {elapsed/60:.1f} minutes")
            
            all_results[capacity] = results
            
        except Exception as e:
            print(f"  ❌ Error tuning {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "═" * 70)
    print("  TUNING SUMMARY")
    print("═" * 70)
    
    for capacity, results in all_results.items():
        dataset_name = "PrabayarModel (unified)" if capacity == "base" else f"Prabayar{capacity}Model"
        valid = [r for r in results if not r.get("diverged")]
        print(f"\n  {dataset_name}:")
        print(f"    Converged: {len(valid)}/{len(results)} candidates")
        
        if valid:
            best = valid[0]
            print(f"    Best metrics:")
            print(f"      RMSE  = {best['rmse']:.4f}")
            print(f"      MAE   = {best['mae']:.4f}")
            print(f"      MAPE  = {best['mape']:.4f}%")
            print(f"      R²    = {best['r2']:.6f}")
            print(f"      Epoch = {best.get('best_epoch')}")
            print(f"    Best params:")
            for key, val in best['params'].items():
                print(f"      {key}: {val}")
        else:
            print(f"    ⚠️  No converged candidates!")
    
    print("\n" + "═" * 70)
    print("  Tuning complete! Use best parameters in config files.")
    print("═" * 70)


if __name__ == "__main__":
    main()
