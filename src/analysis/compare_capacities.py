"""
Comparative Analysis Module for Capacity-Specific Models

This module compares performance across different capacity models.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config.config import config


def load_capacity_metrics(capacity: str) -> dict:
    """Load metrics for a specific capacity model."""
    cfg = config["capacity_configs"][capacity]
    metrics_path = os.path.join(cfg["metrics_dir"], "evaluation_metrics.json")
    
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_metrics() -> Dict[str, dict]:
    """Load metrics for all capacity models."""
    all_metrics = {}
    
    for capacity in config["capacity_configs"].keys():
        metrics = load_capacity_metrics(capacity)
        if metrics:
            all_metrics[capacity] = metrics
    
    return all_metrics


def generate_comparison_report(output_path: str = "results/comparative_analysis/comparison_report.json"):
    """
    Generate comprehensive comparison report across all capacity models.
    
    Args:
        output_path: Path to save the report
    """
    all_metrics = load_all_metrics()
    
    if not all_metrics:
        print("No metrics found. Please train models first.")
        return
    
    # Extract key metrics
    comparison = []
    
    for capacity, metrics in sorted(all_metrics.items(), key=lambda x: int(x[0])):
        eval_metrics = metrics.get("evaluasi_skala_asli", {})
        data_info = metrics.get("data_info", {})
        
        comparison.append({
            "capacity": f"{capacity}VA",
            "total_samples": data_info.get("total_samples", 0),
            "train_samples": data_info.get("train_samples", 0),
            "test_samples": data_info.get("test_samples", 0),
            "architecture": metrics.get("arsitektur", []),
            "total_epochs": metrics.get("total_epochs", 0),
            "rmse": eval_metrics.get("rmse", 0),
            "mae": eval_metrics.get("mae", 0),
            "mape": eval_metrics.get("mape", 0),
            "r2": eval_metrics.get("r2", 0),
            "learning_rate": metrics.get("learning_rate", 0),
            "batch_size": metrics.get("batch_size", 0),
        })
    
    # Calculate statistics
    rmse_values = [c["rmse"] for c in comparison]
    mae_values = [c["mae"] for c in comparison]
    mape_values = [c["mape"] for c in comparison]
    r2_values = [c["r2"] for c in comparison]
    
    statistics = {
        "rmse": {
            "mean": float(np.mean(rmse_values)),
            "std": float(np.std(rmse_values)),
            "min": float(np.min(rmse_values)),
            "max": float(np.max(rmse_values)),
            "best_capacity": comparison[np.argmin(rmse_values)]["capacity"],
        },
        "mae": {
            "mean": float(np.mean(mae_values)),
            "std": float(np.std(mae_values)),
            "min": float(np.min(mae_values)),
            "max": float(np.max(mae_values)),
            "best_capacity": comparison[np.argmin(mae_values)]["capacity"],
        },
        "mape": {
            "mean": float(np.mean(mape_values)),
            "std": float(np.std(mape_values)),
            "min": float(np.min(mape_values)),
            "max": float(np.max(mape_values)),
            "best_capacity": comparison[np.argmin(mape_values)]["capacity"],
        },
        "r2": {
            "mean": float(np.mean(r2_values)),
            "std": float(np.std(r2_values)),
            "min": float(np.min(r2_values)),
            "max": float(np.max(r2_values)),
            "best_capacity": comparison[np.argmax(r2_values)]["capacity"],
        },
    }
    
    report = {
        "generated_at": "2026-09-18",
        "models_compared": len(comparison),
        "comparison": comparison,
        "statistics": statistics,
    }
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\nComparison report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("  CAPACITY MODELS COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Capacity':<12} {'Samples':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'R²':<10}")
    print("-"*70)
    
    for item in comparison:
        print(f"{item['capacity']:<12} "
              f"{item['total_samples']:<10} "
              f"{item['rmse']:<12.4f} "
              f"{item['mae']:<12.4f} "
              f"{item['mape']:<10.2f} "
              f"{item['r2']:<10.4f}")
    
    print("\n" + "-"*70)
    print("\nBest Performance:")
    print(f"  RMSE: {statistics['rmse']['best_capacity']} ({statistics['rmse']['min']:.4f})")
    print(f"  MAE:  {statistics['mae']['best_capacity']} ({statistics['mae']['min']:.4f})")
    print(f"  MAPE: {statistics['mape']['best_capacity']} ({statistics['mape']['min']:.2f}%)")
    print(f"  R²:   {statistics['r2']['best_capacity']} ({statistics['r2']['max']:.4f})")
    
    return report


def plot_metrics_comparison(output_dir: str = "results/comparative_analysis"):
    """
    Generate comparison plots for all metrics.
    
    Args:
        output_dir: Directory to save plots
    """
    all_metrics = load_all_metrics()
    
    if not all_metrics:
        print("No metrics found. Please train models first.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    capacities = []
    samples = []
    rmse_values = []
    mae_values = []
    mape_values = []
    r2_values = []
    
    for capacity in sorted(all_metrics.keys(), key=lambda x: int(x)):
        metrics = all_metrics[capacity]
        eval_metrics = metrics.get("evaluasi_skala_asli", {})
        data_info = metrics.get("data_info", {})
        
        capacities.append(f"{capacity}VA")
        samples.append(data_info.get("total_samples", 0))
        rmse_values.append(eval_metrics.get("rmse", 0))
        mae_values.append(eval_metrics.get("mae", 0))
        mape_values.append(eval_metrics.get("mape", 0))
        r2_values.append(eval_metrics.get("r2", 0))
    
    # Plot 1: RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(capacities, rmse_values, color='#1565C0', alpha=0.8)
    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('RMSE (hari)', fontsize=12)
    ax.set_title('RMSE Comparison Across Capacity Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 2: MAE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(capacities, mae_values, color='#388E3C', alpha=0.8)
    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('MAE (hari)', fontsize=12)
    ax.set_title('MAE Comparison Across Capacity Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 3: MAPE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(capacities, mape_values, color='#F57C00', alpha=0.8)
    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('MAPE Comparison Across Capacity Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mape_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 4: R² Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(capacities, r2_values, color='#7B1FA2', alpha=0.8)
    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Comparison Across Capacity Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r2_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 5: Sample Size vs Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE vs Sample Size
    ax1.scatter(samples, rmse_values, s=100, color='#1565C0', alpha=0.7)
    for i, cap in enumerate(capacities):
        ax1.annotate(cap, (samples[i], rmse_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Training Samples', fontsize=12)
    ax1.set_ylabel('RMSE (hari)', fontsize=12)
    ax1.set_title('Sample Size vs RMSE', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # R² vs Sample Size
    ax2.scatter(samples, r2_values, s=100, color='#7B1FA2', alpha=0.7)
    for i, cap in enumerate(capacities):
        ax2.annotate(cap, (samples[i], r2_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Training Samples', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Sample Size vs R²', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_size_vs_performance.png"), dpi=150)
    plt.close()
    
    print(f"\nComparison plots saved to: {output_dir}")


if __name__ == "__main__":
    # Generate comparison report
    report = generate_comparison_report()
    
    # Generate comparison plots
    plot_metrics_comparison()
