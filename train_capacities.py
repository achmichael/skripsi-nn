"""
Train Capacity-Specific Models

Entry point script untuk melatih model berdasarkan kapasitas daya listrik.
Sesuai dengan revisi examiner untuk memisahkan model per kategori kapasitas.

Usage:
    # Train all capacity models
    python train_capacities.py

    # Train specific capacity
    python train_capacities.py --capacity 900

    # Train multiple capacities
    python train_capacities.py --capacity 450 900 1300

    # With analysis
    python train_capacities.py --with-analysis
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.training.train_by_capacity import train_single_capacity, train_all_capacities
from src.analysis.compare_capacities import generate_comparison_report, plot_metrics_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Train capacity-specific models for token duration prediction"
    )
    parser.add_argument(
        "--capacity",
        nargs="+",
        choices=["450", "900", "1300", "2200", "3500"],
        help="Specific capacity/capacities to train. If not specified, trains all."
    )
    parser.add_argument(
        "--with-analysis",
        action="store_true",
        help="Generate comparative analysis after training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed logs (default: True)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "▓"*70)
    print("  CAPACITY-SPECIFIC MODEL TRAINING")
    print("  Thesis Revision: Separate Models by Electrical Power Capacity")
    print("▓"*70 + "\n")
    
    # Train models
    if args.capacity:
        # Train specific capacities
        print(f"Training models for: {', '.join([c + 'VA' for c in args.capacity])}\n")
        results = {}
        for cap in args.capacity:
            result = train_single_capacity(cap, verbose=args.verbose)
            results[cap] = result
    else:
        # Train all capacities
        print("Training models for all capacities\n")
        results = train_all_capacities(verbose=args.verbose)
    
    # Generate analysis if requested
    if args.with_analysis:
        print("\n" + "▓"*70)
        print("  GENERATING COMPARATIVE ANALYSIS")
        print("▓"*70 + "\n")
        
        try:
            generate_comparison_report()
            plot_metrics_comparison()
            print("\n✓ Comparative analysis completed successfully")
        except Exception as e:
            print(f"\n✗ Failed to generate analysis: {e}")
    
    print("\n" + "▓"*70)
    print("  TRAINING COMPLETE")
    print("▓"*70 + "\n")
    
    # Print model locations
    print("Trained models saved to:")
    for capacity, result in results.items():
        if "error" not in result:
            print(f"  [{capacity}VA] {result.get('model_path', 'N/A')}")
    
    print("\nNext steps:")
    print("  1. Review metrics in results/prabayar_<capacity>/metrics/")
    print("  2. Run comparative analysis: python -m src.analysis.compare_capacities")
    print("  3. Test predictions: python predict_capacity.py")
    print()


if __name__ == "__main__":
    main()
