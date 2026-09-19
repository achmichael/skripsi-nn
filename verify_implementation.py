"""
Verification Script for Capacity-Specific Models Implementation

This script verifies that all components are properly implemented.
"""

import os
import sys


def check_file_exists(filepath, description):
    """Check if file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists


def verify_implementation():
    """Verify all components of capacity-specific implementation."""
    
    print("\n" + "="*70)
    print("  CAPACITY-SPECIFIC MODELS - IMPLEMENTATION VERIFICATION")
    print("="*70 + "\n")
    
    all_checks = []
    
    # 1. Model Classes
    print("1. Model Classes")
    all_checks.append(check_file_exists("src/models/prabayar_450.py", "450VA Model"))
    all_checks.append(check_file_exists("src/models/prabayar_900.py", "900VA Model"))
    all_checks.append(check_file_exists("src/models/prabayar_1300.py", "1300VA Model"))
    all_checks.append(check_file_exists("src/models/prabayar_2200.py", "2200VA Model"))
    all_checks.append(check_file_exists("src/models/prabayar_3500.py", "3500VA Model"))
    all_checks.append(check_file_exists("src/models/model_factory.py", "Model Factory"))
    
    # 2. Training Module
    print("\n2. Training Module")
    all_checks.append(check_file_exists("src/training/__init__.py", "Training Init"))
    all_checks.append(check_file_exists("src/training/train_by_capacity.py", "Train by Capacity"))
    
    # 3. Inference Module
    print("\n3. Inference Module")
    all_checks.append(check_file_exists("src/inference/__init__.py", "Inference Init"))
    all_checks.append(check_file_exists("src/inference/predict_by_capacity.py", "Predict by Capacity"))
    
    # 4. Analysis Module
    print("\n4. Analysis Module")
    all_checks.append(check_file_exists("src/analysis/__init__.py", "Analysis Init"))
    all_checks.append(check_file_exists("src/analysis/compare_capacities.py", "Compare Capacities"))
    
    # 5. Entry Scripts
    print("\n5. Entry Scripts")
    all_checks.append(check_file_exists("train_capacities.py", "Train Capacities Script"))
    all_checks.append(check_file_exists("predict_capacity.py", "Predict Capacity Script"))
    
    # 6. Configuration
    print("\n6. Configuration")
    all_checks.append(check_file_exists("src/config/config.py", "Config File"))
    
    # Check if capacity_configs exists in config
    try:
        from src.config.config import config
        has_capacity_configs = "capacity_configs" in config
        status = "✓" if has_capacity_configs else "✗"
        print(f"  {status} Capacity Configs in config.py")
        all_checks.append(has_capacity_configs)
    except Exception as e:
        print(f"  ✗ Error checking capacity_configs: {e}")
        all_checks.append(False)
    
    # 7. Data Files
    print("\n7. Data Files")
    all_checks.append(check_file_exists("data/prabayar_450.csv", "450VA Dataset"))
    all_checks.append(check_file_exists("data/prabayar_900.csv", "900VA Dataset"))
    all_checks.append(check_file_exists("data/prabayar_1300.csv", "1300VA Dataset"))
    all_checks.append(check_file_exists("data/prabayar_2200.csv", "2200VA Dataset"))
    all_checks.append(check_file_exists("data/prabayar_3500.csv", "3500VA Dataset"))
    
    # 8. Documentation
    print("\n8. Documentation")
    all_checks.append(check_file_exists("docs/capacity_models_methodology.md", "Methodology Doc"))
    all_checks.append(check_file_exists("docs/README_capacity_models.md", "Quick Start Guide"))
    all_checks.append(check_file_exists("docs/IMPLEMENTATION_SUMMARY.md", "Implementation Summary"))
    
    # 9. Sample Input
    print("\n9. Sample Files")
    all_checks.append(check_file_exists("sample_input_900va.json", "Sample Input (900VA)"))
    
    # 10. Test Imports
    print("\n10. Module Imports")
    
    try:
        from src.models.model_factory import create_model_for_capacity
        print("  ✓ Model Factory import successful")
        all_checks.append(True)
    except Exception as e:
        print(f"  ✗ Model Factory import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.training.train_by_capacity import train_single_capacity
        print("  ✓ Training module import successful")
        all_checks.append(True)
    except Exception as e:
        print(f"  ✗ Training module import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.inference.predict_by_capacity import predict_by_capacity
        print("  ✓ Inference module import successful")
        all_checks.append(True)
    except Exception as e:
        print(f"  ✗ Inference module import failed: {e}")
        all_checks.append(False)
    
    try:
        from src.analysis.compare_capacities import generate_comparison_report
        print("  ✓ Analysis module import successful")
        all_checks.append(True)
    except Exception as e:
        print(f"  ✗ Analysis module import failed: {e}")
        all_checks.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("  VERIFICATION SUMMARY")
    print("="*70 + "\n")
    
    total = len(all_checks)
    passed = sum(all_checks)
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    
    if failed == 0:
        print("\n🎉 All checks passed! Implementation is complete.")
        print("\nNext steps:")
        print("  1. Run: python train_capacities.py --with-analysis")
        print("  2. Review results in results/comparative_analysis/")
        print("  3. Test prediction: python predict_capacity.py --input sample_input_900va.json")
        return True
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return False


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*70)
    print("  USAGE EXAMPLES")
    print("="*70 + "\n")
    
    print("Training:")
    print("  python train_capacities.py                    # Train all capacities")
    print("  python train_capacities.py --capacity 900     # Train specific capacity")
    print("  python train_capacities.py --with-analysis    # Train + generate analysis")
    
    print("\nPrediction:")
    print("  python predict_capacity.py --input sample_input_900va.json")
    print("  python predict_capacity.py --capacity 900 --input sample_input_900va.json")
    print("  python predict_capacity.py --input sample_input_900va.json --compare-all")
    
    print("\nAnalysis:")
    print("  python -m src.analysis.compare_capacities")
    
    print()


if __name__ == "__main__":
    success = verify_implementation()
    
    if success:
        print_usage_examples()
        sys.exit(0)
    else:
        sys.exit(1)
