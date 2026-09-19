"""
Predict using Capacity-Specific Models

Script untuk melakukan prediksi menggunakan model spesifik berdasarkan kapasitas.

Usage:
    # Auto-detect capacity from input
    python predict_capacity.py --input sample_input.json

    # Specify capacity explicitly
    python predict_capacity.py --capacity 900 --input sample_input.json

    # Compare predictions from all models
    python predict_capacity.py --input sample_input.json --compare-all
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.predict_by_capacity import (
    predict_by_capacity,
    predict_with_all_models,
    get_model_recommendation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Predict token duration using capacity-specific models"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with preprocessed features"
    )
    parser.add_argument(
        "--capacity",
        type=str,
        choices=["450", "900", "1300", "2200", "3500"],
        help="Specific capacity model to use. If not specified, auto-detects from input."
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare predictions from all capacity models"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save prediction results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    print("\n" + "="*70)
    print("  CAPACITY-SPECIFIC MODEL PREDICTION")
    print("="*70 + "\n")
    
    # Get capacity info
    capacity_value = input_data.get("Daya_Listrik_Rumah_VA", "Unknown")
    print(f"Input Capacity: {capacity_value} VA")
    
    if args.compare_all:
        # Predict with all models
        print("\nPredicting with all capacity models...\n")
        results = predict_with_all_models(input_data)
        
        print(f"{'Capacity':<12} {'Prediction (hari)':<20} {'Status':<15}")
        print("-"*70)
        
        for capacity in sorted(results.keys(), key=lambda x: int(x)):
            result = results[capacity]
            if "error" in result:
                print(f"{capacity}VA{'':<8} {'N/A':<20} Error")
            else:
                pred = result["prediction"]
                print(f"{capacity}VA{'':<8} {pred:<20.2f} ✓")
        
        output_data = {
            "input_capacity": capacity_value,
            "predictions": {
                f"{k}VA": v for k, v in results.items()
            }
        }
        
    else:
        # Single prediction
        use_auto_routing = args.capacity is None
        
        if use_auto_routing:
            recommendation = get_model_recommendation(
                capacity_value if isinstance(capacity_value, int) else 900
            )
            print(f"Recommended Model: {recommendation['recommended_model']}")
            print(f"Reason: {recommendation['reason']}\n")
        
        result = predict_by_capacity(
            input_data=input_data,
            capacity=args.capacity,
            use_auto_routing=use_auto_routing
        )
        
        print(f"Model Used: {result['capacity_used']}")
        print(f"Model Path: {result['model_path']}")
        print(f"\nPrediction: {result['prediction']:.2f} hari")
        
        output_data = {
            "input_capacity": capacity_value,
            "model_used": result['capacity_used'],
            "prediction": result['prediction'],
            "metadata": result['metadata'],
        }
    
    # Save output if specified
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
