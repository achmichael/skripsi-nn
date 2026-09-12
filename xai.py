import os
import json
import numpy as np

from src.models.prabayar import PrabayarModel
from src.pipeline.preprocessing import (
       load_and_preprocess,
       train_test_split,
       transform_standard_scaler,
       transform_target,
   )
from src.pipeline.feature_extraction import extract_features_and_target
from src.xai.integrated_gradients import IntegratedGradients
from src.xai.visualize import (
       plot_attributions,
       plot_attributions_heatmap,
       plot_comparison_predictions,
   )
from src.config.config import config


def run_explainability_analysis():
    """
    Menjalankan analisis explainability menggunakan Integrated Gradients.
    """
    model_type = "prabayar"
    cfg = config[model_type]

    print("=" * 70)
    print("EXPLAINABLE AI ANALYSIS - INTEGRATED GRADIENTS")
    print("=" * 70)
    # Load trained model
    print(f"\n[1/6] Loading trained model from: {cfg['model_path']}")
    model, metadata = PrabayarModel.load(cfg['model_path'])
    print(f"   ✓ Model loaded successfully")
    print(f"   Architecture: {metadata['layer_sizes']}")
    # Load and preprocess data
    print(f"\n[2/6] Loading dataset: {cfg['dataset_path']}")
    rows, _ = load_and_preprocess(cfg['dataset_path'])
    x_data, y_data, feature_columns, target_column = extract_features_and_target(
        df=rows,
        model_type=model_type,
    )
    print(f"   ✓ Dataset loaded: {len(rows)} samples, {len(feature_columns)} features")
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data=x_data,
        y_data=y_data,
        test_ratio=0.2,
        seed=42,
    )
    # Scale features
    x_scaler = metadata['x_scaler']
    y_scaler = metadata['y_scaler']
    x_train_scaled = transform_standard_scaler(x_train, x_scaler)
    x_test_scaled = transform_standard_scaler(x_test, x_scaler)
    y_test_scaled = transform_target(y_test, y_scaler)
    print(f"   Train: {len(x_train_scaled)}, Test: {len(x_test_scaled)}")
    # Initialize Integrated Gradients
    print(f"\n[3/6] Initializing Integrated Gradients")
    ig = IntegratedGradients(model, baseline_strategy="mean")
    ig.set_baseline(np.asarray(x_train_scaled))
    print(f"   ✓ IG initialized with baseline strategy: zeros")
    # Analyze sample predictions
    print(f"\n[4/6] Computing attributions for test samples")
    num_samples_analyze = min(20, len(x_test_scaled))
    results = []
    for i in range(num_samples_analyze):
        result = ig.explain(np.asarray(x_test_scaled[i]), steps=50)
        results.append(result)
        if i < 3:  # Print detail untuk 3 sample pertama
            print(f"\n   Sample {i+1}:")
            print(f"      Prediction: {result['predictions']:.4f}")
            print(f"      Baseline: {result['baseline_prediction']:.4f}")
            print(f"      Delta: {result['delta']:.4f}")
            top_features = ig.get_top_features(
                result['attributions'],
                feature_columns,
                top_k=5
            )
            print(f"      Top 5 Contributing Features:")
            for feat_name, attr_val in top_features:
                print(f"         {feat_name}: {attr_val:+.6f}")
    print(f"\n   ✓ Computed attributions for {num_samples_analyze} samples")
    # Create output directory
    explainability_dir = "results/prabayar/explainability"
    os.makedirs(explainability_dir, exist_ok=True)
    # Visualizations
    print(f"\n[5/6] Generating visualizations")
    # Plot 1: Average attribution across all samples
    avg_attributions = np.mean([r['attributions'] for r in results], axis=0)
    plot_attributions(
        avg_attributions,
        feature_columns,
        top_k=20,
        save_path=os.path.join(explainability_dir, "average_attributions.png"),
        title="Average Feature Attributions (Integrated Gradients)"
    )
    print(f"   ✓ Average attributions plot saved")
    # Plot 2: Attribution untuk sample dengan error tertinggi
    predictions = [r['predictions'] for r in results]
    errors = [abs(predictions[i] - y_test_scaled[i]) for i in range(num_samples_analyze)]
    max_error_idx = np.argmax(errors)
    plot_attributions(
        results[max_error_idx]['attributions'],
        feature_columns,
        top_k=20,
        save_path=os.path.join(explainability_dir, "worst_prediction_attributions.png"),
        title=f"Attributions for Worst Prediction (Error: {errors[max_error_idx]:.4f})"
    )
    print(f"   ✓ Worst prediction attributions plot saved")
    # Plot 3: Heatmap untuk 10 samples pertama
    plot_attributions_heatmap(
        [r['attributions'] for r in results[:10]],
        feature_columns,
        sample_indices=list(range(10)),
        save_path=os.path.join(explainability_dir, "attribution_heatmap.png"),
        title="Attribution Heatmap (First 10 Samples)"
    )
    print(f"   ✓ Attribution heatmap saved")
    # Plot 4: Comparison predictions vs attributions
    attribution_magnitudes = [np.sum(np.abs(r['attributions'])) for r in results]
    plot_comparison_predictions(
        y_test_scaled[:num_samples_analyze],
        predictions,
        attribution_magnitudes,
        save_path=os.path.join(explainability_dir, "prediction_vs_attribution.png"),
        title="Prediction Quality vs Attribution Magnitude"
    )
    print(f"   ✓ Comparison plot saved")
    # Save detailed results
    print(f"\n[6/6] Saving detailed results")
    # Summary statistics
    summary = {
        "model_type": model_type,
        "num_samples_analyzed": num_samples_analyze,
        "baseline_strategy": "zeros",
        "integration_steps": 50,
        "average_attribution_magnitude": float(np.mean(attribution_magnitudes)),
        "top_10_most_important_features": []
    }
    # Get global feature importance
    abs_avg_attr = np.abs(avg_attributions)
    top_indices = np.argsort(abs_avg_attr)[::-1][:10]
    for idx in top_indices:
        summary["top_10_most_important_features"].append({
            "feature": feature_columns[idx],
            "avg_attribution": float(avg_attributions[idx]),
            "abs_avg_attribution": float(abs_avg_attr[idx])
        })
    summary_path = os.path.join(explainability_dir, "explainability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"   ✓ Summary saved to: {summary_path}")
    # Print summary
    print("\n" + "=" * 70)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nTop 10 Most Important Features (by avg absolute attribution):")
    for i, feat_info in enumerate(summary["top_10_most_important_features"], 1):
        print(f"   {i:2d}. {feat_info['feature']:<40} | {feat_info['abs_avg_attribution']:>10.6f}")
    print(f"\nAll results saved to: {explainability_dir}/")
    print("=" * 70)

if __name__ == "__main__":
       run_explainability_analysis()
