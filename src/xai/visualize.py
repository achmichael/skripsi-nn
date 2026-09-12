import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.cm as cm

def plot_attributions(attributions: np.ndarray, feature_names: list[str], top_k: int = 15, save_path: Optional[str] = None, title: str = "Feature Attributions (Integrated Gradients)"):
    abs_attr = np.abs(attributions)
    top_indices = np.argsort(abs_attr)[::-1][:top_k]

    top_features = [feature_names[idx] for idx in top_indices]

    top_values = [attributions[idx] for idx in top_indices]

    colors = ['#4CAF50' if val >= 0 else '#F44336' for val in top_values]
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attribution plot saved to: {save_path}")
        plt.close()

def plot_attributions_heatmap(attributions_batch: list[np.ndarray], feature_names: list[str], sample_indices: Optional[list[int]] = None, save_path: Optional[str] = None, title: str = "Attributions heatmap"):
    attr_matrix = np.vstack(attributions_batch)
    if sample_indices is None:
        sample_indices = list(range(len(attributions_batch)))

    fig, ax = plt.subplots(figsize=(14, max(8, len(sample_indices) * 0.3)))

    # Normalize untuk colormap
    vmax = np.abs(attr_matrix).max()
    im = ax.imshow(
               attr_matrix,
               cmap='RdBu_r',
               aspect='auto',
               vmin=-vmax,
               vmax=vmax
           )

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(sample_indices)))
    ax.set_yticklabels([f"Sample {i}" for i in sample_indices], fontsize=9)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Features", fontsize=11)
    ax.set_ylabel("Samples", fontsize=11)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attribution Score", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    plt.close()

def plot_comparison_predictions(
       actual_values: list[float],
       predicted_values: list[float],
       attribution_scores: list[float],
       save_path: Optional[str] = None,
       title: str = "Prediction vs Attribution Magnitude"
   ):
       """
       Plot comparison antara actual, predicted, dan attribution magnitude.

       Args:
           actual_values: Actual target values
           predicted_values: Predicted values
           attribution_scores: Total attribution scores (sum of absolute)
           save_path: Path untuk menyimpan plot
           title: Judul plot
       """
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

       # Plot 1: Actual vs Predicted
       ax1.scatter(actual_values, predicted_values, alpha=0.6, s=50)

       min_val = min(min(actual_values), min(predicted_values))
       max_val = max(max(actual_values), max(predicted_values))
       ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)

       ax1.set_xlabel("Actual Values", fontsize=11)
       ax1.set_ylabel("Predicted Values", fontsize=11)
       ax1.set_title("Actual vs Predicted", fontsize=12, fontweight='bold')
       ax1.grid(alpha=0.3)

       # Plot 2: Attribution Magnitude vs Prediction Error
       errors = [abs(p - a) for p, a in zip(predicted_values, actual_values)]

       ax2.scatter(attribution_scores, errors, alpha=0.6, s=50, color='orange')
       ax2.set_xlabel("Total Attribution Magnitude", fontsize=11)
       ax2.set_ylabel("Absolute Prediction Error", fontsize=11)
       ax2.set_title("Attribution vs Error", fontsize=12, fontweight='bold')
       ax2.grid(alpha=0.3)

       plt.suptitle(title, fontsize=14, fontweight='bold')
       plt.tight_layout()

       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches='tight')
           print(f"Comparison plot saved to: {save_path}")

       plt.close()


