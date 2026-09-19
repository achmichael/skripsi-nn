"""
Prabayar Model for 3500VA Capacity
Specialized neural network for predicting token duration for 3500VA electrical capacity.

NOTE: This capacity has very limited data (11 samples), so this model may have limited accuracy.
Consider using Prabayar2200Model as fallback or merging with 2200VA data.
"""

import json
import numpy as np
from src.models.prabayar import PrabayarModel


class Prabayar3500Model(PrabayarModel):
    """
    Neural Network Model optimized for 3500VA capacity households.
    
    WARNING: This model is trained on only 11 samples.
    Architectural considerations:
    - Very simple architecture to avoid overfitting
    - High regularization
    - Should be used with caution or fallback to 2200VA model
    """
    
    def __init__(
        self,
        layer_sizes: list[int],
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 0.0,
        l1_lambda_input: float = 0.0,
        asymmetric_alpha: float = 0.5,
    ):
        super().__init__(
            layer_sizes=layer_sizes,
            seed=seed,
            clip_value=clip_value,
            l2_lambda=l2_lambda,
            l1_lambda_input=l1_lambda_input,
            asymmetric_alpha=asymmetric_alpha,
        )
        self.capacity = "3500VA"
    
    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save model with capacity identifier."""
        data: dict = {
            "model_class": "Prabayar3500Model",
            "capacity": "3500VA",
            "layer_sizes": self.layer_sizes,
            "clip_value": self.clip_value,
            "l2_lambda": self.l2_lambda,
            "l1_lambda_input": self.l1_lambda_input,
            "asymmetric_alpha": self.asymmetric_alpha,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

        if metadata is not None:
            data["metadata"] = metadata
            # Add warning about limited data
            data["metadata"]["warning"] = "Model trained on only 11 samples. Use with caution."

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> tuple["Prabayar3500Model", dict]:
        """Load 3500VA specific model."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_sizes: list[int] = data["layer_sizes"]
        clip_value: float = data.get("clip_value", 5.0)
        l2_lambda: float = data.get("l2_lambda", 0.0)
        l1_lambda_input: float = data.get("l1_lambda_input", 0.0)
        asymmetric_alpha: float = data.get("asymmetric_alpha", 0.5)

        model = cls(
            layer_sizes=layer_sizes,
            clip_value=clip_value,
            l2_lambda=l2_lambda,
            l1_lambda_input=l1_lambda_input,
            asymmetric_alpha=asymmetric_alpha,
        )
        model.weights = [np.array(w, dtype=np.float32) for w in data["weights"]]
        model.biases = [np.array(b, dtype=np.float32) for b in data["biases"]]

        metadata: dict = data.get("metadata", {})
        return model, metadata
    
    def get_summary(self) -> str:
        total_params = sum(
            self.layer_sizes[l] * self.layer_sizes[l + 1] + self.layer_sizes[l + 1]
            for l in range(self.num_layers - 1)
        )
        return (
            f"Prabayar3500Model (3500VA Capacity - LIMITED DATA) | Arsitektur: {self.layer_sizes} | "
            f"Total parameter: {total_params:,} | "
            f"Clip value: {self.clip_value}"
        )
