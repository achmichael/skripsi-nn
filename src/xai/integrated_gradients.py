import numpy as np
from typing import Callable
from src.models.neural_network import NeuralNetwork

class IntegratedGradients:

    def __init__(self, model: NeuralNetwork, baseline_strategy: str = "zeros"):
        """
            Args:
            model: Model objek
            baseline strategy:
                - zeros: baseline all zeros
                - mean: baseline dari mean training data
                - random: baseline dari random noise
        """
        self.model = model
        self.baseline_strategy = baseline_strategy
        self.baseline = None

    def set_baseline(self, x_train: np.ndarray | None = None):
        if self.baseline_strategy == "zeros":
            self.baseline = None
        elif self.baseline_strategy == "mean" and x_train is not None:
            self.baseline = np.mean(x_train, 0)
        elif self.baseline_strategy == "random":
            self.baseline = None

    def explain(self, x: np.ndarray, steps: int = 50, baseline: np.ndarray | None = None):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if baseline is None:
            if self.baseline is None:
                if self.baseline_strategy  == "zeros":
                    baseline = np.zeros_like(x)
                elif self.baseline_strategy == "random":
                    baseline = np.random.randn(*x.shape) * 0.01
                else:
                    baseline = np.zeros_like(x)
            else:
                baseline = self.baseline.reshape(1, -1)
        else:
            baseline = baseline.reshape(1, -1)

        # integrated gradients computation
        alphas = np.linspace(0, 1, steps + 1)
        gradients = []

        for alpha in alphas:
            # interpolasi input
            x_interp = baseline + alpha * (x - baseline)
            # hitung gradient pada interpolasi input
            grad = self.model.input_gradients(x_interp)
            gradients.append(grad)

        # rata rata gradient menggunakan aturan trapezoidal
        avg_gradients = np.mean(gradients, 0)

        # integrated gradients = (x - baseline) * avg_gradients
        attributions = (x.flatten() - baseline.flatten()) * avg_gradients

        # ambil predictions
        prediction = self.model.predict(x)
        baseline_prediction = self.model.predict(baseline)

        return {
            "attributions": attributions,
            "predictions": float(np.asarray(prediction).squeeze()),
            "baseline_prediction": float(np.asarray(baseline_prediction).squeeze()),
            "delta": float(
                np.asarray(prediction).squeeze() -
                np.asarray(baseline_prediction).squeeze()
            )
        }

    def explain_batch(self, x_batch: np.ndarray, steps: int = 50) -> list[dict]:
        results = []
        for i in range(x_batch.shape[0]):
            result = self.explain(x_batch[i], steps)
            results.append(result)

        return results

    def get_top_features(self, attributions: np.ndarray, feature_names: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        abs_attr = np.abs(attributions)
        top_indices = np.argsort(abs_attr)[::-1][:top_k]

        top_features = [
            (feature_names[idx], attributions[idx])
            for idx in top_indices
        ]
        return top_features
