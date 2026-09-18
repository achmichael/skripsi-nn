import numpy as np
from typing import Callable, Optional
from src.models.neural_network import NeuralNetwork

class IntegratedGradients:

    def __init__(self, model: NeuralNetwork, baseline_strategy: str = "mean", enable_smoothgrad: bool = False, noise_samples: int = 10, noise_scale: float = 0.01):
        """
            Args:
            model: Model objek
            baseline strategy:
                - zeros: baseline all zeros
                - mean: baseline dari mean training data
                - random: baseline dari random noise
            enable_smoothgrad: Jika True, gunakan SmoothGrad untuk mengurangi noise (Smilkov et al., 2017)
            noise_samples: Jumlah samples untuk SmoothGrad
            noise_scale: Standar deviasi noise untuk SmoothGrad
        """
        self.model = model
        self.baseline_strategy = baseline_strategy
        self.baseline = None
        self.enable_smoothgrad = enable_smoothgrad
        self.noise_samples = noise_samples
        self.noise_scale = noise_scale
        self.multiple_baselines = []  # Untuk Expected Gradients

    def set_baseline(self, x_train: np.ndarray | None = None):
        """Set baseline untuk IG computation.

        Untuk regression dengan output yang mungkin memiliki range besar,
        gunakan multiple baselines untuk stabilitas (Expected Gradients approach).
        """
        if self.baseline_strategy == "zeros":
            self.baseline = None
        elif self.baseline_strategy == "mean" and x_train is not None:
            self.baseline = np.mean(x_train, 0)
            # Tambahkan variasi baseline untuk Expected Gradients (Erion et al., 2021)
            # Gunakan persentil untuk capture distribusi data
            self.multiple_baselines = [
                np.mean(x_train, 0),  # mean
                np.percentile(x_train, 25, axis=0),  # Q1
                np.percentile(x_train, 75, axis=0),  # Q3
            ]
        elif self.baseline_strategy == "random":
            self.baseline = None
        # print('baseline results', self.baseline)

    def explain(self, x: np.ndarray, steps: int = 50, baseline: np.ndarray | None = None, use_multiple_baselines: bool = True):
        """Compute Integrated Gradients attributions.

        Args:
            x: Input sample
            steps: Number of integration steps (rekomendasi: 100-300)
            baseline: Custom baseline (optional)
            use_multiple_baselines: Jika True, gunakan Expected Gradients dengan multiple baselines

        Returns:
            Dictionary dengan attributions, predictions, baseline_prediction, delta, dan metrics
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Jika SmoothGrad enabled, compute IG dengan noise sampling
        if self.enable_smoothgrad:
            return self._explain_with_smoothgrad(x, steps, baseline, use_multiple_baselines)

        # Determine baselines to use
        baselines_to_use = []
        if baseline is not None:
            baselines_to_use = [baseline.reshape(1, -1)]
        elif use_multiple_baselines and len(self.multiple_baselines) > 0:
            # Expected Gradients: rata-ratakan attributions dari multiple baselines
            baselines_to_use = [b.reshape(1, -1) for b in self.multiple_baselines]
        else:
            # Single baseline
            if self.baseline is None:
                if self.baseline_strategy == "zeros":
                    baselines_to_use = [np.zeros_like(x)]
                elif self.baseline_strategy == "random":
                    baselines_to_use = [np.random.randn(*x.shape) * 0.01]
                else:
                    baselines_to_use = [np.zeros_like(x)]
            else:
                baselines_to_use = [self.baseline.reshape(1, -1)]

        # Compute attributions for each baseline
        all_attributions = []
        all_deltas = []
        all_baseline_preds = []

        for baseline_i in baselines_to_use:
            result = self._compute_ig_single_baseline(x, baseline_i, steps)
            all_attributions.append(result['attributions'])
            all_deltas.append(result['delta'])
            all_baseline_preds.append(result['baseline_prediction'])

        # Average attributions jika menggunakan multiple baselines
        attributions = np.mean(all_attributions, axis=0)

        # Compute prediction
        prediction = self.model.predict(x)
        avg_baseline_pred = np.mean(all_baseline_preds)
        avg_delta = np.mean(all_deltas)

        # Completeness check
        completeness_error = abs(np.sum(attributions) - avg_delta)

        # Sensitivity check: fitur dengan perbedaan besar harus punya attribution non-zero
        sensitivity_violations = self._check_sensitivity(x, baselines_to_use[0], attributions)

        return {
            "attributions": attributions,
            "predictions": float(np.asarray(prediction).squeeze()),
            "baseline_prediction": float(avg_baseline_pred),
            "delta": float(avg_delta),
            "completeness_error": float(completeness_error),
            "sensitivity_violations": sensitivity_violations,
            "num_baselines_used": len(baselines_to_use),
        }

    def _compute_ig_single_baseline(self, x: np.ndarray, baseline: np.ndarray, steps: int) -> dict:
        """Compute IG untuk single baseline dengan numerical stability."""
        # Integrated gradients computation
        alphas = np.linspace(0, 1, steps + 1)
        gradients = []

        for alpha in alphas:
            # Interpolasi input
            x_interp = baseline + alpha * (x - baseline)

            # Tambahkan small epsilon untuk numerical stability di ReLU dead zones
            # (Ancona et al., 2018)
            epsilon = 1e-7
            x_interp = x_interp + np.random.randn(*x_interp.shape) * epsilon

            # Hitung gradient pada interpolasi input
            try:
                grad = self.model.input_gradients(x_interp)
                # Clip gradient untuk mencegah exploding values
                grad = np.clip(grad, -1e6, 1e6)
                gradients.append(grad)
            except Exception as e:
                # Fallback jika gradient computation gagal
                print(f"Warning: Gradient computation failed at alpha={alpha}: {e}")
                gradients.append(np.zeros_like(x.flatten()))

        # Rata-rata gradient (simple mean, bukan trapezoidal karena sudah steps+1 points)
        # Untuk trapezoidal yang benar, gunakan np.trapz atau weight endpoints
        avg_gradients = np.mean(gradients, 0)

        # Integrated gradients = (x - baseline) * avg_gradients
        attributions = (x.flatten() - baseline.flatten()) * avg_gradients

        # Ambil predictions
        prediction = self.model.predict(x)
        baseline_prediction = self.model.predict(baseline)

        delta = float(np.asarray(prediction).squeeze() - np.asarray(baseline_prediction).squeeze())

        return {
            "attributions": attributions,
            "prediction": float(np.asarray(prediction).squeeze()),
            "baseline_prediction": float(np.asarray(baseline_prediction).squeeze()),
            "delta": delta,
        }

    def _explain_with_smoothgrad(self, x: np.ndarray, steps: int, baseline: Optional[np.ndarray], use_multiple_baselines: bool) -> dict:
        """Compute IG with SmoothGrad (Smilkov et al., 2017).

        SmoothGrad menambahkan Gaussian noise ke input dan rata-ratakan attributions
        untuk mengurangi visual noise dan meningkatkan robustness.
        """
        all_attributions = []

        for _ in range(self.noise_samples):
            # Tambahkan Gaussian noise
            noise = np.random.normal(0, self.noise_scale, x.shape)
            x_noisy = x + noise

            # Compute IG untuk noisy input
            # Disable smoothgrad recursion
            original_smoothgrad = self.enable_smoothgrad
            self.enable_smoothgrad = False

            result = self.explain(x_noisy, steps, baseline, use_multiple_baselines)
            all_attributions.append(result['attributions'])

            self.enable_smoothgrad = original_smoothgrad

        # Average attributions across noise samples
        smoothed_attributions = np.mean(all_attributions, axis=0)

        # Compute final prediction (tanpa noise)
        prediction = self.model.predict(x)

        # Recompute metrics dengan smoothed attributions
        if baseline is None:
            if self.baseline is not None:
                baseline = self.baseline.reshape(1, -1)
            else:
                baseline = np.zeros_like(x)
        else:
            baseline = baseline.reshape(1, -1)

        baseline_prediction = self.model.predict(baseline)
        delta = float(np.asarray(prediction).squeeze() - np.asarray(baseline_prediction).squeeze())
        completeness_error = abs(np.sum(smoothed_attributions) - delta)
        sensitivity_violations = self._check_sensitivity(x, baseline, smoothed_attributions)

        return {
            "attributions": smoothed_attributions,
            "predictions": float(np.asarray(prediction).squeeze()),
            "baseline_prediction": float(np.asarray(baseline_prediction).squeeze()),
            "delta": float(delta),
            "completeness_error": float(completeness_error),
            "sensitivity_violations": sensitivity_violations,
            "num_baselines_used": 1,
            "smoothgrad_samples": self.noise_samples,
        }

    def _check_sensitivity(self, x: np.ndarray, baseline: np.ndarray, attributions: np.ndarray, threshold: float = 1e-3) -> int:
        """Check Sensitivity Axiom (Sundararajan et al., 2017).

        Axiom: Jika feature i berbeda antara input dan baseline, dan menyebabkan
        perbedaan output, maka attribution untuk feature i harus non-zero.

        Returns:
            Number of sensitivity violations detected
        """
        x_flat = x.flatten()
        baseline_flat = baseline.flatten()

        # Cari features dengan perbedaan signifikan
        feature_diffs = np.abs(x_flat - baseline_flat)
        significant_diff_mask = feature_diffs > threshold

        # Check apakah features dengan diff signifikan punya attribution ~0
        violations = 0
        for i in range(len(attributions)):
            if significant_diff_mask[i] and abs(attributions[i]) < 1e-6:
                violations += 1

        return violations

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
