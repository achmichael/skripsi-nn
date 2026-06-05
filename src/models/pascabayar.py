import json
import math
import random

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork


class PascabayarModel(NeuralNetwork):
    def __init__(
        self,
        layer_sizes: list[int],
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 0.0,
    ):
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes minimal harus memiliki 2 elemen (input dan output)."
            )
        if layer_sizes[-1] != 1:
            raise ValueError(
                f"Output layer harus berukuran 1 untuk regresi, "
                f"tetapi mendapat {layer_sizes[-1]}."
            )

        self.layer_sizes = layer_sizes
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_sizes)

        if seed is not None:
            random.seed(seed)

        # Inisialisasi bobot dan bias untuk setiap layer (l -> l+1)
        # weights[l][j][i] = bobot dari neuron i di layer l ke neuron j di layer l+1
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        for l in range(self.num_layers - 1):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            std = math.sqrt(2.0 / fan_in)  # He initialization

            layer_weights = []
            for j in range(fan_out):
                neuron_weights = []
                for i in range(fan_in):
                    # this is for random initialization and prevent zero weights
                    u1 = random.random() or 1e-10
                    u2 = random.random()
                    # this is for normal distribution (box muller transform)
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    neuron_weights.append(z * std)
                layer_weights.append(neuron_weights)

            self.weights.append(layer_weights)
            self.biases.append([0.0] * fan_out)

        # Cache untuk forward/backward pass
        self._activations: list[list[float]] = []   # output tiap layer (setelah aktivasi)
        self._pre_activations: list[list[float]] = []  # nilai sebelum aktivasi (z)

    def forward(self, inputs: list[float]) -> float:
        """
        Propagasi maju melewati semua layer.

        Hidden layers menggunakan ReLU, output layer menggunakan linear.

        Args:
            inputs: Vektor fitur input (sudah dinormalisasi).

        Returns:
            Nilai prediksi output (float, skala normalisasi).
        """
        self._activations = [inputs]
        self._pre_activations = []

        current = inputs

        for l in range(self.num_layers - 1):
            is_output_layer = (l == self.num_layers - 2)
            fan_out = self.layer_sizes[l + 1]
            z_layer = []
            a_layer = []

            for j in range(fan_out):
                z = self.biases[l][j]
                for i, x in enumerate(current):
                    z += self.weights[l][j][i] * x

                z_layer.append(z)
                # Hidden layers: ReLU | Output layer: linear
                a_layer.append(z if is_output_layer else relu(z))

            self._pre_activations.append(z_layer)
            self._activations.append(a_layer)
            current = a_layer

        return current[0]

    def backward(self, target: float, learning_rate: float) -> None:
        """
        Backpropagation dengan gradient clipping.

        Args:
            target       : Nilai target aktual (skala normalisasi).
            learning_rate: Laju pembelajaran.
        """
        prediction = self._activations[-1][0]

        # Gradient loss terhadap output: d(MSE)/d(output) = 2*(pred - target)
        output_grad = 2.0 * (prediction - target)

        # Gradients terhadap output dari tiap layer (delta)
        # Mulai dari output layer ke arah input
        deltas: list[list[float]] = [None] * (self.num_layers - 1)  # type: ignore

        # Output layer (linear, tidak ada ReLU)
        deltas[-1] = [self._clip_gradient(output_grad, self.clip_value)]

        # Hidden layers (dari kanan ke kiri)
        for l in range(self.num_layers - 3, -1, -1):
            fan_out_next = self.layer_sizes[l + 2]
            fan_out_curr = self.layer_sizes[l + 1]
            delta_curr = []

            for i in range(fan_out_curr):
                grad = 0.0
                for j in range(fan_out_next):
                    grad += self.weights[l + 1][j][i] * deltas[l + 1][j]

                grad *= relu_derivative(self._pre_activations[l][i])
                delta_curr.append(self._clip_gradient(grad, self.clip_value))

            deltas[l] = delta_curr

        # Update bobot dan bias (dengan L2 weight decay)
        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l]
            for j in range(self.layer_sizes[l + 1]):
                for i in range(self.layer_sizes[l]):
                    grad = deltas[l][j] * inputs_l[i]
                    # L2 regularization: tambah weight decay
                    if self.l2_lambda > 0:
                        grad += self.l2_lambda * self.weights[l][j][i]
                    self.weights[l][j][i] -= learning_rate * grad
                self.biases[l][j] -= learning_rate * deltas[l][j]

    def train_one_sample(
        self,
        inputs: list[float],
        target: float,
        learning_rate: float,
    ) -> float:
        """
        Satu langkah pelatihan: forward → loss → backward.

        Args:
            inputs       : Vektor fitur input (sudah dinormalisasi).
            target       : Nilai target aktual (sudah dinormalisasi).
            learning_rate: Laju pembelajaran.

        Returns:
            MSE loss untuk sampel ini.
        """
        prediction = self.forward(inputs)
        loss = self._mse_loss(prediction, target)
        self.backward(target, learning_rate)
        return loss

    def predict(self, inputs: list[float]) -> float:
        """
        Inferensi tanpa memperbarui bobot.

        Args:
            inputs: Vektor fitur input (sudah dinormalisasi).

        Returns:
            Nilai prediksi (float, skala normalisasi).
        """
        return self.forward(inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        """
        Simpan bobot model dan metadata ke file JSON.

        Args:
            path    : Path file JSON tujuan.
            metadata: Dict tambahan (feature_columns, scaler, layer_sizes, dll).
        """
        data: dict = {
            "model_class": "PascabayarModel",
            "layer_sizes": self.layer_sizes,
            "clip_value": self.clip_value,
            "weights": self.weights,
            "biases": self.biases,
        }

        if metadata is not None:
            data["metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> tuple["PascabayarModel", dict]:
        """
        Muat model dari file JSON.

        Args:
            path: Path file JSON sumber.

        Returns:
            Tuple (model, metadata).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_sizes: list[int] = data["layer_sizes"]
        clip_value: float = data.get("clip_value", 5.0)

        model = cls(layer_sizes=layer_sizes, clip_value=clip_value)
        model.weights = data["weights"]
        model.biases = data["biases"]

        metadata: dict = data.get("metadata", {})
        return model, metadata

    # ------------------------------------------------------------------
    # Override get_summary
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Ringkasan arsitektur model."""
        total_params = sum(
            self.layer_sizes[l] * self.layer_sizes[l + 1] + self.layer_sizes[l + 1]
            for l in range(self.num_layers - 1)
        )
        return (
            f"PascabayarModel | Arsitektur: {self.layer_sizes} | "
            f"Total parameter: {total_params:,} | "
            f"Clip value: {self.clip_value}"
        )