import json
import math
import numpy as np

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork


class PascabayarModel(NeuralNetwork):
    def __init__(
        self,
        layer_sizes: list[int],
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 1e-4,
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
            np.random.seed(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for l in range(self.num_layers - 1):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            std = math.sqrt(2.0 / fan_in)

            W = np.random.randn(fan_out, fan_in) * std
            b = np.zeros(fan_out)

            self.weights.append(W)
            self.biases.append(b)

        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._activations = [inputs]
        self._pre_activations = []

        current = inputs

        for l in range(self.num_layers - 1):
            is_output_layer = (l == self.num_layers - 2)
            
            z = np.dot(current, self.weights[l].T) + self.biases[l]
            self._pre_activations.append(z)
            
            a = z if is_output_layer else relu(z)
            self._activations.append(a)
            current = a

        return current

    def backward(self, target: np.ndarray, learning_rate: float) -> None:
        pass

    def train_one_sample(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float,
    ) -> float:
        return self.train_batch(inputs[np.newaxis, :], target[np.newaxis, :], learning_rate)

    def train_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        learning_rate: float,
    ) -> float:
        batch_size = x_batch.shape[0]

        prediction = self.forward(x_batch)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)

        total_loss = self._mse_loss(prediction, y_batch)

        output_grad = 2.0 * (prediction - y_batch) / batch_size
        output_grad = self._clip_gradient(output_grad, self.clip_value)

        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = output_grad

        for l in range(self.num_layers - 3, -1, -1):
            grad = np.dot(deltas[l + 1], self.weights[l + 1])
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = self._clip_gradient(grad, self.clip_value)

        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l]
            
            grad_w = np.dot(deltas[l].T, inputs_l)
            if self.l2_lambda > 0:
                grad_w += self.l2_lambda * self.weights[l]
                
            grad_b = np.sum(deltas[l], axis=0)

            self.weights[l] -= learning_rate * grad_w
            self.biases[l] -= learning_rate * grad_b

        return float(total_loss)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        data: dict = {
            "model_class": "PascabayarModel",
            "layer_sizes": self.layer_sizes,
            "clip_value": self.clip_value,
            "l2_lambda": self.l2_lambda,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

        if metadata is not None:
            data["metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> tuple["PascabayarModel", dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_sizes: list[int] = data["layer_sizes"]
        clip_value: float = data.get("clip_value", 5.0)
        l2_lambda: float = data.get("l2_lambda", 1e-4)

        model = cls(layer_sizes=layer_sizes, clip_value=clip_value, l2_lambda=l2_lambda)
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
            f"PascabayarModel (Vectorized) | Arsitektur: {self.layer_sizes} | "
            f"Total parameter: {total_params:,} | "
            f"Clip value: {self.clip_value} | L2: {self.l2_lambda}"
        )