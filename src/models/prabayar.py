import json
import math
import numpy as np

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork


class PrabayarModel(NeuralNetwork):
    def __init__(
        self,
        layer_sizes: list[int],
        seed: int | None = None,
        clip_value: float = 5.0,
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
        self.num_layers = len(layer_sizes)

        if seed is not None:
            np.random.seed(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for l in range(self.num_layers - 1):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            std = math.sqrt(2.0 / fan_in)  # He initialization

            W = np.random.randn(fan_out, fan_in) * std
            b = np.zeros(fan_out)

            self.weights.append(W)
            self.biases.append(b)

        # Cache untuk forward/backward pass (list of arrays)
        self._activations: list[np.ndarray] = []      
        self._pre_activations: list[np.ndarray] = []  

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._activations = [inputs]
        self._pre_activations = []

        current = inputs

        for l in range(self.num_layers - 1):
            is_output_layer = (l == self.num_layers - 2)
            
            # current shape: (batch_size, fan_in) or (fan_in,)
            # W shape: (fan_out, fan_in)
            # W.T shape: (fan_in, fan_out)
            # z shape: (batch_size, fan_out)
            z = np.dot(current, self.weights[l].T) + self.biases[l]
            
            self._pre_activations.append(z)
            
            a = z if is_output_layer else relu(z)
            self._activations.append(a)
            current = a

        return current

    def backward(self, target: np.ndarray, learning_rate: float) -> None:
        pass # Only train_batch is used in optimization

    def train_one_sample(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float,
    ) -> float:
        # Konversi ke bentuk batch berukuran 1
        return self.train_batch(inputs[np.newaxis, :], target[np.newaxis, :], learning_rate)

    def train_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        learning_rate: float,
    ) -> float:
        batch_size = x_batch.shape[0]

        # 1. Forward pass
        prediction = self.forward(x_batch) # shape: (batch_size, 1)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)

        total_loss = self._mse_loss(prediction, y_batch)

        # 2. Backward pass
        # Output layer gradient
        output_grad = 2.0 * (prediction - y_batch) / batch_size # shape: (batch_size, 1)
        output_grad = self._clip_gradient(output_grad, self.clip_value)

        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = output_grad

        # Hidden layers gradient
        for l in range(self.num_layers - 3, -1, -1):
            # deltas[l+1] shape: (batch_size, fan_out_next)
            # weights[l+1] shape: (fan_out_next, fan_out_curr)
            grad = np.dot(deltas[l + 1], self.weights[l + 1]) # shape: (batch_size, fan_out_curr)
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = self._clip_gradient(grad, self.clip_value)

        # 3. Update bobot dan bias
        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l] # shape: (batch_size, fan_in)
            
            # gradient weights shape: (fan_out, fan_in)
            grad_w = np.dot(deltas[l].T, inputs_l)
            grad_b = np.sum(deltas[l], axis=0)

            self.weights[l] -= learning_rate * grad_w
            self.biases[l] -= learning_rate * grad_b

        return float(total_loss)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        data: dict = {
            "model_class": "PrabayarModel",
            "layer_sizes": self.layer_sizes,
            "clip_value": self.clip_value,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

        if metadata is not None:
            data["metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> tuple["PrabayarModel", dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_sizes: list[int] = data["layer_sizes"]
        clip_value: float = data.get("clip_value", 5.0)

        model = cls(layer_sizes=layer_sizes, clip_value=clip_value)
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
            f"PrabayarModel (Vectorized) | Arsitektur: {self.layer_sizes} | "
            f"Total parameter: {total_params:,} | "
            f"Clip value: {self.clip_value}"
        )
