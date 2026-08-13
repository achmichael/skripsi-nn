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

        # Adam optimizer parameters & state
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._activations = [inputs]
        self._pre_activations = []

        current = inputs

        for l in range(self.num_layers - 1):
            is_output_layer = (l == self.num_layers - 2)
            
            if is_output_layer:
                z = np.dot(current, self.weights[l].T)
            else:
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

        # δ_out = (1/m)(ŷ − y) — tanpa faktor 2, konsisten dengan L = (1/(2m))Σ(ŷ−y)²
        output_grad = (prediction - y_batch) / batch_size

        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = output_grad

        # Hidden layer deltas — tanpa clipping pada delta
        for l in range(self.num_layers - 3, -1, -1):
            grad = np.dot(deltas[l + 1], self.weights[l + 1])
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = grad

        self.t += 1

        # Update weight dan bias
        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l]
            
            grad_w = np.dot(deltas[l].T, inputs_l)

            # L2 regularization: Grad += (λ/m)W — setelah gradien dihitung
            if self.l2_lambda > 0:
                grad_w += (self.l2_lambda / batch_size) * self.weights[l]

            # Gradient clipping — setelah L2, pada gradien weight
            grad_w = self._clip_gradient(grad_w, self.clip_value)
                
            # Adam update for weights
            self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * grad_w
            self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (grad_w ** 2)
            m_w_hat = self.m_w[l] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[l] / (1 - self.beta2 ** self.t)

            self.weights[l] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            # Bias update — skip output layer (output layer tidak punya bias)
            if l < self.num_layers - 2:
                grad_b = np.sum(deltas[l], axis=0)
                grad_b = self._clip_gradient(grad_b, self.clip_value)
                
                # Adam update for biases
                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grad_b
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grad_b ** 2)
                m_b_hat = self.m_b[l] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[l] / (1 - self.beta2 ** self.t)

                self.biases[l] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

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