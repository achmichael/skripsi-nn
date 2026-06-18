import json
import math
import random
import numpy as np
from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork

class PascabayarPlaceValueModel(NeuralNetwork):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 7,
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 0.0,
        component_activation: str = "relu",
    ):
        if input_size <= 0:
            raise ValueError("input_size harus > 0")
        if hidden_size != 7:
            raise ValueError("Eksperimen ini mensyaratkan hidden_size = 7")
        if component_activation not in {"relu", "linear"}:
            raise ValueError("component_activation harus 'relu', atau 'linear'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda
        self.component_activation = component_activation

        if seed is not None:
            np.random.seed(seed)

        # Layer 1: input -> hidden(7)
        self.w1: list[list[float]] = []
        self.b1: list[float] = [0.0] * hidden_size

        std1 = math.sqrt(2.0 / input_size)
        for _ in range(hidden_size):
            row = []
            for _ in range(input_size):
                u1 = random.random() or 1e-10
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                row.append(z * std1)
            self.w1.append(row)

        # Layer 2: hidden(7) -> output(1)
        self.w2: list[float] = []
        self.b2: float = 0.0

        std2 = math.sqrt(2.0 / hidden_size)
        for _ in range(hidden_size):
            u1 = random.random() or 1e-10
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            self.w2.append(z * std2)

        # Cache forward pass
        self._x: list[float] = []
        self._z_hidden: list[float] = []
        self._h: list[float] = []
        self._z_out: float = 0.0
        self._y_pred: float = 0.0

    def _output_activate(self, z: float) -> float:
        if self.component_activation == "relu":
            return relu(z)
        return z

    def _output_activate_derivative(self, z: float) -> float:
        if self.component_activation == "relu":
            return relu_derivative(z)
        return 1.0

    def forward(self, inputs: list[float]) -> float:
        self._x = inputs
        # Hidden layer pertama
        self._z_hidden = []
        self._h = []
        for j in range(self.hidden_size):
            z = self.b1[j]
            for i in range(self.input_size):
                z += self.w1[j][i] * inputs[i]
            self._z_hidden.append(z)
            self._h.append(relu(z))

        # Output layer (1 neuron)
        z_out = self.b2
        for j in range(self.hidden_size):
            z_out += self.w2[j] * self._h[j]
            
        self._z_out = z_out
        self._y_pred = self._output_activate(z_out)
        
        return self._y_pred

    def backward(self, target: float, learning_rate: float) -> None:
        # dMSE/dy_pred = 2 * (y_pred - target)
        grad_y = 2.0 * (self._y_pred - target)
        grad_y = self._clip_gradient(grad_y, self.clip_value)

        # dL/dz_out
        d_loss_d_z_out = grad_y * self._output_activate_derivative(self._z_out)
        d_loss_d_z_out = self._clip_gradient(d_loss_d_z_out, self.clip_value)

        # dL/dh dan update w2, b2
        d_loss_d_h = [0.0] * self.hidden_size
        
        for j in range(self.hidden_size):
            d_loss_d_h[j] = self.w2[j] * d_loss_d_z_out
            
            grad_w2 = d_loss_d_z_out * self._h[j]
            if self.l2_lambda > 0:
                grad_w2 += self.l2_lambda * self.w2[j]
            self.w2[j] -= learning_rate * grad_w2

        self.b2 -= learning_rate * d_loss_d_z_out

        # dL/dz_hidden
        d_loss_d_z_hidden = []
        for j in range(self.hidden_size):
            grad = d_loss_d_h[j] * relu_derivative(self._z_hidden[j])
            d_loss_d_z_hidden.append(self._clip_gradient(grad, self.clip_value))

        # Update w1, b1
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                grad_w1 = d_loss_d_z_hidden[j] * self._x[i]
                if self.l2_lambda > 0:
                    grad_w1 += self.l2_lambda * self.w1[j][i]
                self.w1[j][i] -= learning_rate * grad_w1
            self.b1[j] -= learning_rate * d_loss_d_z_hidden[j]

    def train_one_sample(
        self,
        inputs: list[float],
        target: float,
        learning_rate: float,
    ) -> float:
        y_pred = self.forward(inputs)
        loss = self._mse_loss(y_pred, target)
        self.backward(target, learning_rate)
        return loss

    def train_batch(
        self,
        x_batch,
        y_batch,
        learning_rate: float,
    ) -> float:
        total_loss = 0.0
        n = len(x_batch)
        for i in range(n):
            x_val = x_batch[i].tolist() if isinstance(x_batch[i], np.ndarray) else x_batch[i]
            y_val = float(y_batch[i][0] if getattr(y_batch, "ndim", 1) == 2 else y_batch[i])
            total_loss += self.train_one_sample(x_val, y_val, learning_rate)
        return total_loss / n if n > 0 else 0.0

    def predict(self, inputs):
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            return np.array([self.forward(x.tolist()) for x in inputs])
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], list):
            return np.array([self.forward(x) for x in inputs])
        else:
            if isinstance(inputs, np.ndarray):
                inputs = inputs.tolist()
            return self.forward(inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        data: dict = {
            "model_class": "PascabayarPlaceValueModel",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "clip_value": self.clip_value,
            "l2_lambda": self.l2_lambda,
            "component_activation": self.component_activation,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

        if metadata is not None:
            data["metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> tuple["PascabayarPlaceValueModel", dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(
            input_size=data["input_size"],
            hidden_size=data.get("hidden_size", 7),
            clip_value=data.get("clip_value", 5.0),
            l2_lambda=data.get("l2_lambda", 0.0),
            component_activation=data.get("component_activation", "relu"),
        )

        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]

        metadata: dict = data.get("metadata", {})
        return model, metadata

    def get_summary(self) -> str:
        params = (
            self.input_size * self.hidden_size + self.hidden_size
            + self.hidden_size * 1 + 1
        )
        return (
            "PascabayarPlaceValueModel (Refactored to MLP) | "
            f"Arsitektur: [{self.input_size}, {self.hidden_size}, 1] | "
            f"Aktivasi hidden: ReLU | Aktivasi output: {self.component_activation} | "
            f"Total parameter: {params:,}"
        )
