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
        l2_lambda: float = 0.0,
        asymmetric_alpha: float = 0.5,
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
        self.asymmetric_alpha = asymmetric_alpha
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

        # Adam optimizer parameters & state
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

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
        pass # Only train_batch is used in optimization

    def train_one_sample(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float,
    ) -> float:
        # Konversi ke bentuk batch berukuran 1
        return self.train_batch(inputs[np.newaxis, :], target[np.newaxis, :], learning_rate)

    def train_batch_weighted(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        learning_rate: float,
        weights: list[float],
    ) -> float:
        batch_size = x_batch.shape[0]

        # 1. Forward pass
        prediction = self.forward(x_batch) # shape: (batch_size, 1)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)

        w_np = np.array(weights, dtype=np.float32).reshape(-1, 1)

        diff = prediction - y_batch
        
        # Asymmetric Loss mask
        # Jika diff > 0 (over-estimate), bobot penalti adalah self.asymmetric_alpha
        # Jika diff < 0 (under-estimate), bobot penalti adalah 1.0 - self.asymmetric_alpha
        # Karena sebelumnya MSE standar menggunakan faktor pengali 1, kita men-scale asym_mask dengan x2 
        # (sehingga jika alpha=0.5, multiplier = 1, kembali menjadi MSE standar).
        asym_mask = np.where(diff > 0, self.asymmetric_alpha * 2, (1.0 - self.asymmetric_alpha) * 2)

        squared_errors = asym_mask * (diff ** 2)
        total_loss = float(np.mean(w_np * squared_errors))
        
        # Weighted gradient
        output_grad = w_np * asym_mask * diff / batch_size

        # 2. Backward pass
        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = output_grad

        # Hidden layer deltas — tanpa clipping pada delta
        for l in range(self.num_layers - 3, -1, -1):
            # deltas[l+1] shape: (batch_size, fan_out_next)
            # weights[l+1] shape: (fan_out_next, fan_out_curr)
            grad = np.dot(deltas[l + 1], self.weights[l + 1]) # shape: (batch_size, fan_out_curr)
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = grad

        self.t += 1

        # 3. Update bobot dan bias
        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l] # shape: (batch_size, fan_in)
            
            # gradient weights shape: (fan_out, fan_in)
            grad_w = np.dot(deltas[l].T, inputs_l)
            
            # L2 regularization: Grad += (λ/m)W
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

        # Mengembalikan unweighted loss agar grafik loss konsisten
        unweighted_loss = self._mse_loss(prediction, y_batch)
        return float(unweighted_loss)

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

        diff = prediction - y_batch
        asym_mask = np.where(diff > 0, self.asymmetric_alpha * 2, (1.0 - self.asymmetric_alpha) * 2)

        squared_errors = asym_mask * (diff ** 2)
        total_loss = float(np.mean(squared_errors) / 2.0) # _mse_loss behaviour

        output_grad = asym_mask * diff / batch_size

        # 2. Backward pass
        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = output_grad

        # Hidden layer deltas — tanpa clipping pada delta
        for l in range(self.num_layers - 3, -1, -1):
            grad = np.dot(deltas[l + 1], self.weights[l + 1]) 
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = grad

        self.t += 1

        # 3. Update bobot dan bias
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
