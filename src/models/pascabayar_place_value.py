"""
PascabayarPlaceValueModel

Eksperimen arsitektur Place Value Decomposition Neural Network.

Struktur:
- Input layer: jumlah fitur final setelah preprocessing
- Hidden layer: 7 neuron, aktivasi ReLU
- Output komponen: 7 neuron (komponen kontribusi nilai tempat)
- Final output: penjumlahan seluruh komponen

Catatan metodologis:
Model ini eksperimen arsitektur untuk memecah estimasi biaya menjadi 7 komponen
nilai tempat yang dijumlahkan menjadi prediksi total.
Model dibatasi skala maksimum tiap komponen agar interpretasi place-value lebih konsisten.
Tetap tidak mengklaim neuron pasti belajar digit satuan-jutaan secara sempurna.
"""

import json
import math
import random

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork


class PascabayarPlaceValueModel(NeuralNetwork):
    """
    Model NN dengan hidden=7 dan output komponen=7.

    Forward:
      h = ReLU(XW1 + b1)
      c = activation(W2h + b2)
      y_pred = sum(c)

    Loss dihitung terhadap y_pred (bukan loss per-komponen terpisah).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 7,
        num_components: int = 7,
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 0.0,
        component_activation: str = "relu",
    ):
        if input_size <= 0:
            raise ValueError("input_size harus > 0")
        if hidden_size != 7:
            raise ValueError("Eksperimen ini mensyaratkan hidden_size = 7")
        if num_components != 7:
            raise ValueError("Eksperimen ini mensyaratkan num_components = 7")
        if component_activation not in {"relu", "linear"}:
            raise ValueError("component_activation harus 'relu', atau 'linear'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_components = num_components
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda
        self.component_activation = component_activation

        if seed is not None:
            random.seed(seed)

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

        # Layer 2: hidden -> 7 komponen
        self.w2: list[list[float]] = []
        self.b2: list[float] = [0.0] * num_components

        std2 = math.sqrt(2.0 / hidden_size)
        for _ in range(num_components):
            row = []
            for _ in range(hidden_size):
                u1 = random.random() or 1e-10
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                row.append(z * std2)
            self.w2.append(row)

        # Batas skala komponen pada ruang target normalisasi.
        # Jika target dinormalisasi y_norm = y_rp / 1_000_000, maka batas jadi:
        # [9, 90, 900, 9000, 90000, 900000, 1000000] / 1_000_000
        self.component_max_values: list[float] = [
            9.0 / 1_000_000.0,
            90.0 / 1_000_000.0,
            900.0 / 1_000_000.0,
            9_000.0 / 1_000_000.0,
            90_000.0 / 1_000_000.0,
            900_000.0 / 1_000_000.0,
            1_000_000.0 / 1_000_000.0,
        ]

        # Aturan konsep digit (rupiah):
        # satuan, puluhan, ratusan, ribuan, puluhan ribu, ratusan ribu, jutaan.
        self.component_places_rp: list[int] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
        self.component_max_digits: list[int] = [9, 9, 9, 9, 9, 9, 1]
        self.max_representable_rp: int = sum(
            d * p for d, p in zip(self.component_max_digits, self.component_places_rp)
        )

        # Cache forward pass
        self._x: list[float] = []
        self._z_hidden: list[float] = []
        self._h: list[float] = []
        self._z_comp: list[float] = []
        self._sigmoid_comp: list[float] = []
        self._c: list[float] = []
        self._y_pred: float = 0.0

    def _component_activate(self, z: float, idx: int) -> tuple[float, float]:
        """
        Aktivasi komponen output.

        Return:
        - out: output komponen
        - aux: nilai bantu untuk turunan (sigmoid output jika scaled_sigmoid)
        """
        if self.component_activation == "relu":
            return relu(z), 0.0

        return z, 0.0

    def _component_activate_derivative(self, z: float, idx: int, aux: float) -> float:
        if self.component_activation == "relu":
            return relu_derivative(z)

        return 1.0

    def forward(self, inputs: list[float]) -> float:
        self._x = inputs

        # Hidden layer
        self._z_hidden = []
        self._h = []
        for j in range(self.hidden_size):
            z = self.b1[j]
            for i in range(self.input_size):
                z += self.w1[j][i] * inputs[i]
            self._z_hidden.append(z)
            self._h.append(relu(z))

        # Komponen output
        self._z_comp = []
        self._sigmoid_comp = []
        self._c = []
        for k in range(self.num_components):
            z = self.b2[k]
            for j in range(self.hidden_size):
                z += self.w2[k][j] * self._h[j]
            self._z_comp.append(z)
            c_k, aux = self._component_activate(z, k)
            self._c.append(c_k)
            self._sigmoid_comp.append(aux)

        # Prediksi final = jumlah komponen
        self._y_pred = sum(self._c)
        return self._y_pred

    @staticmethod
    def _normalized_to_rp_int(value_normalized: float) -> int:
        """Konversi ruang normalisasi (dibagi 1e6) ke integer rupiah."""
        return int(round(value_normalized * 1_000_000.0))

    @staticmethod
    def _components_rp_to_normalized(components_rp: list[int]) -> list[float]:
        """Konversi komponen rupiah -> ruang normalisasi (dibagi 1e6)."""
        return [v / 1_000_000.0 for v in components_rp]

    def _decompose_rp_to_digit_components(self, amount_rp: int) -> list[int]:
        """
        Dekomposisi integer rupiah ke komponen nilai tempat berbasis digit.

        Aturan:
        - satuan: 0..9
        - puluhan: 0..90
        - ratusan: 0..900
        - ribuan: 0..9000
        - puluhan ribu: 0..90000
        - ratusan ribu: 0..900000
        - jutaan: 0 atau 1_000_000
        """
        clipped = max(0, min(self.max_representable_rp, amount_rp))

        return [
            (clipped % 10),
            ((clipped // 10) % 10) * 10,
            ((clipped // 100) % 10) * 100,
            ((clipped // 1_000) % 10) * 1_000,
            ((clipped // 10_000) % 10) * 10_000,
            ((clipped // 100_000) % 10) * 100_000,
            (clipped // 1_000_000) * 1_000_000,
        ]

    def decompose_total_to_digit_components(self, total_normalized: float) -> list[float]:
        """
        Dekomposisi prediksi total ke 7 komponen digit dalam ruang normalisasi.

        Catatan:
        - Prediksi di-round ke rupiah integer dulu.
        - Nilai di-clip ke rentang representasi model [0, 1_999_999].
        """
        total_rp = self._normalized_to_rp_int(total_normalized)
        comps_rp = self._decompose_rp_to_digit_components(total_rp)
        return self._components_rp_to_normalized(comps_rp)

    def backward(self, target: float, learning_rate: float) -> None:
        # d(MSE)/dy = 2*(y_pred - y)
        d_loss_d_y = 2.0 * (self._y_pred - target)
        d_loss_d_y = self._clip_gradient(d_loss_d_y, self.clip_value)

        # Karena y = sum(c_k), maka dL/dc_k = dL/dy * 1
        d_loss_d_c = [d_loss_d_y for _ in range(self.num_components)]

        # dL/dz_comp_k
        d_loss_d_z_comp = []
        for k in range(self.num_components):
            grad = d_loss_d_c[k] * self._component_activate_derivative(
                self._z_comp[k],
                k,
                self._sigmoid_comp[k],
            )
            d_loss_d_z_comp.append(self._clip_gradient(grad, self.clip_value))

        # Update w2, b2 + hitung grad ke hidden
        d_loss_d_h = [0.0] * self.hidden_size

        for k in range(self.num_components):
            for j in range(self.hidden_size):
                # grad ke hidden pakai bobot lama (sebelum update)
                d_loss_d_h[j] += self.w2[k][j] * d_loss_d_z_comp[k]

                grad_w2 = d_loss_d_z_comp[k] * self._h[j]
                if self.l2_lambda > 0:
                    grad_w2 += self.l2_lambda * self.w2[k][j]
                self.w2[k][j] -= learning_rate * grad_w2

            self.b2[k] -= learning_rate * d_loss_d_z_comp[k]

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
        pred = self.forward(inputs)
        loss = self._mse_loss(pred, target)
        self.backward(target, learning_rate)
        return loss

    def train_batch(
        self,
        x_batch,
        y_batch,
        learning_rate: float,
    ) -> float:
        import numpy as np
        total_loss = 0.0
        n = len(x_batch)
        for i in range(n):
            # Handle both lists and numpy arrays
            x_val = x_batch[i].tolist() if isinstance(x_batch[i], np.ndarray) else x_batch[i]
            y_val = float(y_batch[i][0] if getattr(y_batch, "ndim", 1) == 2 else y_batch[i])
            total_loss += self.train_one_sample(x_val, y_val, learning_rate)
        return total_loss / n if n > 0 else 0.0

    def predict(self, inputs):
        import numpy as np
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            return np.array([self.forward(x.tolist()) for x in inputs])
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], list):
            return np.array([self.forward(x) for x in inputs])
        else:
            # Single sample, maybe numpy array
            if isinstance(inputs, np.ndarray):
                inputs = inputs.tolist()
            return self.forward(inputs)

    def predict_components(self, inputs: list[float]) -> list[float]:
        """
        Inferensi komponen place-value pada skala normalisasi.

        Output memakai dekomposisi digit dari prediksi total agar patuh aturan
        place-value (bukan output mentah multi-output regression).
        """
        total = self.forward(inputs)
        return self.decompose_total_to_digit_components(total)

    def predict_component_details(self, inputs: list[float]) -> dict:
        """
        Detail komponen output.

        Keterangan:
        - component_outputs_learned: output mentah head komponen model
        - component_outputs: output final patuh aturan digit (hasil dekomposisi total)
        - y_pred_total: jumlah komponen mentah model
        """
        self.forward(inputs)
        component_outputs_digit = self.decompose_total_to_digit_components(self._y_pred)

        return {
            "component_inputs": list(self._z_comp),
            "component_outputs_learned": list(self._c),
            "component_outputs": component_outputs_digit,
            "component_max_values": list(self.component_max_values),
            "y_pred_total": self._y_pred,
        }

    def save(self, path: str, metadata: dict | None = None) -> None:
        data: dict = {
            "model_class": "PascabayarPlaceValueModel",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_components": self.num_components,
            "clip_value": self.clip_value,
            "l2_lambda": self.l2_lambda,
            "component_activation": self.component_activation,
            "component_max_values": self.component_max_values,
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
            num_components=data.get("num_components", 7),
            clip_value=data.get("clip_value", 5.0),
            l2_lambda=data.get("l2_lambda", 0.0),
            component_activation=data.get("component_activation", "relu"),
        )

        model.component_max_values = data.get("component_max_values", model.component_max_values)

        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]

        metadata: dict = data.get("metadata", {})
        return model, metadata

    def get_summary(self) -> str:
        params = (
            self.input_size * self.hidden_size + self.hidden_size
            + self.hidden_size * self.num_components + self.num_components
        )
        return (
            "PascabayarPlaceValueModel | "
            f"Arsitektur: [{self.input_size}, {self.hidden_size}, {self.num_components}] + sum | "
            f"Aktivasi hidden: ReLU | Aktivasi komponen: {self.component_activation} (bounded) | "
            "Output komponen final: digit-constrained decomposition | "
            f"Total parameter: {params:,}"
        )
