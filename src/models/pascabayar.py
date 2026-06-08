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

    # ------------------------------------------------------------------
    # Mini-Batch Gradient Descent
    # ------------------------------------------------------------------

    def _compute_deltas(self, target: float) -> list[list[float]]:
        """
        Hitung delta (error signal) untuk setiap neuron di setiap layer.
        Dipanggil SETELAH forward() karena membutuhkan _activations dan
        _pre_activations yang sudah terisi.

        Rumus:
          - Output layer (linear): delta = d(MSE)/d(output) = 2*(pred - target)
          - Hidden layer (ReLU) : delta_i = (Σ_j w_ji * delta_j) * ReLU'(z_i)
            (error mengalir mundur dari layer berikutnya melalui bobot)

        Args:
            target: Nilai target aktual untuk sampel ini.

        Returns:
            deltas: List of list, deltas[l][j] = delta untuk neuron j di layer l.
        """
        prediction = self._activations[-1][0]
        output_grad = 2.0 * (prediction - target)

        deltas: list[list[float]] = [None] * (self.num_layers - 1)  # type: ignore

        # Output layer (aktivasi linear, jadi delta = output_grad langsung)
        deltas[-1] = [self._clip_gradient(output_grad, self.clip_value)]

        # Hidden layers: propagasi error dari kanan ke kiri
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

        return deltas

    def train_batch(
        self,
        x_batch: list[list[float]],
        y_batch: list[float],
        learning_rate: float,
    ) -> float:
        """
        Melatih model dengan satu mini-batch data menggunakan
        Mini-Batch Gradient Descent.

        ═══════════════════════════════════════════════════════
        PERBEDAAN UTAMA DENGAN train_one_sample / backward:
        ═══════════════════════════════════════════════════════
        Pada Pure SGD (train_one_sample + backward):
          → Setiap 1 baris data langsung update bobot.
          → Dengan data kuesioner yang noisy, bobot "terombang-ambing"
            karena tiap sampel menarik bobot ke arah berbeda.
          → Ini menyebabkan "regression to the mean".

        Pada Mini-Batch GD (train_batch):
          → Kita proses batch_size sampel dulu (misal 16 atau 32).
          → Gradient dari setiap sampel DIAKUMULASI, bukan langsung
            dipakai update.
          → Setelah seluruh batch selesai, gradient DIRATA-RATAKAN
            (dibagi batch_size), baru kemudian update bobot SEKALI.
          → Rata-rata ini "meredam" noise dari sampel individual,
            sehingga arah update bobot lebih stabil dan akurat.

        ALGORITMA STEP-BY-STEP:
        ───────────────────────
        1. Buat akumulator gradient = 0 (bentuknya sama persis dengan
           struktur weights dan biases model)

        2. Untuk SETIAP sampel dalam batch:
           a. Forward pass  → hitung prediksi
           b. Compute deltas → hitung error signal (delta) per neuron
           c. Hitung gradient:
              - gradient bobot = delta_j × input_i
              - gradient bias  = delta_j
           d. TAMBAHKAN gradient ke akumulator (JANGAN update bobot!)

        3. Setelah SEMUA sampel dalam batch selesai:
           a. Bagi akumulator dengan batch_size → gradient rata-rata
           b. Clip gradient rata-rata
           c. Update bobot:
              w -= learning_rate × (avg_grad + l2_lambda × w)
              │                      │              └─ L2 regularization:
              │                      │                 mendorong bobot besar
              │                      │                 menuju nol, mencegah
              │                      │                 overfitting
              │                      └─ sinyal dari data (sudah dihaluskan
              │                         karena dirata-rata dari batch_size
              │                         sampel)
              └─ laju pembelajaran
           d. Update bias:
              b -= learning_rate × avg_grad_bias
              (bias TIDAK di-regularisasi, ini praktik standar)

        Args:
            x_batch      : List of list, setiap elemen = 1 input sample.
            y_batch      : List of float, target untuk setiap sampel.
            learning_rate: Laju pembelajaran.

        Returns:
            Rata-rata MSE loss untuk batch ini.
        """
        batch_size = len(x_batch)

        # =============================================================
        # LANGKAH 1: Inisialisasi akumulator gradient ke nol
        # =============================================================
        # Strukturnya mengikuti self.weights dan self.biases:
        #   acc_w[l][j][i] = akumulasi gradient untuk weights[l][j][i]
        #   acc_b[l][j]    = akumulasi gradient untuk biases[l][j]
        acc_w: list[list[list[float]]] = []
        acc_b: list[list[float]] = []

        for l in range(self.num_layers - 1):
            layer_w = []
            for j in range(self.layer_sizes[l + 1]):
                layer_w.append([0.0] * self.layer_sizes[l])
            acc_w.append(layer_w)
            acc_b.append([0.0] * self.layer_sizes[l + 1])

        total_loss = 0.0

        # =============================================================
        # LANGKAH 2: Loop setiap sampel → forward, delta, akumulasi
        # =============================================================
        for s in range(batch_size):
            inputs = x_batch[s]
            target = y_batch[s]

            # 2a. Forward pass
            prediction = self.forward(inputs)

            # Catat loss untuk monitoring
            total_loss += self._mse_loss(prediction, target)

            # 2b. Hitung deltas (error signal per neuron)
            deltas = self._compute_deltas(target)

            # 2c-2d. Hitung gradient dan AKUMULASI
            for l in range(self.num_layers - 1):
                # input ke layer l = output dari layer sebelumnya
                inputs_l = self._activations[l]

                for j in range(self.layer_sizes[l + 1]):
                    for i in range(self.layer_sizes[l]):
                        # gradient bobot = delta × input
                        acc_w[l][j][i] += deltas[l][j] * inputs_l[i]

                    # gradient bias = delta
                    acc_b[l][j] += deltas[l][j]

        # =============================================================
        # LANGKAH 3: Rata-rata gradient, clip, lalu update bobot
        # =============================================================
        for l in range(self.num_layers - 1):
            for j in range(self.layer_sizes[l + 1]):
                for i in range(self.layer_sizes[l]):
                    # 3a. Rata-ratakan gradient
                    avg_grad = acc_w[l][j][i] / batch_size

                    # 3b. Clip gradient
                    avg_grad = self._clip_gradient(avg_grad, self.clip_value)

                    # 3c. Update bobot DENGAN L2 regularization (weight decay)
                    # Tambahkan l2_lambda * w ke gradient → mendorong bobot
                    # besar menuju nol agar model tidak overfitting
                    if self.l2_lambda > 0:
                        avg_grad += self.l2_lambda * self.weights[l][j][i]

                    self.weights[l][j][i] -= learning_rate * avg_grad

                # 3d. Update bias (TANPA L2 — bias tidak di-regularisasi)
                avg_bias_grad = acc_b[l][j] / batch_size
                avg_bias_grad = self._clip_gradient(avg_bias_grad, self.clip_value)
                self.biases[l][j] -= learning_rate * avg_bias_grad

        # Kembalikan rata-rata loss batch
        return total_loss / batch_size

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