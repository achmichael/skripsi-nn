import math
import random
from abc import ABC, abstractmethod


class NeuralNetwork(ABC):
    """
    Abstract base class untuk semua model Neural Network dalam proyek ini.

    Subclass wajib mengimplementasikan:
      - forward(inputs)
      - backward(target, learning_rate)
      - train_one_sample(inputs, target, learning_rate)
      - predict(inputs)
      - save(path, metadata)
      - load(path)  [classmethod]
    """

    # ------------------------------------------------------------------
    # Abstract Methods — wajib diimplementasikan oleh subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, inputs: list[float]) -> float:
        """
        Melakukan propagasi maju (forward pass).

        Args:
            inputs: Vektor fitur input (sudah dinormalisasi).

        Returns:
            Nilai prediksi output (float).
        """
        ...

    @abstractmethod
    def backward(self, target: float, learning_rate: float) -> None:
        """
        Melakukan propagasi mundur (backpropagation) dan memperbarui bobot.

        Args:
            target       : Nilai target aktual (sudah dinormalisasi).
            learning_rate: Laju pembelajaran.
        """
        ...

    @abstractmethod
    def train_one_sample(
        self,
        inputs: list[float],
        target: float,
        learning_rate: float,
    ) -> float:
        """
        Menjalankan satu langkah pelatihan untuk satu sampel data.

        Urutan:
          1. forward(inputs)
          2. hitung loss
          3. backward(target, learning_rate)

        Args:
            inputs       : Vektor fitur input (sudah dinormalisasi).
            target       : Nilai target aktual (sudah dinormalisasi).
            learning_rate: Laju pembelajaran.

        Returns:
            Nilai loss (MSE) untuk sampel ini.
        """
        ...

    @abstractmethod
    def train_batch(
        self,
        x_batch: list[list[float]],
        y_batch: list[float],
        learning_rate: float,
    ) -> float:
        """
        Melatih model dengan satu mini-batch data.

        Proses:
          1. Untuk setiap sampel dalam batch:
             - forward pass → hitung prediksi
             - backward pass → hitung gradient (delta)
             - akumulasi gradient (TANPA update bobot)
          2. Setelah semua sampel diproses:
             - rata-ratakan gradient terakumulasi (bagi dengan batch_size)
             - clip gradient
             - update bobot: w -= lr * (avg_grad + l2_lambda * w)
             - update bias:  b -= lr * avg_grad_bias

        Args:
            x_batch      : List berisi input tiap sampel dalam batch.
            y_batch      : List berisi target tiap sampel dalam batch.
            learning_rate: Laju pembelajaran.

        Returns:
            Rata-rata MSE loss untuk batch ini.
        """
        ...

    @abstractmethod
    def predict(self, inputs: list[float]) -> float:
        """
        Melakukan inferensi tanpa memperbarui bobot (inference only).

        Args:
            inputs: Vektor fitur input (sudah dinormalisasi).

        Returns:
            Nilai prediksi output (float).
        """
        ...

    @abstractmethod
    def save(self, path: str, metadata: dict | None = None) -> None:
        """
        Menyimpan bobot model beserta metadata ke file JSON.

        Args:
            path    : Path lengkap file JSON tujuan.
            metadata: Dict opsional berisi informasi tambahan seperti
                      feature_columns, scaler, layer_sizes, dll.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> tuple["NeuralNetwork", dict]:
        """
        Memuat model dari file JSON yang sebelumnya disimpan via save().

        Args:
            path: Path lengkap file JSON sumber.

        Returns:
            Tuple (model, metadata) di mana:
              - model   : Instance NeuralNetwork yang sudah diisi bobotnya.
              - metadata: Dict berisi informasi tambahan yang disimpan saat save().
        """
        ...

    # ------------------------------------------------------------------
    # Concrete Helper Methods — tersedia untuk semua subclass
    # ------------------------------------------------------------------

    @staticmethod
    def _mse_loss(prediction: float, target: float) -> float:
        """Menghitung Mean Squared Error untuk satu sampel."""
        error = prediction - target
        return error * error

    @staticmethod
    def _he_init(fan_in: int, seed: int | None = None) -> float:
        """
        Menghasilkan satu nilai bobot inisialisasi He (He et al., 2015).

        Skala = sqrt(2 / fan_in), sesuai untuk aktivasi ReLU.

        Args:
            fan_in: Jumlah neuron pada layer sebelumnya.
            seed  : Seed opsional untuk reproduksibilitas.

        Returns:
            Nilai bobot acak terdistribusi normal dengan skala He.
        """
        if seed is not None:
            random.seed(seed)
        std = math.sqrt(2.0 / fan_in)
        # Box-Muller transform untuk distribusi normal
        u1 = random.random() or 1e-10
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return z * std

    @staticmethod
    def _clip_gradient(gradient: float, clip_value: float) -> float:
        """
        Memotong (clip) nilai gradient agar tidak melebihi batas.

        Args:
            gradient  : Nilai gradient yang akan diclip.
            clip_value: Nilai batas absolut maksimum.

        Returns:
            Gradient setelah diclip dalam rentang [-clip_value, clip_value].
        """
        return max(-clip_value, min(clip_value, gradient))

    def get_summary(self) -> str:
        """
        Mengembalikan ringkasan tekstual arsitektur model (opsional).

        Subclass dapat meng-override metode ini untuk menampilkan detail
        arsitektur seperti jumlah layer, neuron per layer, dan parameter.

        Returns:
            String deskripsi model.
        """
        return f"{self.__class__.__name__}()"
