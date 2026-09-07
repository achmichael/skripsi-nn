import math
import random
import numpy as np
from abc import ABC, abstractmethod


class NeuralNetwork(ABC):
    """
    Abstract base class untuk semua model Neural Network dalam proyek ini.

    Subclass wajib mengimplementasikan:
      - forward(inputs)
      - backward(target, learning_rate)
      - train_one_sample(inputs, target, learning_rate)
      - train_batch(x_batch, y_batch, learning_rate)
      - predict(inputs)
      - save(path, metadata)
      - load(path)  [classmethod]
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray, cat_inputs: list[dict[str, int]] = None) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, target: np.ndarray, learning_rate: float) -> None:
        ...

    @abstractmethod
    def train_one_sample(
        self,
        inputs: np.ndarray,
        cat_inputs: dict[str, int],
        target: np.ndarray,
        learning_rate: float,
    ) -> float:
        ...

    @abstractmethod
    def train_batch(
        self,
        x_batch: np.ndarray,
        x_cat_batch: list[dict[str, int]],
        y_batch: np.ndarray,
        learning_rate: float,
    ) -> float:
        ...

    @abstractmethod
    def get_feature_contributions(self) -> np.ndarray:
        ...

    @abstractmethod
    def predict(self, inputs: np.ndarray, cat_inputs: list[dict[str, int]] = None) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str, metadata: dict | None = None) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> tuple["NeuralNetwork", dict]:
        ...

    @staticmethod
    def _mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
        error = prediction - target
        # print('prediction', prediction)
        # print('target', target)
        # print('error', error)
        # mengkonversi nilai numpy (nilai rata-rata error yang sudah dikuadratkan) menjadi float
        return float(np.mean(error ** 2))

    @staticmethod
    def _he_init(fan_in: int, seed: int | None = None) -> float:
        if seed is not None:
            np.random.seed(seed)
        return float(np.random.randn() * math.sqrt(2.0 / fan_in))

    # clip value = 10, maka gradient clipping menerapkan aturan berikut:
    # 1. Jika terdapat nilai yang lebih kecil dari -10 misal -50.75 maka nilai akan dipaksa menjadi -10
    # 2. Jika terdapat nilai yang lebih besar dari 10 misal 50.22 maka nilai akan dipaksa menjadi 10
    # 3. Jika nilai berada pada rentang -10 hingga 10 misal (0.05) nilainya akan dibiarkan apa adanya
    @staticmethod
    def _clip_gradient(gradient: np.ndarray, clip_value: float) -> np.ndarray:
        return np.clip(gradient, -clip_value, clip_value)

    def get_summary(self) -> str:
        return f"{self.__class__.__name__}()"
