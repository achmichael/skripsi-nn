import numpy as np

def relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, value)

def relu_derivative(value: np.ndarray) -> np.ndarray:
    return np.where(value > 0, 1.0, 0.0)