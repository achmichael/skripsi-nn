import json
import math
import random

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork

def mean_squared_error_single(prediction: float, target: float) -> float:
    error = prediction - target
    return error * error


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    total_error = 0.0

    for prediction, target in zip(predictions, targets):
        total_error += mean_squared_error_single(prediction, target)

    return total_error / len(targets)


def mean_absolute_error(predictions: list[float], targets: list[float]) -> float:
    total_error = 0.0

    for prediction, target in zip(predictions, targets):
        total_error += abs(prediction - target)

    return total_error / len(targets)


def train_model(
    model: NeuralNetwork,
    x_train: list[list[float]],
    y_train: list[float],
    learning_rate: float,
    patience: int = 5,
    min_delta: float = 1e-4,
    epochs: int | None = None,
) -> list[float]:
    """
    Train with early stopping.
    - epochs=None: unlimited, stop by patience only.
    - min_delta: minimum loss improvement to count as progress.
    """
    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0

    while True:
        epoch += 1

        if epochs is not None and epoch > epochs:
            print(f"Mencapai batas maksimum {epochs} epoch.")
            break

        total_loss = 0.0

        for inputs, target in zip(x_train, y_train):
            loss = model.train_one_sample(inputs, target, learning_rate)
            total_loss += loss

        average_loss = total_loss / len(x_train)
        history.append(average_loss)

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {average_loss:.8f}")

        if average_loss < best_loss - min_delta:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping di epoch {epoch}. "
                f"Tidak ada perbaikan selama {patience} epoch terakhir. "
                f"Loss terbaik: {best_loss:.8f}"
            )
            break

    return history


def evaluate_model(
    model: NeuralNetwork,
    x_test: list[list[float]],
    y_test: list[float],
) -> dict:
    predictions = []

    for inputs in x_test:
        predictions.append(model.predict(inputs))

    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)

    return {
        "mse": mse,
        "mae": mae,
        "predictions": predictions,
    }
