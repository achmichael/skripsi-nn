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
    x_val: list[list[float]] | None = None,
    y_val: list[float] | None = None,
) -> dict:
    """
    Train dengan early stopping dan opsional validation loss per epoch.

    Args:
        model        : Instance NeuralNetwork yang akan dilatih.
        x_train      : Fitur data training (sudah dinormalisasi).
        y_train      : Target data training (sudah dinormalisasi).
        learning_rate: Laju pembelajaran.
        patience     : Jumlah epoch tanpa perbaikan sebelum early stopping.
        min_delta    : Minimum penurunan loss yang dianggap sebagai perbaikan.
        epochs       : Batas maksimum epoch (None = tidak terbatas).
        x_val        : Fitur data validasi opsional (sudah dinormalisasi).
        y_val        : Target data validasi opsional (sudah dinormalisasi).

    Returns:
        Dict berisi:
          - "train_loss": list train loss per epoch
          - "val_loss"  : list val loss per epoch (kosong jika x_val tidak diberikan)
    """
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    best_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0

    has_val = x_val is not None and y_val is not None

    while True:
        epoch += 1

        if epochs is not None and epoch > epochs:
            print(f"Mencapai batas maksimum {epochs} epoch.")
            break

        # --- Shuffle training data setiap epoch (P4) ---
        combined_train = list(zip(x_train, y_train))
        random.shuffle(combined_train)

        # --- Training pass (update bobot) ---
        total_train_loss = 0.0
        for inputs, target in combined_train:
            loss = model.train_one_sample(inputs, target, learning_rate)
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(x_train)
        train_loss_history.append(avg_train_loss)

        # --- Validation pass (inferensi saja, tanpa update bobot) ---
        if has_val:
            total_val_loss = 0.0
            for inputs, target in zip(x_val, y_val):  # type: ignore[arg-type]
                pred = model.predict(inputs)
                total_val_loss += mean_squared_error_single(pred, target)
            avg_val_loss = total_val_loss / len(y_val)  # type: ignore[arg-type]
            val_loss_history.append(avg_val_loss)
            val_info = f" | Val Loss: {avg_val_loss:.8f}"
        else:
            val_info = ""

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.8f}{val_info}")

        # --- Early stopping berbasis val loss (jika ada), fallback ke train loss (P6) ---
        monitor_loss = avg_val_loss if has_val else avg_train_loss
        if monitor_loss < best_loss - min_delta:
            best_loss = monitor_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            monitor_name = "val" if has_val else "train"
            print(
                f"Early stopping di epoch {epoch}. "
                f"Tidak ada perbaikan {monitor_name} loss selama {patience} epoch terakhir. "
                f"Loss terbaik: {best_loss:.8f}"
            )
            break

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }


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
