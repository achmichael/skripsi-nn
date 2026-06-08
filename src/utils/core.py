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
    batch_size: int = 16,
    patience: int = 5,
    min_delta: float = 1e-4,
    epochs: int | None = None,
    x_val: list[list[float]] | None = None,
    y_val: list[float] | None = None,
    lr_decay: float = 0.0,
) -> dict:
    """
    Train dengan Mini-Batch Gradient Descent, early stopping,
    dan opsional validation loss per epoch.

    ════════════════════════════════════════════════════════════
    CARA KERJA MINI-BATCH DALAM TRAINING LOOP:
    ════════════════════════════════════════════════════════════
    Jika ada 100 data training dan batch_size=16:
      - Batch 1: data ke-0  s/d 15  (16 sampel)
      - Batch 2: data ke-16 s/d 31  (16 sampel)
      - ...
      - Batch 6: data ke-80 s/d 95  (16 sampel)
      - Batch 7: data ke-96 s/d 99  (4 sampel — sisa terakhir)

    Setiap batch → 1× update bobot.
    Jadi per epoch ada 7 update, bukan 100 (Pure SGD) atau 1 (Full-Batch).

    KENAPA ini membantu data kuesioner yang noisy:
      - Pure SGD: 1 baris noisy langsung mengubah bobot → tidak stabil
      - Mini-Batch: 16 baris dirata-ratakan → noise saling meredam
      - Bobot bergerak ke arah yang lebih "benar" dan konsisten

    Args:
        model        : Instance NeuralNetwork yang akan dilatih.
        x_train      : Fitur data training (sudah dinormalisasi).
        y_train      : Target data training (sudah dinormalisasi).
        learning_rate: Laju pembelajaran.
        batch_size   : Jumlah sampel per mini-batch (default 16).
                       Rekomendasi: 16 atau 32 untuk data kuesioner.
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
    n_samples = len(x_train)
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    best_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0
    best_weights = None
    best_biases = None

    has_val = x_val is not None and y_val is not None

    while True:
        epoch += 1

        if epochs is not None and epoch > epochs:
            print(f"Mencapai batas maksimum {epochs} epoch.")
            break

        # --- Shuffle training data setiap epoch ---
        # Agar komposisi batch berbeda tiap epoch → model tidak
        # menghafal urutan data
        indices = list(range(n_samples))
        random.shuffle(indices)
        x_shuffled = [x_train[i] for i in indices]
        y_shuffled = [y_train[i] for i in indices]

        # --- Learning rate decay ---
        # Reduce LR over time: lr_t = lr_0 / (1 + decay * epoch)
        current_lr = learning_rate / (1.0 + lr_decay * epoch)

        # --- Training pass: potong data menjadi mini-batch ---
        total_train_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Latih model pada batch ini dan dapatkan rata-rata loss-nya
            batch_loss = model.train_batch(x_batch, y_batch, current_lr)
            total_train_loss += batch_loss
            n_batches += 1

        # Rata-rata loss dari semua batch dalam epoch ini
        avg_train_loss = total_train_loss / n_batches
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

        # --- Early stopping berbasis val loss (jika ada), fallback ke train loss ---
        monitor_loss = avg_val_loss if has_val else avg_train_loss
        if monitor_loss < best_loss - min_delta:
            best_loss = monitor_loss
            epochs_without_improvement = 0
            # Simpan salinan bobot terbaik (deep copy manual)
            if hasattr(model, 'weights') and hasattr(model, 'biases'):
                import copy
                best_weights = copy.deepcopy(model.weights)
                best_biases = copy.deepcopy(model.biases)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            monitor_name = "val" if has_val else "train"
            print(
                f"Early stopping di epoch {epoch}. "
                f"Tidak ada perbaikan {monitor_name} loss selama {patience} epoch terakhir. "
                f"Loss terbaik: {best_loss:.8f}"
            )
            # Restore bobot terbaik
            if best_weights is not None and best_biases is not None:
                model.weights = best_weights
                model.biases = best_biases
                print("Bobot model dikembalikan ke checkpoint terbaik.")
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
