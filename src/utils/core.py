import json
import math
import random
import copy
import numpy as np

from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork
from src.pipeline.preprocessing import inverse_transform_target

def compute_sample_weight(
    y_normalized: float,
    y_scaler: dict,
    use_log: bool = False,
    model_type: str = "pascabayar",
) -> float:
    """
    Compute per-sample loss weight based on the original Rupiah value
    of the target. Samples with low bill amounts receive higher weight
    to compensate for their under-representation in the training set.

    Weight scheme:
      Aktual < 100,000 Rp  → weight = 3.0  (low bill, underrepresented)
      Aktual < 200,000 Rp  → weight = 1.5  (medium-low, slightly boost)
      Aktual >= 200,000 Rp → weight = 1.0  (normal, no boost)

    Args:
        y_normalized : target value in normalized scale (after minmax 
                       and optional log transform)
        y_scaler     : scaler dict from fit_target_scaler(), used to 
                       convert normalized value back to Rupiah
        use_log      : whether log transform was applied to target,
                       must match the use_log_transform config value

    Returns:
        float: sample weight >= 1.0
    """
    # Convert normalized → original scale (Rupiah for pascabayar, days for prabayar)
    y_original = inverse_transform_target(y_normalized, y_scaler)

    if model_type == "pascabayar":
        if y_original < 100_000:
            return 2.0
        elif y_original < 200_000:
            return 1.5
        else:
            return 1.0
    elif model_type == "prabayar":
        if y_original <= 10.0:
            return 2.5
        elif y_original >= 25.0:
            return 2.0
        else:
            return 1.0
    
    return 1.0

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
    # ── NEW PARAMETERS ──────────────────────────────────────
    use_sample_weights: bool = False,
    y_scaler: dict | None = None,
    use_log: bool = False,
    model_type: str = "pascabayar",
    # ────────────────────────────────────────────────────────
) -> dict:
    """
    use_sample_weights : if True, apply per-sample loss weighting based on original Rupiah value
    y_scaler          : required if use_sample_weights=True, used to inverse-transform for weight lookup
    use_log           : must match use_log_transform config value, passed through to inverse_transform_target
    """
    
    # Convert lists to NumPy arrays for fast vectorized operations
    X_train_np = np.array(x_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32).reshape(-1, 1)

    n_samples = len(X_train_np)
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    best_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0
    best_weights = None
    best_biases = None

    has_val = x_val is not None and y_val is not None
    if has_val:
        X_val_np = np.array(x_val, dtype=np.float32)
        y_val_np = np.array(y_val, dtype=np.float32).reshape(-1, 1)

    if use_sample_weights and y_scaler is not None:
        w_counts = {1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0}
        for y_val_i in y_train:
            w = compute_sample_weight(float(y_val_i), y_scaler, use_log, model_type)
            w_counts[w] = w_counts.get(w, 0) + 1
            
        if model_type == "pascabayar":
            print(f"[Sample Weights] weight=2.0 (< 100rb) : {w_counts.get(2.0, 0)} samples")
            print(f"[Sample Weights] weight=1.5 (< 200rb) : {w_counts.get(1.5, 0)} samples")
            print(f"[Sample Weights] weight=1.0 (>= 200rb): {w_counts.get(1.0, 0)} samples")
        else:
            print(f"[Sample Weights] weight=2.5 (<= 10 hari) : {w_counts.get(2.5, 0)} samples")
            print(f"[Sample Weights] weight=2.0 (>= 25 hari) : {w_counts.get(2.0, 0)} samples")
            print(f"[Sample Weights] weight=1.0 (normal) : {w_counts.get(1.0, 0)} samples")

    while True:
        epoch += 1

        if epochs is not None and epoch > epochs:
            print(f"Mencapai batas maksimum {epochs} epoch.")
            break

        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train_np[indices]
        y_shuffled = y_train_np[indices]

        current_lr = learning_rate / (1.0 + lr_decay * epoch)

        total_train_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            if use_sample_weights and y_scaler is not None:
                # Compute per-sample weights for this batch
                weights = [
                    compute_sample_weight(
                        y_normalized=float(y_val_i.item()),
                        y_scaler=y_scaler,
                        use_log=use_log,
                        model_type=model_type,
                    )
                    for y_val_i in y_batch
                ]
                batch_loss = model.train_batch_weighted(
                    x_batch, y_batch, current_lr, weights
                )
            else:
                batch_loss = model.train_batch(x_batch, y_batch, current_lr)
            
            total_train_loss += batch_loss
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches
        train_loss_history.append(avg_train_loss)

        if has_val:
            # Vectorized validation prediction
            val_preds = model.predict(X_val_np)
            avg_val_loss = float(np.mean((val_preds - y_val_np) ** 2))
            val_loss_history.append(avg_val_loss)
            val_info = f" | Val Loss: {avg_val_loss:.8f}"
        else:
            val_info = ""

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.8f}{val_info}")

        monitor_loss = avg_val_loss if has_val else avg_train_loss
        if monitor_loss < best_loss - min_delta:
            best_loss = monitor_loss
            epochs_without_improvement = 0
            if hasattr(model, 'weights') and hasattr(model, 'biases'):
                best_weights = [np.copy(w) for w in model.weights]
                best_biases = [np.copy(b) for b in model.biases]
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            monitor_name = "val" if has_val else "train"
            print(
                f"Early stopping di epoch {epoch}. "
                f"Tidak ada perbaikan {monitor_name} loss selama {patience} epoch terakhir. "
                f"Loss terbaik: {best_loss:.8f}"
            )
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
    
    # Can process all at once for speed
    X_test_np = np.array(x_test, dtype=np.float32)
    preds_np = model.predict(X_test_np)
    
    predictions = preds_np.flatten().tolist()

    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)

    return {
        "mse": mse,
        "mae": mae,
        "predictions": predictions,
    }
