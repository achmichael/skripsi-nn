import json

from src.layers.input import InputLayer
from src.layers.hidden import HiddenLayer
from src.layers.output import OutputLayer


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int = 32):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = InputLayer(input_size)
        self.hidden_layer = HiddenLayer(input_size, hidden_size)
        self.output_layer = OutputLayer(hidden_size)

    def forward(self, inputs: list[float]) -> float:
        x = self.input_layer.forward(inputs)
        p = self.hidden_layer.forward(x)
        y = self.output_layer.forward(p)

        return y

    def train_one_sample(
        self,
        inputs: list[float],
        target: float,
        learning_rate: float,
    ) -> float:
        prediction = self.forward(inputs)

        loss = mean_squared_error_single(prediction, target)

        output_gradients = self.output_layer.backward(target, learning_rate)
        self.hidden_layer.backward(output_gradients, learning_rate)

        return loss

    def predict(self, inputs: list[float]) -> float:
        return self.forward(inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "hidden_weights": self.hidden_layer.weights,
            "hidden_biases": self.hidden_layer.biases,
            "output_weights": self.output_layer.weights,
            "output_bias": self.output_layer.bias,
            "metadata": metadata or {},
        }

        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, mode="r", encoding="utf-8") as file:
            model_data = json.load(file)

        model = cls(
            input_size=model_data["input_size"],
            hidden_size=model_data["hidden_size"],
        )

        model.hidden_layer.weights = model_data["hidden_weights"]
        model.hidden_layer.biases = model_data["hidden_biases"]
        model.output_layer.weights = model_data["output_weights"]
        model.output_layer.bias = model_data["output_bias"]

        return model, model_data.get("metadata", {})


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
    - epochs=None: train indefinitely until no improvement for `patience` epochs.
    - epochs=N: train at most N epochs, still apply early stopping.
    - min_delta: minimum loss improvement to count as "better".
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

        # Early stopping check: improvement must exceed min_delta
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
