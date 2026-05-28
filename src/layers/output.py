import random

class OutputLayer:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

        self.weights = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]

        self.bias = 0.0

        self.last_inputs = []
        self.last_output = 0.0

    def forward(self, inputs: list[float]) -> float:
        self.last_inputs = inputs

        output = self.bias

        for input_idx in range(self.hidden_size):
            output += self.weights[input_idx] * inputs[input_idx]

        self.last_output = output

        return output

    def backward(self, target: float, learning_rate: float) -> list[float]:
        prediction = self.last_output

        d_loss_d_output = 2.0 * (prediction - target)

        hidden_gradients = [0.0 for _ in range(self.hidden_size)]

        for input_idx in range(self.hidden_size):
            hidden_gradients[input_idx] = self.weights[input_idx] * d_loss_d_output
            gradient_weight = d_loss_d_output * self.last_inputs[input_idx]
            self.weights[input_idx] -= learning_rate * gradient_weight

        self.bias -= learning_rate * d_loss_d_output

        return hidden_gradients
