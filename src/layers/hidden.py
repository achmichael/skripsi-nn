import random
from src.activations.ReLU import relu, relu_derivative

class HiddenLayer:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weights = [
            [random.uniform(-0.5, 0.5) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]

        self.biases = [0.0 for _ in range(hidden_size)]

        self.last_inputs = []
        self.last_z = []
        self.last_outputs = []
    
    def forward(self, inputs: list[float]) -> list[float]:
        self.last_inputs = inputs
        # to store preactivation
        self.last_z = []
        # to store output after activation
        self.last_outputs = []

        for neuron_index in range(self.hidden_size):
            z = self.biases[neuron_index]

            for input_index in range(self.input_size):
                z += self.weights[neuron_index][input_index] * inputs[input_index]

            output = relu(z)

            self.last_z.append(z)
            self.last_outputs.append(output)

        return self.last_outputs

    
    def backward(self, output_gradients: list[float], learning_rate: float) -> list[float]:
        input_gradients = [0.0 for _ in range(self.input_size)]

        for neuron_index in range(self.hidden_size):
            dz = output_gradients[neuron_index] * relu_derivative(
                self.last_z[neuron_index]
            )

            for input_index in range(self.input_size):
                input_gradients[input_index] += (
                    dz * self.weights[neuron_index][input_index]
                )

                gradient_weight = dz * self.last_inputs[input_index]

                self.weights[neuron_index][input_index] -= (
                    learning_rate * gradient_weight
                )

            self.biases[neuron_index] -= learning_rate * dz

        return input_gradients
        