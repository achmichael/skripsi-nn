def relu(value: float) -> float:
    return max(0.0, value)


def relu_derivative(value: float) -> float:
    return 1.0 if value > 0 else 0.0