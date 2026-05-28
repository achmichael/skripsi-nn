class InputLayer:
    def __init__(self, input_size: int):
        self.input_size = input_size


    def forward(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Jumlah input tidak sesuai. Diharapkan {self.input_size}, "
                f"tetapi menerima {len(inputs)}"
            )

        return inputs