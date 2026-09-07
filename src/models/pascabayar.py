import json
import math
import numpy as np
from src.activations.ReLU import relu, relu_derivative
from src.models.neural_network import NeuralNetwork
from src.models.embedding import MultiFeatureEmbedding

class PascabayarModel(NeuralNetwork):
    def __init__(
        self,
        layer_sizes: list[int],
        embedding_configs: list[dict] = None,
        seed: int | None = None,
        clip_value: float = 5.0,
        l2_lambda: float = 1e-4,
    ):
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes minimal harus memiliki 2 elemen (input dan output)."
            )
        if layer_sizes[-1] != 1:
            raise ValueError(
                f"Output layer harus berukuran 1 untuk regresi, "
                f"tetapi mendapat {layer_sizes[-1]}."
            )

        self.layer_sizes = layer_sizes
        self.clip_value = clip_value
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_sizes)

        if seed is not None:
            np.random.seed(seed)
            
        # Inisialisasi layer embedding jika ada konfigurasi
        self.embedding_layer = None
        self.has_embeddings = False
        if embedding_configs and len(embedding_configs) > 0:
            self.embedding_layer = MultiFeatureEmbedding(embedding_configs, seed=seed if seed else 42)
            self.has_embeddings = True

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for l in range(self.num_layers - 1):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            std = math.sqrt(2.0 / fan_in)

            W = np.random.randn(fan_out, fan_in) * std
            b = np.zeros(fan_out)

            self.weights.append(W)
            self.biases.append(b)

        # Adam optimizer parameters & state
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []
        self._original_numeric_dim = 0 # Cache ukuran asli input numerik

    def _process_input_with_embeddings(self, inputs: np.ndarray, cat_inputs: list[dict[str, int]] = None) -> np.ndarray:
        """Menggabungkan input numerik berskala dengan output embedding"""
        self._original_numeric_dim = inputs.shape[1]
        
        if not self.has_embeddings or not cat_inputs:
            return inputs
            
        # Ubah list of dict menjadi dictionary of lists sesuai kebutuhan MultiFeatureEmbedding
        batch_size = len(cat_inputs)
        formatted_cat_data = {cfg["name"]: [] for cfg in self.embedding_layer.feature_configs}
        
        for sample in cat_inputs:
            for key in formatted_cat_data.keys():
                formatted_cat_data[key].append(sample.get(key, 0))
                
        # Dapatkan vektor embedding
        emb_vectors = self.embedding_layer.forward(formatted_cat_data)
        emb_np = np.array(emb_vectors, dtype=np.float32)
        
        # Gabungkan secara horizontal: [Numeric_Features, Embedding_Features]
        combined_inputs = np.hstack((inputs, emb_np))
        return combined_inputs

    def forward(self, inputs: np.ndarray, cat_inputs: list[dict[str, int]] = None) -> np.ndarray:
        # Gabungkan numerik dan kategori (jika ada)
        processed_inputs = self._process_input_with_embeddings(inputs, cat_inputs)
        
        self._activations = [processed_inputs]
        self._pre_activations = []

        current = processed_inputs

        for l in range(self.num_layers - 1):
            is_output_layer = (l == self.num_layers - 2)

            if is_output_layer:
                z = np.dot(current, self.weights[l].T)
            else:
                z = np.dot(current, self.weights[l].T) + self.biases[l]

            self._pre_activations.append(z)

            # Logging nilai fitur sebelum dan sesudah masuk fungsi aktivasi ReLU
            # if not is_output_layer:
            #     print(f"[Pascabayar Forward] Layer {l+1} - Sebelum ReLU:\n{z}")

            a = z if is_output_layer else relu(z)

            # if not is_output_layer:
            #     print(f"[Pascabayar Forward] Layer {l+1} - Sesudah ReLU:\n{a}")

            self._activations.append(a)
            current = a

        return current

    def backward(self, target: np.ndarray, learning_rate: float) -> None:
        pass

    def get_feature_contributions(self) -> np.ndarray:
         """
         Menghitung tingkat kontribusi tiap fitur input berdasarkan
         rata-rata magnitudo bobot absolut di layer pertama.
         Merespon L1 Regularization yang menekan bobot fitur tak relevan ke 0.
         """
         if not self.weights:
             return np.array([])

         # self.weights[0] shape: (hidden_nodes, input_features)
         # Ambil rata-rata magnitudo absolut per fitur (axis=0)
         importance_scores = np.mean(np.abs(self.weights[0]), axis=0)

         # Normalisasi supaya jumlahnya 1.0 (100%)
         total_score = np.sum(importance_scores)
         if total_score > 0:
             importance_scores = importance_scores / total_score

         return importance_scores

    def train_one_sample(
        self,
        inputs: np.ndarray,
        cat_inputs: dict[str, int],
        target: np.ndarray,
        learning_rate: float,
    ) -> float:
        cat_batch = [cat_inputs] if cat_inputs else None
        return self.train_batch(inputs[np.newaxis, :], cat_batch, target[np.newaxis, :], learning_rate)

    def train_batch(
        self,
        x_batch: np.ndarray,
        x_cat_batch: list[dict[str, int]],
        y_batch: np.ndarray,
        learning_rate: float,
    ) -> float:
        batch_size = x_batch.shape[0]
        prediction = self.forward(x_batch, x_cat_batch)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)

        total_loss = self._mse_loss(prediction, y_batch)
        # print('total loss', total_loss)

        # output grad menggunakan faktor 2, dikarenakan melihat dari perhitungan loss (MSE) menurunkan (y^−y)2
        # menghitung delta out dan membagi dengan batch size
        output_grad = 2 * (prediction - y_batch) / batch_size

        # print('output grad', output_grad)
        deltas = [None] * (self.num_layers - 1)
        # print('deltas', deltas)
        # masukkan value output_grad ke elemen terakhir
        deltas[-1] = output_grad
        # print('deltas after assign value', deltas)
        # print('num layers', self.num_layers)

        # Hidden layer deltas — tanpa clipping pada delta
        for l in range(self.num_layers - 3, -1, -1):
            # propagasi dari layer setelahnya ke layer saat ini
            grad = np.dot(deltas[l + 1], self.weights[l + 1])
            # turunan fungsi relu
            grad *= relu_derivative(self._pre_activations[l])
            deltas[l] = grad

        # Jika punya embedding, propagasi gradien sampai ke layer embedding
        if self.has_embeddings:
            # Gradien error di input layer gabungan
            grad_at_input = np.dot(deltas[0], self.weights[0])
            # Slice/ambil gradien hanya untuk dimensi fitur embedding
            # Shape grad_at_input: (batch_size, dense_input_size)
            # Bagian embedding ada di indeks setelah _original_numeric_dim
            grad_for_embedding = grad_at_input[:, self._original_numeric_dim:].tolist()
            # Pembaruan bobot embedding
            self.embedding_layer.backward(grad_for_embedding, learning_rate, self.l2_lambda)

        self.t += 1

        # Update weight dan bias
        for l in range(self.num_layers - 1):
            inputs_l = self._activations[l]
            # np.dot mewakili sum antara sampel dalam batch dengan perkalian nilai aktivasi layer sebelumnya
            # method dot mewakili operasi sigma, karena otomatis akan menjumlahkan seluruh elemen
            grad_w = np.dot(deltas[l].T, inputs_l)
            # print('gradient bobot layer', l)
            # print('shape input', inputs_l.shape)
            # print('shape delta', deltas[l].shape)
            # print('shape gradient bobot', grad_w.shape)

            # L2 regularization: Grad += (λ/m)W — setelah gradien dihitung
            if self.l2_lambda > 0:
                grad_w += (self.l2_lambda / batch_size) * self.weights[l]

            # print('gradient bobot sebelum clipping', grad_w)
            # Gradient clipping — setelah L2, pada gradien weight
            grad_w = self._clip_gradient(grad_w, self.clip_value)
            # print('gradient bobot setelah clipping', grad_w)

            # Adam update for weights
            self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * grad_w
            self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (grad_w ** 2)
            m_w_hat = self.m_w[l] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[l] / (1 - self.beta2 ** self.t)

            self.weights[l] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            # self.weights[l] -= learning_rate * grad_w

            # Bias update — skip output layer (output layer tidak punya bias)
            if l < self.num_layers - 2:
                grad_b = np.sum(deltas[l], axis=0)
                grad_b = self._clip_gradient(grad_b, self.clip_value)

                # Adam update for biases
                # Momentum bias
                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grad_b
                # Variansi bias
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grad_b ** 2)
                # Momentum bias yang sudah dikoreksi
                m_b_hat = self.m_b[l] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[l] / (1 - self.beta2 ** self.t)

                self.biases[l] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return float(total_loss)

    def predict(self, inputs: np.ndarray, cat_inputs: list[dict[str, int]] = None) -> np.ndarray:
        return self.forward(inputs, cat_inputs)

    def save(self, path: str, metadata: dict | None = None) -> None:
        data: dict = {
            "model_class": "PascabayarModel",
            "layer_sizes": self.layer_sizes,
            "clip_value": self.clip_value,
            "l2_lambda": self.l2_lambda,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        
        if self.has_embeddings:
            # Simpan bobot tiap dictionary embedding
            data["embedding_configs"] = self.embedding_layer.feature_configs
            data["embedding_weights"] = {
                name: emb.get_weights() 
                for name, emb in self.embedding_layer.embeddings.items()
            }

        if metadata is not None:
            data["metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> tuple["PascabayarModel", dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_sizes: list[int] = data["layer_sizes"]
        clip_value: float = data.get("clip_value", 5.0)
        l2_lambda: float = data.get("l2_lambda", 1e-4)
        embedding_configs = data.get("embedding_configs", None)

        model = cls(
            layer_sizes=layer_sizes, 
            embedding_configs=embedding_configs,
            clip_value=clip_value, 
            l2_lambda=l2_lambda
        )
        
        model.weights = [np.array(w, dtype=np.float32) for w in data["weights"]]
        model.biases = [np.array(b, dtype=np.float32) for b in data["biases"]]
        
        if model.has_embeddings and "embedding_weights" in data:
            for name, w in data["embedding_weights"].items():
                model.embedding_layer.embeddings[name].load_weights(w)

        metadata: dict = data.get("metadata", {})
        return model, metadata

    def get_summary(self) -> str:
        total_params = sum(
            self.layer_sizes[l] * self.layer_sizes[l + 1] + self.layer_sizes[l + 1]
            for l in range(self.num_layers - 1)
        )
        return (
            f"PascabayarModel (Vectorized) | Arsitektur: {self.layer_sizes} | "
            f"Total parameter: {total_params:,} | "
            f"Clip value: {self.clip_value} | L2: {self.l2_lambda}"
        )
