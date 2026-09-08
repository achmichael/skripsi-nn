"""
Entity Embedding Module

Implementasi manual lapisan Entity Embedding tanpa menggunakan library ML.
Digunakan untuk memetakan kategori diskrit ke ruang vektor kontinu (dense vector).

Modul ini diintegrasikan sebagai komponen (layer) di dalam forward & backward pass
Neural Network utama.
"""

import math
import random


class EntityEmbedding:
    """
    Manual Entity Embedding Layer.
    
    Mengubah nilai indeks integer (kategori) menjadi vektor float.
    Misal: Kategori 'Kulkas_2_Pintu' (index 2) -> [0.45, -0.12, 0.88]
    """

    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        """
        Inisialisasi tabel bobot embedding.
        
        Args:
            vocab_size: Jumlah total kategori unik (termasuk nilai OOV jika ada).
            embedding_dim: Dimensi vektor embedding yang dihasilkan (N).
            seed: Random seed untuk reproduktibilitas.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Inisialisasi bobot (weights) embedding dengan Xavier/Glorot Initialization
        # Matriks ukuran: [vocab_size][embedding_dim]
        random.seed(seed)
        limit = math.sqrt(6.0 / (vocab_size + embedding_dim))
        
        self.weights = []
        for _ in range(vocab_size):
            row = [random.uniform(-limit, limit) for _ in range(embedding_dim)]
            self.weights.append(row)
            
        # Cache untuk menyimpan input terakhir saat forward pass
        # Dibutuhkan untuk backward pass
        self._last_inputs = None

    def forward(self, x_indices: list[int]) -> list[list[float]]:
        """
        Melakukan lookup/mapping indeks ke vektor embedding.
        
        Args:
            x_indices: List berisi indeks kategori. Panjang = batch_size.
                       Contoh batch size 3: [2, 0, 1]
                       
        Returns:
            List 2D berukuran [batch_size][embedding_dim]
        """
        self._last_inputs = x_indices
        
        output = []
        for idx in x_indices:
            # Proteksi terhadap index out-of-bound
            safe_idx = int(idx)
            if safe_idx < 0 or safe_idx >= self.vocab_size:
                # Jika out of vocab, gunakan index 0 sebagai fallback/unknown
                safe_idx = 0
                
            output.append(self.weights[safe_idx])
            
        return output

    def backward(self, d_out: list[list[float]], learning_rate: float, l2_lambda: float = 0.0):
        """
        Melakukan pembaruan bobot embedding berdasarkan gradien dari layer selanjutnya.
        
        Pada embedding, kita HANYA memperbarui baris bobot yang terpakai (diakses)
        pada forward pass terakhir (Sparse update).
        
        Args:
            d_out: Gradien error yang mengalir dari layer setelahnya.
                   Ukuran: [batch_size][embedding_dim]
            learning_rate: Kecepatan belajar.
            l2_lambda: Parameter regularisasi L2 (Weight decay) opsional.
        """
        if self._last_inputs is None:
            raise RuntimeError("Backward pass dipanggil sebelum forward pass.")
            
        batch_size = len(self._last_inputs)
        
        # Iterasi setiap sampel dalam batch
        for b_idx, cat_idx in enumerate(self._last_inputs):
            safe_idx = int(cat_idx)
            if safe_idx < 0 or safe_idx >= self.vocab_size:
                safe_idx = 0
                
            # Ambil gradien untuk sampel ini
            grad_row = d_out[b_idx]
            
            # Update baris bobot yang bersangkutan
            for i in range(self.embedding_dim):
                gradient = grad_row[i]
                
                # L2 Regularization penalty
                if l2_lambda > 0:
                    gradient += l2_lambda * self.weights[safe_idx][i]
                    
                # Pembaruan bobot (Gradient Descent)
                # Dibagi batch_size karena gradien d_out umumnya adalah rata-rata/jumlah per batch
                self.weights[safe_idx][i] -= learning_rate * gradient

    def get_weights(self) -> list[list[float]]:
        """Mengambil matriks bobot untuk disimpan (Save Model)."""
        return self.weights

    def load_weights(self, weights: list[list[float]]):
        """Memuat matriks bobot dari file (Load Model)."""
        if len(weights) != self.vocab_size or len(weights[0]) != self.embedding_dim:
            raise ValueError("Dimensi bobot tidak cocok dengan inisialisasi layer.")
        self.weights = weights


class MultiFeatureEmbedding:
    """
    Manager untuk mengelola beberapa fitur kategorikal sekaligus.
    
    Jika ada 3 fitur kategorikal, class ini akan membuat 3 layer EntityEmbedding,
    melakukan forward lookup, lalu menggabungkan (concatenate) hasilnya.
    """
    
    def __init__(self, feature_configs: list[dict], seed: int = 42):
        """
        Args:
            feature_configs: List of dictionary konfigurasi untuk tiap fitur.
                Contoh:
                [
                    {"name": "Kulkas_Kategori", "vocab_size": 5, "dim": 3},
                    {"name": "Alat_Lain_1_Jenis", "vocab_size": 8, "dim": 4}
                ]
        """
        self.feature_configs = feature_configs
        self.embeddings = {}
        self.total_output_dim = 0
        
        for cfg in feature_configs:
            name = cfg["name"]
            v_size = cfg["vocab_size"]
            dim = cfg["dim"]
            
            self.embeddings[name] = EntityEmbedding(v_size, dim, seed)
            self.total_output_dim += dim

    def forward(self, batch_categorical_data: dict[str, list[int]]) -> list[list[float]]:
        """
        Args:
            batch_categorical_data: Dictionary berisi list of batch indices per feature.
                Misal: {
                    "Kulkas_Kategori": [2, 0, 1], # batch size 3
                    "Alat_Lain_1_Jenis": [7, 2, 4]
                }
                
        Returns:
            Concatenated embedding vectors: list berukuran [batch_size][total_output_dim]
        """
        # Validasi batch size
        first_key = list(batch_categorical_data.keys())[0]
        batch_size = len(batch_categorical_data[first_key])
        
        # Buat wadah output
        concatenated_outputs = [[] for _ in range(batch_size)]
        
        # Iterasi setiap fitur yang di-embed
        for cfg in self.feature_configs:
            name = cfg["name"]
            indices = batch_categorical_data[name]
            
            # Dapatkan vektor [batch_size][dim]
            emb_vectors = self.embeddings[name].forward(indices)
            
            # Gabungkan vektor secara horizontal untuk tiap sampel
            for i in range(batch_size):
                concatenated_outputs[i].extend(emb_vectors[i])
                
        return concatenated_outputs

    def backward(self, d_out_concatenated: list[list[float]], learning_rate: float, l2_lambda: float = 0.0):
        """
        Memecah gradien gabungan kembali ke masing-masing layer embedding.
        
        Args:
            d_out_concatenated: Gradien dari layer selanjutnya berukuran [batch_size][total_output_dim]
        """
        batch_size = len(d_out_concatenated)
        current_dim_offset = 0
        
        for cfg in self.feature_configs:
            name = cfg["name"]
            dim = cfg["dim"]
            
            # Potong (slice) gradien khusus untuk fitur ini
            d_out_feature = []
            for b in range(batch_size):
                row_grad = d_out_concatenated[b]
                sliced_grad = row_grad[current_dim_offset : current_dim_offset + dim]
                d_out_feature.append(sliced_grad)
                
            # Lakukan backward pada embedding spesifik ini
            self.embeddings[name].backward(d_out_feature, learning_rate, l2_lambda)
            
            # Geser kursor offset
            current_dim_offset += dim
