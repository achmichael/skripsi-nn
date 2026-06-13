config = {
    "prabayar": {
        "dataset_path": "data/prabayar.csv",
        "model_path": "results/prabayar/models/model_prabayar.json",
        "metrics_dir": "results/prabayar/metrics",
        "layer_sizes": None,  # set dynamically: [input, 64, 32, 1]
        "hidden_layers": [64, 32],
        "learning_rate": 0.005,
        "patience": 5,
        "min_delta": 1e-4,
        "clip_value": 5.0,
        "batch_size": 16,
        "target_label": "durasi token (hari)",
        "target": "Token_Habis_Dalam_Hari" 
    },
    "pascabayar": {
        "dataset_path": "data/pascabayar.csv",
        "model_path": "results/pascabayar/models/model_pascabayar.json",
        "metrics_dir": "results/pascabayar/metrics",
        "layer_sizes": None,  # set dynamically: [input, ...hidden..., 1]
        "hidden_layers": [128, 64, 32],
        "learning_rate": 0.005,

        "patience": 30,
        "min_delta": 1e-5,
        "clip_value": 5.0,
        "batch_size": 16,
        "l2_lambda": 1e-3,
        "lr_decay": 0.001,
        "target_label": "estimasi biaya (Rp)",
        # "target": "Tagihan_Bulan_Terakhir_Rp",
        "target": "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp",
        "use_log_transform": False,
    },
    "features": {
        "prabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Nominal_Token_Terakhir_Rp",
            "Frekuensi_Isi_Token_Per_Bulan",

            "Kulkas_Jumlah",
            "Kulkas_Kategori",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_Kategori",
            "TV_EstimasiWattPerUnit",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_PK_Kategori",
            "AC_Kategori",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_Kategori",
            "Kipas_EstimasiWattPerUnit",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_Kategori",
            "RiceCooker_EstimasiWattPerUnit",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_Kategori",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            # Alat_Lain_1_Jenis: one-hot nominal (tidak ada urutan antar jenis alat)
            # "Alat_Lain_1_Jenis__Charger HP/perangkat kecil",
            # "Alat_Lain_1_Jenis__Blender/Mixer",
            # "Alat_Lain_1_Jenis__Setrika",
            # "Alat_Lain_1_Jenis__Dispenser",
            # "Alat_Lain_1_Jenis__Komputer/Laptop",
            # "Alat_Lain_1_Jenis__Pompa air",
            # "Alat_Lain_1_Jenis__Oven/Microwave",
            # "Alat_Lain_1_Jenis__Lainnya",
            # "Alat_Lain_1_Jumlah",
            # "Alat_Lain_1_Kategori",
            # "Alat_Lain_1_EstimasiWatt",

            # # Alat_Lain_2_Jenis: one-hot nominal
            # "Alat_Lain_2_Jenis__Charger HP/perangkat kecil",
            # "Alat_Lain_2_Jenis__Blender/Mixer",
            # "Alat_Lain_2_Jenis__Setrika",
            # "Alat_Lain_2_Jenis__Dispenser",
            # "Alat_Lain_2_Jenis__Komputer/Laptop",
            # "Alat_Lain_2_Jenis__Pompa air",
            # "Alat_Lain_2_Jenis__Oven/Microwave",
            # "Alat_Lain_2_Jenis__Lainnya",
            # "Alat_Lain_2_Jumlah",
            # "Alat_Lain_2_Kategori",
            # "Alat_Lain_2_EstimasiWatt",

            # # Alat_Lain_3_Jenis: one-hot nominal
            # "Alat_Lain_3_Jenis__Charger HP/perangkat kecil",
            # "Alat_Lain_3_Jenis__Blender/Mixer",
            # "Alat_Lain_3_Jenis__Setrika",
            # "Alat_Lain_3_Jenis__Dispenser",
            # "Alat_Lain_3_Jenis__Komputer/Laptop",
            # "Alat_Lain_3_Jenis__Pompa air",
            # "Alat_Lain_3_Jenis__Oven/Microwave",
            # "Alat_Lain_3_Jenis__Lainnya",
            # "Alat_Lain_3_Jumlah",
            # "Alat_Lain_3_Kategori",
            # "Alat_Lain_3_EstimasiWatt",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",
            
            # === ENGINEERED FEATURES ===
            "Estimasi_Fisika_Durasi_Hari",
        ],

        "pascabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            
            # --- LEAKAGE DROPPED ---
            # "Pemakaian_Bulan_Terakhir_kWh" dihapus karena membocorkan target secara langsung
            # "Pemakaian_Rata_Rata_3Bulan_kWh" dihapus karena korelasinya terlalu tinggi (0.77) dengan target
            
            # "Jumlah_Bulan_Tagihan_Terisi" dihapus — ZERO VARIANCE (seluruh data = 3)
            # "Jumlah_Bulan_kWh_Terisi" dihapus (Korelasi 1.0 dengan Jumlah_Bulan_Tagihan_Terisi)
            
            # Tagihan_Relatif_Stabil → cukup 1 kolom (2 kolom one-hot = perfectly complementary)
            "Tagihan_Relatif_Stabil__Ya, relatif stabil",

            "Kulkas_Jumlah",
            # "Kulkas_Kategori" dihapus (Korelasi >0.92 dengan Kulkas_EstimasiWattPerUnit)
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            # "TV_Kategori" dihapus (Korelasi >0.97 dengan TV_EstimasiJamPerHari)
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            # "AC_PK_Kategori" dihapus (Korelasi >0.93 dengan AC_EstimasiWattPerUnit)
            # "AC_Kategori" dihapus (Korelasi >0.96 dengan AC_EstimasiJamPerHari)
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            # "Kipas_Kategori" dihapus (Korelasi >0.97 dengan Kipas_EstimasiJamPerHari)
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            # "RiceCooker_Kategori" dihapus (Korelasi >0.97 dengan RiceCooker_EstimasiJamPerHari)
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            # "MesinCuci_Kategori" dihapus (Korelasi >0.99 dengan MesinCuci_EstimasiFrekuensiPerMinggu)
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Total_Energi_Alat_Lain_kWhPerHari",
            # "Total_Energi_Utama_kWhPerHari" dihapus (Korelasi >0.99 dengan Total_Energi_Semua_kWhPerHari)
            "Total_Energi_Semua_kWhPerHari",

            "Total_Energi_Semua_kWhPerBulan",
            "Estimasi_Tarif_Per_kWh_Rp",

            # === ENGINEERED FEATURES ===
            # Interaksi tarif × konsumsi — sinyal terkuat untuk prediksi biaya
            "Estimasi_Biaya_Energi_Bulanan_Rp",
            # Interaksi daya × konsumsi — proxy kapasitas pemakaian sebenarnya  
            "Daya_x_TotalEnergi",
            # Estimator biaya berdasarkan aturan PLN + PPJ
            "Estimasi_Fisika_Tagihan_Rp",
        ],
    },

    "numeric_cols": [
        "Jumlah_Anggota_Keluarga",
        "Daya_Listrik_Rumah_VA",

        "Kulkas_Jumlah",
        "Kulkas_EstimasiWattPerUnit",
        "Kulkas_EstimasiJamPerHari",
        "Kulkas_Energi_kWhPerHari",

        "TV_Jumlah",
        "TV_EstimasiWattPerUnit",
        "TV_EstimasiJamPerHari",
        "TV_Energi_kWhPerHari",

        "AC_Jumlah",
        "AC_EstimasiWattPerUnit",
        "AC_EstimasiJamPerHari",
        "AC_Energi_kWhPerHari",

        "Kipas_Jumlah",
        "Kipas_EstimasiWattPerUnit",
        "Kipas_EstimasiJamPerHari",
        "Kipas_Energi_kWhPerHari",

        "RiceCooker_Jumlah",
        "RiceCooker_EstimasiWattPerUnit",
        "RiceCooker_EstimasiJamPerHari",
        "RiceCooker_Energi_kWhPerHari",

        "MesinCuci_Jumlah",
        "MesinCuci_EstimasiWattPerUnit",
        "MesinCuci_EstimasiFrekuensiPerMinggu",
        "MesinCuci_EstimasiDurasiSekaliPakaiJam",
        "MesinCuci_Energi_kWhPerHari",

        "Total_Energi_Utama_kWhPerHari",
        "Total_Energi_Semua_kWhPerHari",

        "Nominal_Token_Terakhir_Rp",
        "Tagihan_Bulan_Terakhir_Rp",
        "Pemakaian_Bulan_Terakhir_kWh",
        "Tagihan_Rata_Rata_3Bulan_Rp",
        "Pemakaian_Rata_Rata_3Bulan_kWh",
        "Jumlah_Bulan_Tagihan_Terisi",
        "Jumlah_Bulan_kWh_Terisi",
        "Total_Energi_Semua_kWhPerBulan",
        "Estimasi_Tarif_Per_kWh_Rp",
        "Estimasi_Biaya_Energi_Bulanan_Rp",
        "Daya_x_TotalEnergi",
        "Estimasi_Fisika_Durasi_Hari",
        "Estimasi_Fisika_Tagihan_Rp",
    ],
    # Ordinal encoding: maps categorical string -> numeric value
    # Ordered by intensity/size so NN can learn magnitude
    "one_hot_encoding": {
        "Status_Subsidi_Listrik": {
            "Subsidi": 0,
            "Non Subsidi": 1,
        },
    },
    "ordinal_encoding": {
        "Kulkas_Kategori": {
            "Tidak ada": 0,
            "Tidak tahu": 1,
            "Kecil / 1 pintu": 2,
            "Sedang / 2 pintu": 3,
            "Besar / side by side": 4,
        },

        # Usage frequency categories (shared by TV, AC, Kipas, RiceCooker)
        "TV_Kategori": {
            "Tidak ada / tidak digunakan": 0,
            "Jarang, kurang dari 2 jam per hari": 1,
            "Sedang, sekitar 2-5 jam per hari": 2,
            "Sering, sekitar 6-10 jam per hari": 3,
            "Sangat sering, lebih dari 10 jam per hari": 4,
        },
        "AC_Kategori": {
            "Tidak ada / tidak digunakan": 0,
            "Jarang, kurang dari 2 jam per hari": 1,
            "Sedang, sekitar 2-5 jam per hari": 2,
            "Sering, sekitar 6-10 jam per hari": 3,
            "Sangat sering, lebih dari 10 jam per hari": 4,
        },
        "Kipas_Kategori": {
            "Tidak ada / tidak digunakan": 0,
            "Jarang, kurang dari 2 jam per hari": 1,
            "Sedang, sekitar 2-5 jam per hari": 2,
            "Sering, sekitar 6-10 jam per hari": 3,
            "Sangat sering, lebih dari 10 jam per hari": 4,
        },
        "RiceCooker_Kategori": {
            "Tidak ada / tidak digunakan": 0,
            "Jarang, kurang dari 2 jam per hari": 1,
            "Sedang, sekitar 2-5 jam per hari": 2,
            "Sering, sekitar 6-10 jam per hari": 3,
            "Sangat sering, lebih dari 10 jam per hari": 4,
        },

        "MesinCuci_Kategori": {
            "Tidak ada / tidak digunakan": 0,
            "Jarang, 1-2 kali per minggu": 1,
            "Sedang, 3-4 kali per minggu": 2,
            "Sering, 5-6 kali per minggu": 3,
            "Sangat sering, hampir setiap hari": 4,
        },

        "AC_PK_Kategori": {
            "Tidak ada AC": 0,
            "Tidak tahu": 1,
            "1/2 PK": 2,
            "3/4 PK": 3,
            "1 PK": 4,
            "1.5 PK": 5,
            "2 PK atau lebih": 6,
        },

        "Alat_Lain_Ada": {
            "Tidak": 0,
            "Ya": 1,
        },

        # Alat_Lain_X_Jenis dipindah ke one_hot_nominal (lihat di bawah)

        # Frequency categories for Alat_Lain
        "Alat_Lain_1_Kategori": {
            "Tidak diisi": 0,
            "Jarang, 1-2 kali per minggu": 1,
            "Sedang, 3-4 kali per minggu": 2,
            "Sering, hampir setiap hari": 3,
            "Sangat sering, setiap hari dan cukup lama": 4,
        },
        "Alat_Lain_2_Kategori": {
            "Tidak diisi": 0,
            "Jarang, 1-2 kali per minggu": 1,
            "Sedang, 3-4 kali per minggu": 2,
            "Sering, hampir setiap hari": 3,
            "Sangat sering, setiap hari dan cukup lama": 4,
        },
        "Alat_Lain_3_Kategori": {
            "Tidak diisi": 0,
            "Jarang, 1-2 kali per minggu": 1,
            "Sedang, 3-4 kali per minggu": 2,
            "Sering, hampir setiap hari": 3,
            "Sangat sering, setiap hari dan cukup lama": 4,
        },

        # Pascabayar-specific
        "Bulan_Tagihan": {
            "Tidak tahu": 0,
            "Januari": 1,
            "Februari": 2,
            "Maret": 3,
            "April": 4,
            "Mei": 5,
            "Juni": 6,
            "Juli": 7,
            "Agustus": 8,
            "September": 9,
            "Oktober": 10,
            "November": 11,
            "Desember": 12,
        },

        "Sumber_Angka_Tagihan": {
            "Perkiraan kasar": 0,
            "Mengingat dari pembayaran terakhir": 1,
            "Melihat bukti pembayaran / struk": 2,
            "Melihat rekening listrik / PLN Mobile": 3,
        },

        "Tagihan_Relatif_Stabil": {
            "Tidak tahu": 0,
            "Tidak, sering berubah": 1,
            "Ya, relatif stabil": 2,
        },
    },

    # Special numeric values for text in otherwise-numeric columns
    "numeric_special": {
        "Daya_Listrik_Rumah_VA": {
            "Tidak tahu": 900,    
            "> 5500": 7700,       
        },
    },

    # Kategori nominal (tanpa urutan) — di-expand menjadi binary columns.
    # Format kolom hasil: "<nama_kolom>__<nilai_kategori>" = 1.0 jika cocok, else 0.0.
    # Kategori baseline ("Tidak diisi") tidak dibuat kolom tersendiri (semua nol = baseline).
    "one_hot_nominal": {
        "Alat_Lain_1_Jenis": [
            "Charger HP/perangkat kecil",
            "Blender/Mixer",
            "Setrika",
            "Dispenser",
            "Komputer/Laptop",
            "Pompa air",
            "Oven/Microwave",
            "Lainnya",
        ],
        "Alat_Lain_2_Jenis": [
            "Charger HP/perangkat kecil",
            "Blender/Mixer",
            "Setrika",
            "Dispenser",
            "Komputer/Laptop",
            "Pompa air",
            "Oven/Microwave",
            "Lainnya",
        ],
        "Alat_Lain_3_Jenis": [
            "Charger HP/perangkat kecil",
            "Blender/Mixer",
            "Setrika",
            "Dispenser",
            "Komputer/Laptop",
            "Pompa air",
            "Oven/Microwave",
            "Lainnya",
        ],
    },
}
