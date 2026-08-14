config = {
    "prabayar": {
        "dataset_path": "data/prabayar.csv",
        "model_path": "results/prabayar/models/model_prabayar.json",
        "metrics_dir": "results/prabayar/metrics",
        "layer_sizes": None, # set dynamically: [input, 64, 32, 1]
        "hidden_layers": [128, 64],
        "learning_rate": 0.001,
        "patience": 10,
        "min_delta": 1e-5,
        "clip_value": 1.0,
        "batch_size": 16,
        "l2_lambda": 1e-2,         # tambahkan L2 — belum ada di config prabayar!
        "lr_decay": 0.001,         # tambahkan LR decay
        "target_label": "durasi token (hari)",
        "target": "Token_Habis_Dalam_Hari",
        "use_log_transform": True,
        "asymmetric_alpha": 0.6,
    },
    "pascabayar": {
        "dataset_path": "data/pascabayar.csv",
        "model_path": "results/pascabayar/models/model_pascabayar.json",
        "metrics_dir": "results/pascabayar/metrics",
        "layer_sizes": None,  # set dynamically: [input, ...hidden..., 1]
        "hidden_layers": [64],
        "learning_rate": 0.005,            
        "patience": 10,
        "min_delta": 1e-6,
        "clip_value": 10.0,
        "batch_size": 16,
        "l2_lambda": 0.01,
        "lr_decay": 0.001,
        "target_label": "estimasi biaya (Rp)",
        "target": "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp",
        "use_log_transform": True,
    },
    "pascabayar_place_value": {
        "dataset_path": "data/pascabayar.csv",
        "model_path": "results/pascabayar_place_value/models/model_pascabayar_place_value.json",
        "metrics_dir": "results/pascabayar_place_value/metrics",
        "layer_sizes": None,  # set dynamically: [input, ...hidden..., 1]
        "hidden_layers": [32, 32],
        "learning_rate": 0.05,

        "patience": 40,
        "min_delta": 1e-5,
        # nilai threshold maksimal untuk pembaruan gradient selama proses backpropagation, karena terkadang nilai gradien bisa sangat besar secara tiba tiba (exploding gradient)
        "clip_value": 1.0,
        "batch_size": 16,
        # nilai penalti untuk bobot yang terlalu besar
        "l2_lambda": 0.001,
        # nilai pengurangan learning rate
        "lr_decay": 0.001,
        "target_label": "estimasi biaya (Rp)",
        "target": "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp",
        "use_log_transform": True,
    },
    "features": {
        "prabayar": [  
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            # "Status_Subsidi_Listrik",
            "Nominal_Token_Terakhir_Rp",
            "Frekuensi_Isi_Token_Per_Bulan",

            "Kulkas_Jumlah",
            # "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            # "TV_Jumlah",
            # "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            # "AC_Jumlah",
            "AC_EstimasiJamPerHari",
            # "AC_Energi_kWhPerHari",

            # "Kipas_Jumlah",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            # "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            # "MesinCuci_Jumlah",
            # "MesinCuci_Kategori",
            # "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            # "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            # "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            # "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            # "Total_Energi_Semua_kWhPerHari",
            
            "Tarif_PLN_Eksak_Rp",
            "Estimasi_kWh_Didapat",
            # "Estimasi_Fisika_Durasi_Hari",
            "Durasi_Dari_Frekuensi",
            "Rasio_Token_vs_Energi",
            # "Token_Nominal_Kategori",
            "Energi_Per_Nominal",
            # "Fisika_vs_Frekuensi_Gap",
            "Rasio_Fisika_vs_Frekuensi",
        ],

        "pascabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Tagihan_Relatif_Stabil__Ya, relatif stabil",

            "Kulkas_Jumlah",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",

            "Total_Energi_Semua_kWhPerBulan",
            "Estimasi_Tarif_Per_kWh_Rp",

            "Estimasi_Biaya_Energi_Bulanan_Rp",
            "Daya_x_TotalEnergi",
            # "Estimasi_Fisika_Tagihan_Rp",
        ],
        "pascabayar_place_value": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            
            "Tagihan_Relatif_Stabil__Ya, relatif stabil",

            "Kulkas_Jumlah",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",

            "Total_Energi_Semua_kWhPerBulan",
            "Estimasi_Tarif_Per_kWh_Rp",

            "Estimasi_Biaya_Energi_Bulanan_Rp",
            "Daya_x_TotalEnergi",
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
        "Tarif_PLN_Eksak_Rp",
        "Estimasi_kWh_Didapat",
        "Estimasi_Fisika_Durasi_Hari",
        "Durasi_Dari_Frekuensi",
        "Rasio_Token_vs_Energi",
        "Token_Nominal_Kategori",
        "Energi_Per_Nominal",
        "Fisika_vs_Frekuensi_Gap",
        "Rasio_Fisika_vs_Frekuensi",
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
