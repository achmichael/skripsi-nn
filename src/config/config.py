config = {
    "prabayar": {
        "dataset_path": "data/prabayar.csv",
        "model_path": "results/prabayar/models/model_prabayar.json",
        "metrics_dir": "results/prabayar/metrics",
        "layer_sizes": None,  # set dynamically: [input, 64, 32, 1]
        "hidden_layers": [64, 32],
        "learning_rate": 0.001,
        "patience": 5,
        "min_delta": 1e-4,
        "clip_value": 5.0,
        "target_label": "durasi token (hari)",
        "target": "Token_Habis_Dalam_Hari" 
    },
    "pascabayar": {
        "dataset_path": "data/pascabayar.csv",
        "model_path": "results/pascabayar/models/model_pascabayar.json",
        "metrics_dir": "results/pascabayar/metrics",
        "layer_sizes": None,  # set dynamically: [input, 128, 64, 32, 1]
        "hidden_layers": [224, 64, 32],
        "learning_rate": 0.0005,
        "patience": 5,
        "min_delta": 1e-5,
        "clip_value": 5.0,
        "target_label": "estimasi biaya (Rp)",
        "target": "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp"
    },
    "drop_columns": [
        "Timestamp",
        "Nama/Inisial",
        "Kota/Kabupaten"
        "Bulan_Tagihan"
        "Sumber_Angka_Tagihan"
    ],
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

            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Jumlah",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_1_EstimasiWatt",

            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Jumlah",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_2_EstimasiWatt",

            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Jumlah",
            "Alat_Lain_3_Kategori",
            "Alat_Lain_3_EstimasiWatt",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",
        ],

        "pascabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Tagihan_Bulan_Terakhir_Rp",
            "Pemakaian_Bulan_Terakhir_kWh",
            "Tagihan_Rata_Rata_3Bulan_Rp",
            "Pemakaian_Rata_Rata_3Bulan_kWh",
            "Jumlah_Bulan_Tagihan_Terisi",
            "Jumlah_Bulan_kWh_Terisi",
            "Bulan_Tagihan",
            "Sumber_Angka_Tagihan",
            "Tagihan_Relatif_Stabil",

            "Kulkas_Jumlah",
            "Kulkas_Kategori",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_Kategori",
            # TV_EstimasiWattPerUnit removed: constant 80 across all rows
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
            # Kipas_EstimasiWattPerUnit removed: constant 45 across all rows
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_Kategori",
            # RiceCooker_EstimasiWattPerUnit removed: constant 300 across all rows
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_Kategori",
            # MesinCuci_EstimasiWattPerUnit removed: constant 350 across all rows
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            # MesinCuci_EstimasiDurasiSekaliPakaiJam removed: constant 1.5 across all rows
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Jumlah",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_1_EstimasiWatt",

            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Jumlah",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_2_EstimasiWatt",

            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Jumlah",
            "Alat_Lain_3_Kategori",
            "Alat_Lain_3_EstimasiWatt",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",

            # Derived features (computed from existing data, not external info)
            "Total_Energi_Semua_kWhPerBulan",
            "Estimasi_Tarif_Per_kWh_Rp",
        ],
    },

    "numeric_cols": [
        "Jumlah_Anggota_Keluarga"
        "Daya_Listrik_Rumah_VA"

        "Kulkas_Jumlah"
        "Kulkas_EstimasiWattPerUnit"
        "Kulkas_EstimasiJamPerHari"
        "Kulkas_Energi_kWhPerHari"

        "TV_Jumlah"
        "TV_EstimasiWattPerUnit"
        "TV_EstimasiJamPerHari"
        "TV_Energi_kWhPerHari"

        "AC_Jumlah"
        "AC_EstimasiWattPerUnit"
        "AC_EstimasiJamPerHari"
        "AC_Energi_kWhPerHari"

        "Kipas_Jumlah"
        "Kipas_EstimasiWattPerUnit"
        "Kipas_EstimasiJamPerHari"
        "Kipas_Energi_kWhPerHari"

        "RiceCooker_Jumlah"
        "RiceCooker_EstimasiWattPerUnit"
        "RiceCooker_EstimasiJamPerHari"
        "RiceCooker_Energi_kWhPerHari"

        "MesinCuci_Jumlah"
        "MesinCuci_EstimasiWattPerUnit"
        "MesinCuci_EstimasiFrekuensiPerMinggu"
        "MesinCuci_EstimasiDurasiSekaliPakaiJam"
        "MesinCuci_Energi_kWhPerHari"

        "Total_Energi_Utama_kWhPerHari"
        "Total_Energi_Semua_kWhPerHari"
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

        "Alat_Lain_1_Jenis": {
            "Tidak diisi": 0,
            "Charger HP/perangkat kecil": 1,
            "Blender/Mixer": 2,
            "Setrika": 3,
            "Dispenser": 4,
            "Komputer/Laptop": 5,
            "Pompa air": 6,
            "Oven/Microwave": 7,
            "Lainnya": 8,
        },
        "Alat_Lain_2_Jenis": {
            "Tidak diisi": 0,
            "Charger HP/perangkat kecil": 1,
            "Blender/Mixer": 2,
            "Setrika": 3,
            "Dispenser": 4,
            "Komputer/Laptop": 5,
            "Pompa air": 6,
            "Oven/Microwave": 7,
            "Lainnya": 8,
        },
        "Alat_Lain_3_Jenis": {
            "Tidak diisi": 0,
            "Charger HP/perangkat kecil": 1,
            "Blender/Mixer": 2,
            "Setrika": 3,
            "Dispenser": 4,
            "Komputer/Laptop": 5,
            "Pompa air": 6,
            "Oven/Microwave": 7,
            "Lainnya": 8,
        },

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
}
