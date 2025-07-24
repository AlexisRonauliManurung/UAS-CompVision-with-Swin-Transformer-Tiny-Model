# Klasifikasi Citra Medis menggunakan Swin Transformer Tiny

Repositori ini berisi kode untuk melatih dan mengevaluasi model **Swin Transformer Tiny** untuk tugas klasifikasi citra medis. Proyek ini menunjukkan alur kerja lengkap mulai dari pra-pemrosesan gambar, pelatihan model, hingga evaluasi performa.

-----

## 📂 Struktur Direktori

```
.
├── data
│   ├── raw
│   │   ├── image1.jpg
│   │   └── ...
│   ├── processed
│   │   ├── processed_train
│   │   ├── processed_val
│   │   └── processed_test
│   ├── images_id_kelas.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models
│   └── swin_tiny_patch4_window7_224.pth
├── early_stopping.py
├── preprocessing.py
├── SwinTransformerTiny.py
├── SwinTransformerTiny.ipynb
└── README.md
```

-----

## 📋 Deskripsi File

| Nama File                  | Deskripsi                                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `preprocessing.py`         | Skrip untuk melakukan pra-pemrosesan gambar, termasuk *CLAHE* (Contrast Limited Adaptive Histogram Equalization) dan pengubahan ukuran. Skrip ini juga membagi dataset menjadi set pelatihan, validasi, dan pengujian. |
| `SwinTransformerTiny.py`   | Skrip utama untuk melatih model Swin Transformer, melakukan evaluasi pada set pengujian, dan menyimpan bobot model yang telah dilatih. |
| `SwinTransformerTiny.ipynb`| Notebook Jupyter yang berisi eksperimen dan visualisasi langkah demi langkah dari proses yang ada di `SwinTransformerTiny.py`.    |
| `early_stopping.py`        | Modul yang berisi kelas `EarlyStopping` untuk menghentikan pelatihan jika tidak ada peningkatan pada loss validasi setelah beberapa epoch. |
| `data/`                    | Direktori yang berisi semua data, baik data mentah (`raw`), data yang telah diproses (`processed`), maupun file CSV.                   |
| `models/`                  | Direktori untuk menyimpan bobot model yang telah dilatih.                                                                              |

-----

## 🚀 Cara Menjalankan

### 1\. Persiapan Lingkungan

Pastikan Anda telah menginstal semua *library* yang dibutuhkan.

```bash
pip install torch torchvision pandas numpy opencv-python scikit-learn matplotlib seaborn tqdm
```

### 2\. Pra-pemrosesan Data

Letakkan semua gambar mentah Anda di dalam direktori `data/raw` dan siapkan file `data/images_id_kelas.csv` dengan kolom `image_id` dan `class`. Kemudian, jalankan skrip pra-pemrosesan:

```bash
python preprocessing.py
```

Skrip ini akan menghasilkan gambar yang telah diproses di dalam direktori `data/processed` dan membuat file `train.csv`, `val.csv`, dan `test.csv`.

### 3\. Pelatihan Model

Untuk memulai proses pelatihan, jalankan skrip berikut:

```bash
python SwinTransformerTiny.py
```

Skrip ini akan:

  - Memuat dataset yang telah diproses.
  - Melatih model **Swin Transformer Tiny**.
  - Menerapkan *early stopping* untuk mencegah *overfitting*.
  - Menyimpan model dengan performa terbaik ke direktori `models/`.
  - Mengevaluasi model pada data uji dan menampilkan laporan klasifikasi serta *confusion matrix*.

-----

## 📊 Hasil Eksperimen

Berikut adalah beberapa hasil yang didapatkan dari proses pelatihan dan evaluasi model.

### Perbandingan Gambar

Perbandingan antara gambar asli dan gambar yang telah melalui tahap pra-pemrosesan dengan *CLAHE*.

### Kurva Pelatihan

Grafik *loss* dan akurasi selama proses pelatihan dan validasi.

### Confusion Matrix

*Confusion matrix* dari hasil prediksi model pada set data pengujian.

