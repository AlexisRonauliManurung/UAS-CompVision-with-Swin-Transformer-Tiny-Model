import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- KONFIGURASI PATH DATA ---
IMG_DIR = 'data/raw'  # Direktori gambar input
CSV_PATH = 'data/images_id_kelas.csv'  # Path file CSV berisi nama file gambar dan label
PREP_TRAIN_DIR = 'data/processed/processed_train'  # Direktori output gambar train setelah preprocessing
PREP_VAL_DIR = 'data/processed/processed_val'      # Direktori output gambar val setelah preprocessing
PREP_TEST_DIR = 'data/processed/processed_test'    # Direktori output gambar test setelah preprocessing

# Membuat direktori output jika belum ada
os.makedirs(PREP_TRAIN_DIR, exist_ok=True)
os.makedirs(PREP_VAL_DIR, exist_ok=True)
os.makedirs(PREP_TEST_DIR, exist_ok=True)

def apply_clahe(img):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pada channel L gambar dalam ruang warna LAB untuk meningkatkan kontras.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def apply_sharpening(img):
    """
    Menerapkan filter sharpening pada gambar untuk menajamkan detail.
    """
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_and_save(df, out_dir):
    """
    Melakukan preprocessing pada gambar sesuai dataframe df dan menyimpan hasilnya ke out_dir.
    Tahapan: ekstraksi green channel, CLAHE, sharpening, resize, simpan.
    """
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing to {out_dir}"):
        img_name = row[0]
        img_path = os.path.join(IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Ekstraksi green channel
        green_channel = image[:, :, 1]
        image = np.stack([green_channel]*3, axis=-1)
        # CLAHE untuk meningkatkan kontras
        image = apply_clahe(image)
        # Sharpening untuk menajamkan gambar
        image = apply_sharpening(image)
        # Resize ke 224x224 pixel
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        # Simpan hasil preprocessing ke direktori output
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# --- SPLIT DATASET (80% train, 10% val, 10% test) ---
df = pd.read_csv(CSV_PATH)  # Membaca file CSV
train_df = df.sample(frac=0.8, random_state=42)  # 80% data untuk train
temp_df = df.drop(train_df.index)                # Sisa data
val_df = temp_df.sample(frac=0.5, random_state=42)  # 10% data untuk val
test_df = temp_df.drop(val_df.index)                # 10% data untuk test

# Simpan split dataset ke file CSV
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# --- PREPROCESS DAN SIMPAN ---
# Proses dan simpan gambar hasil preprocessing untuk masing-masing split
preprocess_and_save(train_df, PREP_TRAIN_DIR)
preprocess_and_save(val_df, PREP_VAL_DIR)
preprocess_and_save(test_df, PREP_TEST_DIR)

# Cek jumlah file hasil preprocessing vs jumlah baris di CSV split
for split_name, df_split, out_dir in [
    ('train', train_df, PREP_TRAIN_DIR),
    ('val', val_df, PREP_VAL_DIR),
    ('test', test_df, PREP_TEST_DIR)
]:
    n_csv = len(df_split)
    n_img = len([f for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{split_name}: {n_csv} di CSV, {n_img} file gambar di {out_dir}")