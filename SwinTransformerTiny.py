# =============================================================================
# STEP 1: IMPORT LIBRARY YANG DIBUTUHKAN
# =============================================================================
import os
import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# =============================================================================
# STEP 2: PRA-PEMROSESAN DAN PEMBAGIAN DATASET
# =============================================================================
print("--- Memulai Pra-pemrosesan dan Pembagian Dataset ---")

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
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing to {out_dir}"):
        img_name = row.iloc[0]
        img_path = os.path.join(IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Ekstraksi green channel
        green_channel = image[:, :, 1]
        image = np.stack([green_channel] * 3, axis=-1)
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
temp_df = df.drop(train_df.index)               # Sisa data
val_df = temp_df.sample(frac=0.5, random_state=42)  # 10% data untuk val
test_df = temp_df.drop(val_df.index)                # 10% data untuk test

# Simpan split dataset ke file CSV baru
TRAIN_CSV = 'data/train.csv'
VAL_CSV = 'data/val.csv'
TEST_CSV = 'data/test.csv'
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

# --- PREPROCESS DAN SIMPAN GAMBAR SESUAI SPLIT ---
preprocess_and_save(train_df, PREP_TRAIN_DIR)
preprocess_and_save(val_df, PREP_VAL_DIR)
preprocess_and_save(test_df, PREP_TEST_DIR)

# --- Verifikasi Jumlah File ---
for split_name, df_split, out_dir in [
    ('train', train_df, PREP_TRAIN_DIR),
    ('val', val_df, PREP_VAL_DIR),
    ('test', test_df, PREP_TEST_DIR)
]:
    n_csv = len(df_split)
    n_img = len([f for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{split_name}: {n_csv} di CSV, {n_img} file gambar di {out_dir}")

print("--- Pra-pemrosesan Selesai ---")

# =============================================================================
# STEP 3: INISIALISASI DAN PELATIHAN MODEL
# =============================================================================
print("\n--- Memulai Inisialisasi dan Pelatihan Model ---")

# --- KONFIGURASI PELATIHAN ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_SAVE_PATH = 'swin_tiny_best.pth'
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0 if platform.system() == "Windows" else 2

print(f"Menggunakan device: {DEVICE}")
print(f"Path model terbaik akan disimpan di: {MODEL_SAVE_PATH}")

# --- KELAS DATASET DAN DATALOADER ---
class FundusProcessedDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# --- TRANSFORMASI DATA (AUGMENTASI & NORMALISASI) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FundusProcessedDataset(TRAIN_CSV, PREP_TRAIN_DIR, transform=train_transform)
val_dataset = FundusProcessedDataset(VAL_CSV, PREP_VAL_DIR, transform=val_test_transform)
test_dataset = FundusProcessedDataset(TEST_CSV, PREP_TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print("DataLoader dan Transformasi siap digunakan.")

# --- MEMBUAT MODEL SWIN TRANSFORMER ---
model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
in_features = model.head.in_features
model.head = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- FUNGSI KERUGIAN, OPTIMIZER, DAN SCHEDULER ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
print("Model Swin Transformer berhasil dibuat dan siap untuk training.")

# --- LOOP PELATIHAN DAN VALIDASI ---
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training", leave=False)
    for images, labels in train_loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # --- VALIDASI ---
    model.eval()
    val_loss = 0
    val_correct = 0
    all_labels_val = []
    all_preds_val = []
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation", leave=False)
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

            all_labels_val.extend(labels.cpu().numpy())
            all_preds_val.extend(predicted.cpu().numpy())

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    scheduler.step(val_acc)

    # --- EARLY STOPPING & SIMPAN MODEL TERBAIK ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model terbaik disimpan ke '{MODEL_SAVE_PATH}' dengan Val Acc: {best_val_acc:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Performa tidak meningkat selama {EARLY_STOPPING_PATIENCE} epoch. Early stopping di epoch {epoch+1}.")
            break
print("--- Pelatihan Selesai ---")

# =============================================================================
# STEP 4: EVALUASI DAN VISUALISASI HASIL
# =============================================================================
print("\n--- Memulai Evaluasi dan Visualisasi ---")

# --- VISUALISASI KURVA PELATIHAN ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.suptitle('Grafik Performa Pelatihan Model')
plt.show()

# --- EVALUASI PADA DATA TEST (MENGGUNAKAN MODEL TERBAIK) ---
# Muat bobot model terbaik yang telah disimpan
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())

print("\nEvaluasi pada data test (menggunakan model terbaik):")
class_names = [f"Severity {i}" for i in range(NUM_CLASSES)]
print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

# --- VISUALISASI CONFUSION MATRIX ---
cm_test = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.show()

# --- BUAT TABEL METRIK ---
acc = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None, zero_division=0)
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
avg_metrics = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
metrics_df.loc['Weighted Avg'] = ['Weighted Avg', avg_metrics[0], avg_metrics[1], avg_metrics[2]]
metrics_df = metrics_df.set_index('Class')

print(f"\nAkurasi Keseluruhan pada Data Test: {acc:.4f}")
print("\nTabel Metrik Performa pada Data Test:")
print(metrics_df)

# --- VISUALISASI PERBANDINGAN GAMBAR ASLI DAN PREPROCESSED ---
print("\n--- Menampilkan Contoh Hasil Pra-pemrosesan ---")
sample_df = pd.read_csv(TRAIN_CSV).sample(3, random_state=42)
plt.figure(figsize=(10, 6))
for i, row in enumerate(sample_df.itertuples()):
    raw_img = cv2.imread(os.path.join(IMG_DIR, row.image_id))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    proc_img = cv2.imread(os.path.join(PREP_TRAIN_DIR, row.image_id))
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(raw_img)
    plt.title(f"Asli: {row.image_id[:10]}...")
    plt.axis('off')
    plt.subplot(2, 3, i + 4)
    plt.imshow(proc_img)
    plt.title(f"Preprocessed: {row.image_id[:10]}...")
    plt.axis('off')
plt.suptitle('Perbandingan Citra Asli dan Hasil Pra-pemrosesan')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n--- Proses Selesai ---")