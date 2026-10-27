# -*- coding: utf-8 -*-
"""
Clasificación de tumores cerebrales (Brain Tumor MRI Dataset, Masoud Nickparvar)
con ResNet-50 preentrenada en ImageNet.

Pensado para Google Colab / Kaggle Notebooks. Cada bloque "# %%" es una celda.

Cambios principales respecto a la versión original (que se quedaba en ~90%):
  1. Tasas de aprendizaje realistas: la original usaba 1e-6 / 1e-5 / 1e-4, que es
     tan bajo que layer3 prácticamente no se movía y en 5 épocas la red no converge.
  2. Se descongela toda la red con LR discriminativo (capas bajas aprenden menos).
  3. AdamW + weight decay + label smoothing + scheduler OneCycle (warm-up + coseno).
  4. Más épocas (20) con early stopping sobre un conjunto de VALIDACIÓN separado
     del Training. El Test solo se usa una vez al final (antes se usaba el Test para
     "elegir" el modelo, lo que da cifras optimistas).
  5. Data augmentation más adecuada para MRI y guardado del MEJOR modelo, no del último.
  6. Mixed precision (AMP) y num_workers para entrenar más rápido en GPU.
"""

# %% 1. Descarga del dataset ---------------------------------------------------
import os
import subprocess

# NO escribas tu API key en el código. En Colab usa los "Secrets" (icono de llave)
# o exporta las variables antes de ejecutar:
#   os.environ['KAGGLE_USERNAME'] = "..."; os.environ['KAGGLE_KEY'] = "..."
try:
    from google.colab import userdata  # type: ignore
    os.environ.setdefault('KAGGLE_USERNAME', userdata.get('KAGGLE_USERNAME'))
    os.environ.setdefault('KAGGLE_KEY', userdata.get('KAGGLE_KEY'))
except Exception:
    pass

DATA_DIR = './brain_tumor_data'
dataset_slug = "masoudnickparvar/brain-tumor-mri-dataset"

if not os.path.isdir(os.path.join(DATA_DIR, 'Training')):
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    print(f"Descargando {dataset_slug}...")
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_slug,
                    '--unzip', '-p', DATA_DIR], check=True)
print("Carpetas disponibles:", os.listdir(DATA_DIR))

# %% 2. Datos ---------------------------------------------------------------
import copy
import random

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_dir = os.path.join(DATA_DIR, 'Training')
test_dir = os.path.join(DATA_DIR, 'Testing')

# Dos vistas del mismo Training: una con augmentation (train) y otra sin ella (val)
full_train_aug = datasets.ImageFolder(train_dir, transform=transform_train)
full_train_eval = datasets.ImageFolder(train_dir, transform=transform_eval)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_eval)

# Split estratificado 85% train / 15% validación
targets = np.array(full_train_aug.targets)
train_idx, val_idx = [], []
for c in np.unique(targets):
    idx = np.where(targets == c)[0]
    np.random.shuffle(idx)
    n_val = int(0.15 * len(idx))
    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])

train_dataset = Subset(full_train_aug, train_idx)
val_dataset = Subset(full_train_eval, val_idx)

BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)
pin = device.type == 'cuda'

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=pin)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=pin)

classes = full_train_aug.classes
num_classes = len(classes)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Clases ({num_classes}): {classes}")

# %% 3. Modelo --------------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, num_classes),
)
model = model.to(device)

# Toda la red entrena, pero con LR discriminativo (capas bajas = LR menor).
stem = [model.conv1, model.bn1, model.layer1]
param_groups = [
    {'params': [p for m in stem for p in m.parameters()], 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 3e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 3e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
max_lrs = [g['lr'] for g in param_groups]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

NUM_EPOCHS = 20
PATIENCE = 5  # early stopping
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lrs, epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader), pct_start=0.15,
)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


# %% 4. Entrenamiento -------------------------------------------------------
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return loss_sum / total, correct / total, torch.cat(all_preds), torch.cat(all_labels)


best_val_acc, best_state, epochs_no_improve = 0.0, None, 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # OneCycle se actualiza por batch

        running_loss += loss.item() * inputs.size(0)
        correct_train += (outputs.argmax(1) == labels).sum().item()
        total_train += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss/total_train:.4f}",
                         acc=f"{correct_train/total_train*100:.2f}%")

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    val_loss, val_acc, _, _ = evaluate(val_loader)

    print(f"Época [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%", flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping: {PATIENCE} épocas sin mejorar en validación.")
            break

# %% 5. Evaluación final en Test (una sola vez, con el mejor modelo) ---------
model.load_state_dict(best_state)
test_loss, test_acc, test_preds, test_labels = evaluate(test_loader)
print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc*100:.2f}%")

# Matriz de confusión y accuracy por clase
cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
for t, p in zip(test_labels, test_preds):
    cm[t, p] += 1
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(classes)
print(cm.numpy())
for i, c in enumerate(classes):
    print(f"  {c:12s}: {cm[i, i].item() / cm[i].sum().item() * 100:.2f}%")

# %% 6. Guardar -------------------------------------------------------------
torch.save(model.state_dict(), 'brain_tumor_resnet50.pth')
print("\nModelo guardado exitosamente como 'brain_tumor_resnet50.pth'")
