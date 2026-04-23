import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

DATA_FILE = "data/gesture_data.csv"
MODEL_FILE = "models/gesture_model.pt"
ENCODER_FILE = "models/gesture_encoder.joblib"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ── Load CSV ──────────────────────────────────────────────────────────────────
rows = []
with open(DATA_FILE, newline='') as f:
    for row in csv.reader(f):
        if row:
            rows.append(row)

labels = [row[0] for row in rows]
features = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
# drop the first 3 columns (wrist x,y,z — always 0 after normalization)
features = features[:, 3:]

encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.int64)

print(f"Classes: {list(encoder.classes_)}")
print(f"Total samples: {len(y)}")
for i, cls in enumerate(encoder.classes_):
    print(f"  {cls}: {np.sum(y == i)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain samples: {len(X_train)}  |  Test samples: {len(X_test)}")

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"Batches per epoch: {len(train_dl)}  (batch size {BATCH_SIZE})")

# ── Model ─────────────────────────────────────────────────────────────────────
n_features = features.shape[1]  # 60 (21 landmarks * 3 - 3 wrist zeros)
n_classes  = len(encoder.classes_)

print(f"\nModel:  {n_features} inputs  →  64  →  64  →  {n_classes} outputs")

model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_classes),
)

counts = np.bincount(y)
weights = 1.0 / counts.astype(np.float32)
weights /= weights.sum()
print(f"\nClass weights (inverse frequency):")
for i, cls in enumerate(encoder.classes_):
    print(f"  {cls}: {weights[i]:.4f}  ({counts[i]} samples)")

loss_fn   = nn.CrossEntropyLoss(weight=torch.tensor(weights))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1:3d}/{EPOCHS}  loss: {avg_loss:.4f}")

print("Training complete.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    X_t = torch.tensor(X_test)
    logits = model(X_t)
    y_pred = logits.argmax(dim=1).numpy()

acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc * 100:.1f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(f"  {'':14}", "  ".join(f"{c:14}" for c in encoder.classes_))
for i, row in enumerate(cm):
    print(f"  {encoder.classes_[i]:14}", "  ".join(f"{v:14}" for v in row))

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)
print(f"\nSaved model to {MODEL_FILE}")
print(f"Saved encoder to {ENCODER_FILE}")
