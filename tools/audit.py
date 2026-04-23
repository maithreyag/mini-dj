"""
Audit the 'none' class for rows that look like real gestures.

Workflow:
  1. Train the model:          python train_gesture.py
  2. Run this audit:           python audit_none.py
  3. Retrain on clean data:    python train_gesture.py
  4. Repeat 2-3 until clean.
"""
import csv
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import Counter

DATA_FILE = "data/gesture_data.csv"
MODEL_FILE = "models/gesture_model.pt"
ENCODER_FILE = "models/gesture_encoder.joblib"

# Flag any "none" row where the model predicts a real gesture at this confidence.
# Lower = more aggressive pruning. Start at 0.5, go lower if issues persist.
CONFLICT_THRESHOLD = 0.4

# ── Load model ────────────────────────────────────────────────────────────────
encoder = joblib.load(ENCODER_FILE)
n_classes = len(encoder.classes_)
none_class_idx = list(encoder.classes_).index("none")

model = nn.Sequential(
    nn.Linear(60, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, n_classes),
)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

# ── Load CSV ──────────────────────────────────────────────────────────────────
all_rows = []
with open(DATA_FILE, newline='') as f:
    for row in csv.reader(f):
        if row:
            all_rows.append(row)

none_indices = [i for i, row in enumerate(all_rows) if row[0] == "none"]
print(f"Total samples: {len(all_rows)}")
print(f"'none' samples to audit: {len(none_indices)}\n")

# ── Run inference on every "none" row ─────────────────────────────────────────
flagged = []  # (row_index, predicted_class, confidence)

for i in none_indices:
    features = np.array([float(v) for v in all_rows[i][1:]], dtype=np.float32)
    features = features[3:]  # drop wrist zeros (same as training)
    x = torch.tensor(features).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze()

    # Check every non-none class
    for cls_idx in range(n_classes):
        if cls_idx == none_class_idx:
            continue
        conf = probs[cls_idx].item()
        if conf >= CONFLICT_THRESHOLD:
            flagged.append((i, str(encoder.classes_[cls_idx]), conf))
            break  # one flag per row is enough

# ── Report ────────────────────────────────────────────────────────────────────
print(f"Flagged: {len(flagged)} / {len(none_indices)} 'none' samples look like a gesture\n")

if not flagged:
    print("Your 'none' data looks clean. No conflicts found.")
    raise SystemExit()

counts = Counter(cls for _, cls, _ in flagged)
for cls, n in counts.most_common():
    confs = [c for _, g, c in flagged if g == cls]
    print(f"  {cls:12s}: {n:4d} rows  (avg conf {np.mean(confs):.2f})")

# ── Prompt for removal ────────────────────────────────────────────────────────
print(f"\nRemove these {len(flagged)} rows from {DATA_FILE}? [y/N] ", end="")
choice = input().strip().lower()

if choice == "y":
    flagged_set = {i for i, _, _ in flagged}
    cleaned = [row for i, row in enumerate(all_rows) if i not in flagged_set]

    with open(DATA_FILE, 'w', newline='') as f:
        csv.writer(f).writerows(cleaned)

    print(f"\nRemoved {len(flagged)} rows. {len(cleaned)} samples remain.")
    # Show new counts
    new_counts = Counter(row[0] for row in cleaned)
    for label, n in sorted(new_counts.items()):
        print(f"  {label}: {n}")
    print("\nNow retrain:  python train_gesture.py")
else:
    print("No changes made.")
