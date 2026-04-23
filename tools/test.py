import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib

MODEL_FILE   = "models/gesture_model.pt"
ENCODER_FILE = "models/gesture_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.7  # below this -> show "none"

def normalize_landmarks(landmarks, width, height):
    points = np.array([[lm.x * width, lm.y * height, lm.z * width]
                       for lm in landmarks], dtype=np.float32)
    points -= points[0]
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points[1:].flatten()  # drop wrist zeros, 60 features

# ── Load model + encoder ──────────────────────────────────────────────────────
encoder   = joblib.load(ENCODER_FILE)
n_classes = len(encoder.classes_)
n_features = 60  # 20 landmarks * 3 (wrist dropped)

model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_classes),
)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

# ── Camera + MediaPipe ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

base_options = mp.tasks.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    label_text = "none"
    conf_text  = ""

    if result.hand_landmarks:
        h, w, _ = frame.shape
        lms = result.hand_landmarks[0]

        # Draw dots
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # Predict
        features = normalize_landmarks(lms, w, h)
        x = torch.tensor(features).unsqueeze(0)  # shape (1, 60)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).squeeze()

        confidence, pred_idx = probs.max(dim=0)
        confidence = confidence.item()

        if confidence >= CONFIDENCE_THRESHOLD:
            label_text = encoder.classes_[pred_idx.item()]
        conf_text = f"{confidence * 100:.0f}%"

    # HUD
    color = (0, 255, 0) if label_text != "none" else (100, 100, 100)
    cv2.putText(frame, label_text, (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
    if conf_text:
        cv2.putText(frame, conf_text, (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

    cv2.imshow("Gesture Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
