import numpy as np
import torch
import torch.nn as nn
import joblib

MODEL_FILE = "models/gesture_model.pt"
ENCODER_FILE = "models/gesture_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.8


def _build_model(n_classes):
    """Build the gesture classifier architecture (must match train_gesture.py)."""
    return nn.Sequential(
        nn.Linear(60, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def normalize_landmarks(landmarks, width, height):
    """Normalize hand landmarks: wrist to origin, scale by middle-finger MCP distance."""
    points = np.array([[lm.x * width, lm.y * height, lm.z * width]
                       for lm in landmarks], dtype=np.float32)
    points -= points[0]
    points[:, 0] *= -1  # mirror x to match training data (collected on flipped frame)
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points[1:].flatten()  # drop wrist zeros, 60 features


class GestureClassifier:
    """Loads the trained gesture model and classifies hand landmarks."""

    def __init__(self, model_path=MODEL_FILE, encoder_path=ENCODER_FILE,
                 confidence=CONFIDENCE_THRESHOLD):
        self.confidence = confidence

        self.encoder = joblib.load(encoder_path)
        n_classes = len(self.encoder.classes_)

        self.model = _build_model(n_classes)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        print(f"Gesture classes: {list(self.encoder.classes_)}")

    def classify(self, landmarks, width, height):
        """
        Classify a single hand's landmarks.

        Returns the gesture string (e.g. "fist-r", "peace-l") or None
        if confidence is below threshold or prediction is "none".
        """
        features = normalize_landmarks(landmarks, width, height)
        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).squeeze()
        confidence, idx = probs.max(dim=0)
        if confidence.item() >= self.confidence:
            gesture = str(self.encoder.classes_[idx.item()])
            if gesture != "none":
                return gesture
        return None

    def classify_all(self, result, width, height):
        """
        Classify gestures for all detected hands.

        Returns dict: {"Left": gesture_or_None, "Right": gesture_or_None}
        """
        gestures = {"Left": None, "Right": None}
        if not result or not result.hand_landmarks:
            return gestures
        for i, handedness in enumerate(result.handedness):
            hand_name = handedness[0].category_name
            gestures[hand_name] = self.classify(
                result.hand_landmarks[i], width, height
            )
        return gestures

    @staticmethod
    def parse_gesture(gesture):
        """
        Parse a gesture string into (action, side).

        "fist-r" → ("fist", "right")
        "peace-l" → ("peace", "left")
        None or invalid → (None, None)
        """
        SIDE_MAP = {"-r": "right", "-l": "left"}
        if gesture and gesture[-2:] in SIDE_MAP:
            return gesture[:-2], SIDE_MAP[gesture[-2:]]
        return None, None
