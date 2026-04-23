import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

OUTPUT_FILE = "data/gesture_data.csv"
CAPTURE_INTERVAL = 0.08  # seconds between captures while recording (~12fps)


def normalize_landmarks(landmarks, width, height):
    """
    Translate so wrist is at origin, scale by wrist->middle_finger_mcp distance.
    Makes features invariant to hand position and size in frame.
    """
    points = np.array([[lm.x * width, lm.y * height, lm.z * width]
                       for lm in landmarks])
    points -= points[0]  # wrist to origin
    scale = np.linalg.norm(points[9])  # landmark 9 = middle finger MCP
    if scale > 0:
        points /= scale
    return points.flatten()


def load_existing(path):
    if not os.path.exists(path):
        return [], {}
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    counts = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    print(f"Loaded {len(rows)} existing samples from {path}")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")
    return rows, counts


def main():
    samples, label_counts = load_existing(OUTPUT_FILE)

    current_label = input("\nEnter first gesture name: ").strip()
    print("\nControls:  R = toggle recording  |  N = new gesture  |  Q = quit & save\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tasks API — IMAGE mode runs synchronously per frame (simpler than LIVE_STREAM)
    base_options = mp.tasks.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    recording = False
    last_capture = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        hand_detected = bool(result.hand_landmarks)

        # Auto-capture while recording
        now = time.time()
        if recording and hand_detected and (now - last_capture) >= CAPTURE_INTERVAL:
            lms = result.hand_landmarks[0]
            h, w, _ = frame.shape
            features = normalize_landmarks(lms, w, h)
            samples.append([current_label] + features.tolist())
            label_counts[current_label] = label_counts.get(current_label, 0) + 1
            last_capture = now

        # Draw landmarks manually (no mp.solutions.drawing_utils available)
        if hand_detected:
            h, w, _ = frame.shape
            for lm in result.hand_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # Recording border
        if recording:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                          (0, 255, 0), 6)

        # HUD
        count = label_counts.get(current_label, 0)
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        rec_text = "REC" if recording else "PAUSED"
        rec_color = (0, 255, 0) if recording else (100, 100, 100)

        cv2.putText(frame, f"Gesture: {current_label}  [{count} samples]",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, rec_text,
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rec_color, 2)
        cv2.putText(frame, "HAND DETECTED" if hand_detected else "NO HAND",
                    (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        cv2.putText(frame, "R: record  |  N: new gesture  |  Q: quit",
                    (20, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)

        cv2.imshow("Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                print(f"  Recording '{current_label}'...")
            else:
                print(f"  Stopped. {current_label}: {label_counts.get(current_label, 0)} samples")
        elif key == ord('n'):
            recording = False
            cv2.destroyWindow("Gesture Collector")
            current_label = input("\nEnter new gesture name: ").strip()
            print(f"  Switched to '{current_label}' — press R to start recording\n")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_FILE, 'w', newline='') as f:
        csv.writer(f).writerows(samples)

    print(f"\nSaved {len(samples)} total samples to {OUTPUT_FILE}")
    print("Final counts:")
    for label, n in sorted(label_counts.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
