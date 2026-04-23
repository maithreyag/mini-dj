import cv2
import mediapipe as mp
import time

LANDMARKS = [4, 8, 12]

MERGE_DIST = 60
UNMERGE_DIST = 80


class HandTracker:
    def __init__(self, model_path='models/hand_landmarker.task'):
        self.latest_result = None

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
            num_hands=2
        )

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time() * 1000

        self.pinch_pos = {"Left": None, "Right": None}
        self.press_pos = {"Left": None, "Right": None}
        self.state = {"Left": 0, "Right": 0}

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def get_latest_result(self):
        return self.latest_result

    def detect_async(self, frame):
        now = time.time() * 1000
        timestamp = int(now - self.start_time)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, timestamp)

    def close(self):
        self.landmarker.close()


def draw_hand_skeleton(frame, tracker, result):
    """
    Draws the hand skeleton overlay (dots and lines).
    Only handles hand visualization, not UI elements.
    """
    # Reset all hand states at start of each frame
    for hand in ["Left", "Right"]:
        tracker.pinch_pos[hand] = None
        tracker.press_pos[hand] = None
        tracker.state[hand] = 0

    if not result or not result.hand_landmarks:
        return frame

    height, width, _ = frame.shape

    for h, hand in enumerate(result.hand_landmarks):
        handedness = result.handedness[h][0].category_name

        dots = []
        state = 0
        i = 0

        while i < len(LANDMARKS):
            if i + 1 < len(LANDMARKS):
                start_idx, end_idx = LANDMARKS[i], LANDMARKS[i + 1]
                start, end = hand[start_idx], hand[end_idx]

                x1, y1 = int(start.x * width), int(start.y * height)
                x2, y2 = int(end.x * width), int(end.y * height)

                dist = abs(x2 - x1) + abs(y2 - y1)

                if (start_idx, end_idx) == (4, 8):
                    threshold = UNMERGE_DIST if state == 1 else MERGE_DIST
                else:
                    threshold = UNMERGE_DIST if state == 2 else UNMERGE_DIST

                if dist <= threshold:
                    x_mid, y_mid = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if (start_idx, end_idx) == (4, 8):
                        tracker.pinch_pos[handedness] = (x_mid, y_mid)
                        state = 1
                    else:
                        tracker.press_pos[handedness] = (x_mid, y_mid)
                        state = 2

                    cv2.circle(frame, (x_mid, y_mid), 6, (255, 255, 255), -1)
                    dots.append((x_mid, y_mid))
                    i += 2
                else:
                    cv2.circle(frame, (x1, y1), 4, (255, 255, 255), -1)
                    dots.append((x1, y1))
                    i += 1
            else:
                idx = LANDMARKS[i]
                landmark = hand[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
                dots.append((x, y))
                i += 1

        tracker.state[handedness] = state

        for i in range(len(dots) - 1):
            start, end = dots[i], dots[i + 1]
            cv2.line(frame, start, end, (255, 255, 255), 1)

    return frame
