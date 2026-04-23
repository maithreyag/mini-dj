import cv2
import os
import curses
from hand_tracking.tracker import HandTracker, draw_hand_skeleton
from hand_tracking.classifier import GestureClassifier
from playback.selector import SongSelector
from playback.ui import PlayButton, StemButton, StartButton, MemoryCueButton, ResetCueButton, TempoResetButton, Deck, Waveform, BPMSlider, VolumeSlider

# Fixed display size: UI and hit-testing are always in this resolution,
# so layout looks the same on every device regardless of camera or window size.
DISPLAY_W = 1280
DISPLAY_H = 720


def pick_song(stdscr, songs, deck_name, default):
    curses.curs_set(0)
    idx = songs.index(default) if default in songs else 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Select song for {deck_name} deck  (↑↓ to scroll, Enter to confirm)\n\n")
        for i, song in enumerate(songs):
            if i == idx:
                stdscr.addstr(f"  > {song}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f"    {song}\n")
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP:
            idx = (idx - 1) % len(songs)
        elif key == curses.KEY_DOWN:
            idx = (idx + 1) % len(songs)
        elif key in (curses.KEY_ENTER, 10, 13):
            return songs[idx]


def select_songs():
    songs = sorted(s for s in os.listdir("songs") if not s.startswith("."))
    def_left  = songs[0]
    def_right = songs[1] if len(songs) > 1 else songs[0]

    def run(stdscr):
        left  = pick_song(stdscr, songs, "LEFT",  def_left)
        right = pick_song(stdscr, songs, "RIGHT", def_right)
        return left, right

    return curses.wrapper(run)

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    cv2.namedWindow('CV DJ Set', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CV DJ Set', width, height)

    def_left, def_right = select_songs()
    print(f"Left: {def_left}  |  Right: {def_right}")

    song_selector = SongSelector()
    song_selector.select("left", def_left)
    song_selector.select("right", def_right)
    song_selector.apply_bpm_sync()

    gesture_classifier = GestureClassifier()

    # Play buttons (display coords: left on left, right on right)
    py = 2 * height // 3
    left_button  = PlayButton(width // 4 - 30,      py, 100, 100, selector=song_selector, side="left")
    right_button = PlayButton(3 * width // 4 - 70,  py, 100, 100, selector=song_selector, side="right")
    cue_radius = 44
    cue_y = py + 100 + 80
    left_cue  = MemoryCueButton(width // 4 + 20,      cue_y, cue_radius, selector=song_selector, side="left")
    right_cue = MemoryCueButton(3 * width // 4 - 20,  cue_y, cue_radius, selector=song_selector, side="right")

    stem_labels = ["bass", "drm", "oth", "vox"]
    stem_size = 52
    gap = 18

    # Left stems (to the left of left play button)
    lx = width // 4 - 30 - (2 * stem_size + gap) - gap
    ly = 2 * height // 3
    left_stems = []
    all_buttons = [left_button, right_button, left_cue, right_cue]
    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        btn = StemButton(
            lx + col * (stem_size + gap), ly + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="left", stem_index=i, label=label)
        left_stems.append(btn)
        all_buttons.append(btn)

    # Right stems (to the right of right play button)
    rx = 3 * width // 4 + 30 + gap
    ry = 2 * height // 3
    right_stems = []
    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        btn = StemButton(
            rx + col * (stem_size + gap), ry + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="right", stem_index=i, label=label)
        right_stems.append(btn)
        all_buttons.append(btn)

    # Decks (top corners)
    deck_radius = height // 4
    left_deck  = Deck(deck_radius + 20,        deck_radius + 20, deck_radius, selector=song_selector, side="left",  label="L")
    right_deck = Deck(width - deck_radius - 20, deck_radius + 20, deck_radius, selector=song_selector, side="right", label="R")
    decks = [left_deck, right_deck]

    wf_height = 60
    wf_y = 2 * deck_radius + 40
    left_wf  = Waveform(left_deck.cx  - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="left")
    right_wf = Waveform(right_deck.cx - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="right")
    waveforms = [left_wf, right_wf]

    slider_w = 2 * stem_size + gap
    slider_h = 44
    slider_y_left  = ly + 2 * (stem_size + gap) + 10
    slider_y_right = ry + 2 * (stem_size + gap) + 10
    slider_y_left  = min(slider_y_left,  height - slider_h - 10)
    slider_y_right = min(slider_y_right, height - slider_h - 10)
    left_slider  = BPMSlider(lx, slider_y_left,  slider_w, slider_h, song_selector, "left")
    right_slider = BPMSlider(rx, slider_y_right, slider_w, slider_h, song_selector, "right")
    sliders = [left_slider, right_slider]

    BPM_SLOW_STEP = 0.005  # rate change per frame while peace/thumb is held

    prev_gestures = {"Left": None, "Right": None}
    print("DJ Hand Tracking Started. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.detect_async(rgb_frame)

            result = tracker.get_latest_result()

            frame = draw_hand_skeleton(frame, tracker, result)

            # Gesture inference — both hands
            gestures = gesture_classifier.classify_all(result, width, height)

            # Process gestures for each hand
            for hand in ["Left", "Right"]:
                gesture = gestures[hand]
                action, side = gesture_classifier.parse_gesture(gesture)

                # Fire one-shot actions on gesture change
                if gesture and gesture != prev_gestures[hand] and action:
                    btn = left_button if side == "left" else right_button
                    if action == "fist":
                        song_selector.pause(side)
                        btn.on = False

                # Continuous actions — fire every frame while gesture is held
                if action == "peace":
                    current_rate = song_selector.rate[side]
                    song_selector.set_rate(side, current_rate - BPM_SLOW_STEP)
                if action == "thumb":
                    current_rate = song_selector.rate[side]
                    song_selector.set_rate(side, current_rate + BPM_SLOW_STEP)

                prev_gestures[hand] = gesture

            for hand in ["Left", "Right"]:
                pinch_pos = tracker.pinch_pos[hand]
                # Flip to display coords for hit detection
                if pinch_pos:
                    pinch_pos = (width - 1 - pinch_pos[0], pinch_pos[1])

                for button in all_buttons:
                    if tracker.state[hand] == 1:
                        button.update(hand, pinch_pos)
                    else:
                        button.pinched[hand] = False

                for deck in decks:
                    if tracker.state[hand] == 1:
                        deck.update(hand, pinch_pos)
                    else:
                        deck.prev_angle[hand] = None
                for slider in sliders:
                    if tracker.state[hand] == 1:
                        slider.update(hand, pinch_pos)

            reversed_frame = cv2.flip(frame, 1)

            for button in all_buttons:
                button.draw(reversed_frame)
                if hasattr(button, 'draw_label'):
                    button.draw_label(reversed_frame)

            for deck in decks:
                deck.draw(reversed_frame)

            for wf in waveforms:
                wf.draw(reversed_frame)

            for slider in sliders:
                slider.draw(reversed_frame)

            cv2.imshow('CV DJ Set', reversed_frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        song_selector.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
