import cv2
from hand_tracker import HandTracker, draw_hand_skeleton
from song_selector import SongSelector
from ui import PlayButton, StemButton, StartButton, MemoryCueButton, ResetCueButton, TempoResetButton, Deck, Waveform, BPMSlider, VolumeSlider

# Fixed display size: UI and hit-testing are always in this resolution,
# so layout looks the same on every device regardless of camera or window size.
DISPLAY_W = 1280
DISPLAY_H = 720

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    width, height = DISPLAY_W, DISPLAY_H

    def_left = "jane"
    def_right = "dougie"

    song_selector = SongSelector()
    song_selector.select("left", def_left)
    song_selector.select("right", def_right)
    song_selector.apply_bpm_sync()

    stem_labels = ["bass", "drm", "oth", "vox"]
    stem_size = 70
    gap = 40
    section_gap = 20  # vertical gap between sections so nothing overlaps
    slider_h = 44
    slider_gap = 24

    # Vertical stack (top to bottom) with fixed gaps so waveform, stems, slider never overlap
    deck_radius = 140
    deck_cy = 20 + deck_radius
    deck_bottom = deck_cy + deck_radius

    wf_height = 60
    wf_y = deck_bottom + section_gap
    wf_bottom = wf_y + wf_height

    stem_block_h = 2 * (stem_size + gap)
    ly = wf_bottom + section_gap
    ry = ly

    slider_y = ly + stem_block_h + slider_gap

    # Decks (top corners)
    left_deck = Deck(deck_radius + 20, deck_cy, deck_radius, selector=song_selector, side="left", label="L")
    right_deck = Deck(width - deck_radius - 20, deck_cy, deck_radius, selector=song_selector, side="right", label="R")
    decks = [left_deck, right_deck]

    left_wf = Waveform(left_deck.cx - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="left")
    right_wf = Waveform(right_deck.cx - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="right")
    waveforms = [left_wf, right_wf]

    lx = width // 4 - 30 - (2 * stem_size + gap) - gap
    rx = 3 * width // 4 + 30 + gap
    slider_w = 2 * stem_size + gap

    py = ly + (stem_block_h - 100) // 2
    left_button = PlayButton(width // 4 - 30, py, 100, 100, selector=song_selector, side="left")
    right_button = PlayButton(3 * width // 4 - 70, py, 100, 100, selector=song_selector, side="right")
    cue_radius = 44
    cue_y = py + 100 + 80
    cue_spacing = 2 * cue_radius + 12  # center-to-center so circles don't overlap (radius 44 → need 88+ gap)
    slider_cue_gap = 50  # gap between slider edge and circles
    # Position circles: both sides symmetric — START (outer), CUE, RST (inner toward center)
    left_start_cx = lx + slider_w + slider_cue_gap + cue_radius
    left_cue_cx = left_start_cx + cue_spacing
    left_reset_cx = left_cue_cx + cue_spacing
    right_start_cx = rx - slider_cue_gap - cue_radius
    right_cue_cx = right_start_cx - cue_spacing
    right_reset_cx = right_cue_cx - cue_spacing
    left_start = StartButton(left_start_cx, cue_y, cue_radius, song_selector, "left")
    left_cue = MemoryCueButton(left_cue_cx, cue_y, cue_radius, song_selector, "left")
    right_start = StartButton(right_start_cx, cue_y, cue_radius, song_selector, "right")
    right_cue = MemoryCueButton(right_cue_cx, cue_y, cue_radius, song_selector, "right")
    left_reset = ResetCueButton(left_reset_cx, cue_y, cue_radius, song_selector, "left")
    right_reset = ResetCueButton(right_reset_cx, cue_y, cue_radius, song_selector, "right")
    tempo_btn_w, tempo_btn_h = 48, 36
    tempo_btn_gap = 8
    tempo_btn_y = slider_y + (slider_h - tempo_btn_h) // 2
    buttons = [left_button, right_button, left_start, left_cue, left_reset, right_start, right_cue, right_reset]

    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        buttons.append(StemButton(
            lx + col * (stem_size + gap), ly + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="left", stem_index=i, label=label))

    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        buttons.append(StemButton(
            rx + col * (stem_size + gap), ry + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="right", stem_index=i, label=label))

    left_slider = BPMSlider(lx, slider_y, slider_w, slider_h, song_selector, "left")
    right_slider = BPMSlider(rx, slider_y, slider_w, slider_h, song_selector, "right")
    # Volume slider: vertical, same area as BPM slider; well above circles so no overlap
    vol_slider_w, vol_slider_h = slider_h, slider_w
    vol_gap = 10
    vol_x_inset = 35
    vol_y_inset = 160  # well above circles so dragging to bottom of slider doesn’t hit CUE/START
    left_vol_x = width // 4 - 30 + 100 + vol_gap + vol_x_inset
    right_vol_x = 3 * width // 4 - 70 - vol_slider_w - vol_gap - vol_x_inset
    vol_y = py - vol_y_inset
    left_vol = VolumeSlider(left_vol_x, vol_y, vol_slider_w, vol_slider_h, song_selector, "left")
    right_vol = VolumeSlider(right_vol_x, vol_y, vol_slider_w, vol_slider_h, song_selector, "right")
    sliders = [left_slider, right_slider, left_vol, right_vol]
    left_tempo = TempoResetButton(lx - tempo_btn_w - tempo_btn_gap, tempo_btn_y, tempo_btn_w, tempo_btn_h, song_selector, "left", sliders=sliders)
    right_tempo = TempoResetButton(rx + slider_w + tempo_btn_gap, tempo_btn_y, tempo_btn_w, tempo_btn_h, song_selector, "right", sliders=sliders)
    buttons.extend([left_tempo, right_tempo])

    cv2.namedWindow("CV DJ Set", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CV DJ Set", DISPLAY_W, DISPLAY_H)

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

            for hand in ["Left", "Right"]:
                pinch_pos = tracker.pinch_pos[hand]
                # Flip to display coords for hit detection
                if pinch_pos:
                    pinch_pos = (width - 1 - pinch_pos[0], pinch_pos[1])

                for button in buttons:
                    if tracker.state[hand] == 1:
                        button.update(hand, pinch_pos)
                    else:
                        button.pinched[hand] = False

                press_pos = tracker.press_pos[hand]
                if press_pos:
                    press_pos = (width - 1 - press_pos[0], press_pos[1])
                for deck in decks:
                    if tracker.state[hand] == 2:
                        deck.update(hand, press_pos)
                    else:
                        deck.prev_angle[hand] = None
                for slider in sliders:
                    if tracker.state[hand] == 1:
                        slider.update(hand, pinch_pos)

            reversed_frame = cv2.flip(frame, 1)

            for button in buttons:
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
