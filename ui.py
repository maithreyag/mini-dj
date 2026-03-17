import cv2
import math
import numpy as np

class Button:
    def __init__(self, x, y, width, height, color=(0, 0, 0), active_color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.active_color = active_color
        self.pinched = {"Left": False, "Right": False}
        self.on = False

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def update(self, hand, pos):
        inside = self.contains(pos)

        if inside and not self.pinched[hand]:
            self.pinched[hand] = True
            if not self.on:
                self.on = True
                self.activate()
            else:
                self.on = False
                self.deactivate()
            return True
        elif not inside:
            self.pinched[hand] = False

        return False

    def draw(self, frame):
        color = self.active_color if self.on else self.color
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            3
        )
        return frame

    def activate(self):
        pass

    def deactivate(self):
        pass


class PlayButton(Button):
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side

    def update(self, hand, pos):
        self.on = self.selector.playing[self.side]
        return super().update(hand, pos)

    def activate(self):
        self.selector.play(self.side)

    def deactivate(self):
        self.selector.pause(self.side)


class _CircleButton(Button):
    """Base for circular buttons (start, memory cue)."""
    def __init__(self, cx, cy, radius, selector, side, label="", **kwargs):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.label = label
        super().__init__(cx - radius, cy - radius, 2 * radius, 2 * radius, **kwargs)
        self.selector = selector
        self.side = side

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def draw(self, frame):
        color = self.active_color if self.on else self.color
        cv2.circle(frame, (self.cx, self.cy), self.radius, color, 3)
        return frame

    def draw_label(self, frame):
        # Center text roughly (label length varies)
        w = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(frame, self.label,
                    (self.cx - w // 2, self.cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame


class StartButton(_CircleButton):
    """Back to start: jump to 0 and stop. Every press triggers (no toggle)."""
    def __init__(self, cx, cy, radius, selector, side, **kwargs):
        super().__init__(cx, cy, radius, selector, side, label="START", **kwargs)

    def update(self, hand, pos):
        if self.contains(pos) and not self.pinched[hand]:
            self.pinched[hand] = True
            self.selector.cue(self.side)
            return True
        if not self.contains(pos):
            self.pinched[hand] = False
        return False

    def activate(self):
        self.selector.cue(self.side)


class MemoryCueButton(_CircleButton):
    """First press sets breakpoint; later presses jump back and play. Every press triggers (no toggle)."""
    def __init__(self, cx, cy, radius, selector, side, **kwargs):
        super().__init__(cx, cy, radius, selector, side, label="CUE", **kwargs)

    def update(self, hand, pos):
        if self.contains(pos) and not self.pinched[hand]:
            self.pinched[hand] = True
            self.selector.trigger_memory_cue(self.side)
            return True
        if not self.contains(pos):
            self.pinched[hand] = False
        return False

    def draw(self, frame):
        self.on = self.selector.cue_point[self.side] is not None
        return super().draw(frame)

    def activate(self):
        self.selector.trigger_memory_cue(self.side)


class StemButton(Button):
    def __init__(self, x, y, width, height, selector, side, stem_index, label="", **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side
        self.stem_index = stem_index
        self.label = label
        self.on = True  # stems start unmuted

    def activate(self):
        self.selector.unmute(self.side, self.stem_index)

    def deactivate(self):
        self.selector.mute(self.side, self.stem_index)

    def draw_label(self, frame):
        cv2.putText(frame, self.label,
                    (self.x + 4, self.y + self.height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame


class Deck:
    def __init__(self, cx, cy, radius, selector, side, label="", color=(255, 255, 255)):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.selector = selector
        self.side = side
        self.label = label
        self.color = color
        self.prev_angle = {"Left": None, "Right": None}
        self.angle = 0

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def update(self, hand, pos):
        inside = self.contains(pos)

        if inside and not self.prev_angle[hand]:
            self.prev_angle[hand] = self.calc_angle(pos)
        elif inside:
            cur_angle = self.calc_angle(pos)
            delta = cur_angle - self.prev_angle[hand]

            # correct for wraparound
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            self.prev_angle[hand] = cur_angle
            self.angle = (self.angle + delta) % (2 * math.pi)

            seek_value = 1.5 * delta

            self.selector.seek(self.side, seek_value)


        elif not inside:
            self.prev_angle[hand] = None

    def calc_angle(self, pos):
        x, y = pos
        dx = x - self.cx
        dy = y - self.cy
        return math.atan2(dy, dx)

    def draw(self, frame):
        stamp = np.zeros((2 * self.radius, 2 * self.radius, 3), dtype=np.uint8)
        cv2.circle(stamp, (self.radius, self.radius), self.radius, self.color, 2)
        if self.label:
            cv2.putText(stamp, self.label,
                        (self.radius - 10, self.radius + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
            
        degrees = math.degrees(self.angle)
        M = cv2.getRotationMatrix2D((self.radius, self.radius), -degrees, 1.0)

        rotated = cv2.warpAffine(stamp, M, (2 * self.radius, 2 * self.radius))

        x1 = self.cx - self.radius                                                                                                                                                                                
        y1 = self.cy - self.radius                                                                                                                                                                                
        x2 = self.cx + self.radius                                                                                                                                                                                
        y2 = self.cy + self.radius                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                            
        mask = rotated > 30

        frame[y1:y2, x1:x2][mask] = rotated[mask]

        return frame


class HorizontalSlider:
    """Horizontal slider. value in [0, 1]. Drag with press (state 2)."""
    def __init__(self, x, y, width, height, value=0.5, color=(100, 100, 100), thumb_color=(200, 200, 200)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.value = max(0.0, min(1.0, value))
        self.color = color
        self.thumb_color = thumb_color

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def update(self, hand, pos):
        if not self.contains(pos):
            return
        px = pos[0]
        self.value = max(0.0, min(1.0, (px - self.x) / self.width))
        self.on_value(self.value)

    def on_value(self, value):
        """Override in subclass to react to value changes."""
        pass

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, 2)
        thumb_x = int(self.x + self.value * self.width)
        thumb_cx = max(self.x + 2, min(self.x + self.width - 2, thumb_x))
        cv2.circle(frame, (thumb_cx, self.y + self.height // 2), self.height // 2 - 2, self.thumb_color, -1)
        return frame


class BPMSlider(HorizontalSlider):
    """Slider that sets deck playback rate: 0 (left) = stop, 1 (right) = 2x avg BPM."""
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, value=0.5, **kwargs)
        self.selector = selector
        self.side = side
        self.selector.set_rate(side, self.value * 2.0)

    def on_value(self, value):
        self.selector.set_rate(self.side, value * 2.0)


class VerticalSlider:
    """Vertical slider. value in [0, 1]. Bottom = 0, top = 1. extend_bottom: extra hit area below for easier mute."""
    def __init__(self, x, y, width, height, value=0.5, color=(100, 100, 100), thumb_color=(200, 200, 200), extend_bottom=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.extend_bottom = extend_bottom
        self.value = max(0.0, min(1.0, value))
        self.color = color
        self.thumb_color = thumb_color

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        bottom = self.y + self.height + self.extend_bottom
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= bottom)

    def update(self, hand, pos):
        if not self.contains(pos):
            return
        py = pos[1]
        # Clamp py to track so bottom of track (and any drag past it) gives value 0
        py_clamped = max(self.y, min(self.y + self.height, py))
        self.value = max(0.0, min(1.0, (self.y + self.height - py_clamped) / self.height))
        self.on_value(self.value)

    def on_value(self, value):
        pass

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, 2)
        thumb_y = int(self.y + (1.0 - self.value) * self.height)
        thumb_cy = max(self.y + 2, min(self.y + self.height - 2, thumb_y))
        cv2.circle(frame, (self.x + self.width // 2, thumb_cy), self.width // 2 - 2, self.thumb_color, -1)
        return frame


class VolumeSlider(VerticalSlider):
    """Vertical deck volume: bottom = mute (0), top = full (1). Only affects gain, never stops playback."""
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, value=1.0, extend_bottom=50, **kwargs)  # top = full; extend hit below so bottom = mute
        self.selector = selector
        self.side = side
        self.selector.set_deck_volume(side, self.value)

    def on_value(self, value):
        self.selector.set_deck_volume(self.side, value)


class Waveform:
    def __init__(self, x, y, width, height, selector, side, color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.selector = selector
        self.side = side
        self.color = color

    def draw(self, frame):
        waveform = self.selector.waveforms[self.side]
        if not len(waveform):
            return frame

        # Map playback position to waveform index
        duration = self.selector.get_duration(self.side)
        position = self.selector.get_position(self.side)
        if duration <= 0:
            return frame

        center_idx = int((position / duration) * len(waveform))
        half_w = self.width // 2

        # Draw bars centered on current position
        for px in range(self.width):
            idx = center_idx - half_w + px
            if idx < 0 or idx >= len(waveform):
                continue
            amp = waveform[idx]
            bar_h = int(amp * self.height)
            x = self.x + px
            y_top = self.y + self.height // 2 - bar_h // 2
            y_bot = self.y + self.height // 2 + bar_h // 2
            cv2.line(frame, (x, y_top), (x, y_bot), self.color, 1)

        # Playhead line
        cx = self.x + half_w
        cv2.line(frame, (cx, self.y), (cx, self.y + self.height), (255, 255, 255), 1)

        return frame

