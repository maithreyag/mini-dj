import cv2
import math
import numpy as np

def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d=0):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom right
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    # Bottom left
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    if thickness > 0:
        # Top border
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        # Right border
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        # Bottom border
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        # Left border
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    else:
        # Fill the center if thickness is negative
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    return img

def overlay_image(background, img_overlay, x, y):
    """Overlay an image with an alpha channel onto the background."""
    if img_overlay is None: return background
    if img_overlay.shape[2] != 4:
        # No alpha channel, convert and assume no transparency
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2BGRA)
        
    h, w, _ = img_overlay.shape
    bg_h, bg_w, _ = background.shape
    
    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1_o, y2_o = max(0, -y), min(h, h - (y + h - bg_h))
    x1_o, x2_o = max(0, -x), min(w, w - (x + w - bg_w))
    
    if y1 >= y2 or x1 >= x2 or y1_o >= y2_o or x1_o >= x2_o:
        return background

    alpha_s = img_overlay[y1_o:y2_o, x1_o:x2_o, 3] / 255.0
    alpha_b = 1.0 - alpha_s

    for c in range(3):
        background[y1:y2, x1:x2, c] = (alpha_s * img_overlay[y1_o:y2_o, x1_o:x2_o, c] +
                                       alpha_b * background[y1:y2, x1:x2, c])
    return background

class Button:
    def __init__(self, x, y, width, height, color=(40, 40, 40), active_color=(235, 99, 37), icon_path=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.active_color = active_color
        self.pinched = {"Left": False, "Right": False}
        self.on = False
        
        self.icon = None
        if icon_path:
            img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                pad = 12
                scale = min((self.width - pad) / img.shape[1], (self.height - pad) / img.shape[0])
                new_w, new_h = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
                self.icon = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
        # Metallic outer chassis ring
        draw_rounded_rect(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (180, 180, 180), 2, 10)
        draw_rounded_rect(frame, (self.x + 2, self.y + 2), (self.x + self.width - 2, self.y + self.height - 2), (40, 40, 40), -1, 8)
        
        # Rubber pad center
        draw_rounded_rect(frame, (self.x + 4, self.y + 4), (self.x + self.width - 4, self.y + self.height - 4), (15, 15, 15), -1, 6)

        # Icon
        if self.icon is not None:
            ox = self.x + (self.width - self.icon.shape[1]) // 2
            oy = self.y + (self.height - self.icon.shape[0]) // 2
            overlay_image(frame, self.icon, ox, oy)

        # State Outline (illuminated LED ring)
        if self.on:
            draw_rounded_rect(frame, (self.x + 4, self.y + 4), (self.x + self.width - 4, self.y + self.height - 4), self.active_color, 2, 6)
        
        return frame

    def activate(self):
        pass

    def deactivate(self):
        pass


class PlayButton(Button):
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, icon_path="components/playbutton.png", **kwargs)
        self.selector = selector
        self.side = side

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        r = min(self.width, self.height) / 2
        return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2

    def update(self, hand, pos):
        self.on = self.selector.playing[self.side]
        return super().update(hand, pos)

    def draw(self, frame):
        # Icon only (no metallic backdrop)
        if self.icon is not None:
            ox = int(self.x + (self.width - self.icon.shape[1]) / 2)
            oy = int(self.y + (self.height - self.icon.shape[0]) / 2)
            overlay_image(frame, self.icon, ox, oy)

        # Draw a circular glowing outline only if ON
        if self.on:
            cx = int(self.x + self.width / 2)
            cy = int(self.y + self.height / 2)
            radius = int(min(self.width, self.height) / 2)
            cv2.circle(frame, (cx, cy), radius, self.active_color, 3)

        return frame

    def activate(self):
        self.selector.play(self.side)

    def deactivate(self):
        self.selector.pause(self.side)


class _CircleButton(Button):
    """Base for circular buttons (start, memory cue)."""
    def __init__(self, cx, cy, radius, selector, side, label="", color=(40, 40, 40), active_color=(235, 99, 37), **kwargs):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.label = label
        super().__init__(cx - radius, cy - radius, 2 * radius, 2 * radius, color=color, active_color=active_color, **kwargs)
        self.selector = selector
        self.side = side

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def draw(self, frame):
        # Metallic outer chassis ring
        cv2.circle(frame, (self.cx, self.cy), self.radius, (180, 180, 180), 2)
        cv2.circle(frame, (self.cx, self.cy), self.radius - 2, (40, 40, 40), -1)
        
        # Rubber pad center
        cv2.circle(frame, (self.cx, self.cy), self.radius - 4, (15, 15, 15), -1)

        # State Outline (illuminated LED ring)
        if self.on:
            cv2.circle(frame, (self.cx, self.cy), self.radius - 4, self.active_color, 2)
        
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


class TempoResetButton(Button):
    """Toggle button: press to reset deck tempo to 1x. Green when at 1x. Ignores updates when pos is in a slider (no slide-into activation)."""
    def __init__(self, x, y, width, height, selector, side, sliders=None, **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side
        self.sliders = sliders or []

    def update(self, hand, pos):
        if any(s.contains(pos) for s in self.sliders):
            return False
        self.on = abs(self.selector.rate[self.side] - 1.0) < 0.01
        return super().update(hand, pos)

    def activate(self):
        self.selector.reset_tempo(self.side)

    def deactivate(self):
        pass

    def draw_label(self, frame):
        cv2.putText(frame, "1x", (self.x + 6, self.y + self.height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame


class ResetCueButton(_CircleButton):
    """Clears the stored cue breakpoint. Every press triggers (no toggle)."""
    def __init__(self, cx, cy, radius, selector, side, **kwargs):
        super().__init__(cx, cy, radius, selector, side, label="RST", **kwargs)

    def update(self, hand, pos):
        if self.contains(pos) and not self.pinched[hand]:
            self.pinched[hand] = True
            self.selector.reset_cue_point(self.side)
            return True
        if not self.contains(pos):
            self.pinched[hand] = False
        return False

    def activate(self):
        self.selector.reset_cue_point(self.side)


class StemButton(Button):
    def __init__(self, x, y, width, height, selector, side, stem_index, label="", **kwargs):
        # Map label to image file
        icon_map = {
            "bass": "components/bass.png",
            "drm": "components/drums.png",
            "oth": "components/melody.png",
            "vox": "components/vocal.png",
        }
        icon_path = icon_map.get(label.lower(), None)
        super().__init__(x, y, width, height, icon_path=icon_path, **kwargs)
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
        # The user requested no labels under the stem buttons since the icons are clear enough.
        return frame


class Deck:
    def __init__(self, cx, cy, radius, selector, side, label="", color=(255, 255, 255)):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.selector = selector
        self.side = side
        self.label = ""  # User requested no L/R labels
        self.color = color
        self.prev_angle = {"Left": None, "Right": None}
        self.angle = 0

        # Load and resize the rotating deck image
        deck_img = cv2.imread("components/deck.png", cv2.IMREAD_UNCHANGED)
        if deck_img is not None:
            self.deck_img = cv2.resize(deck_img, (2 * radius, 2 * radius), interpolation=cv2.INTER_AREA)
        else:
            self.deck_img = None

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
        if self.deck_img is not None:
            degrees = math.degrees(self.angle)
            center = (self.radius, self.radius)
            M = cv2.getRotationMatrix2D(center, -degrees, 1.0)
            # Warp the 4-channel image 
            rotated = cv2.warpAffine(self.deck_img, M, (2 * self.radius, 2 * self.radius))
            
            ox = self.cx - self.radius
            oy = self.cy - self.radius
            overlay_image(frame, rotated, ox, oy)
        else:
            # Fallback primitive drawing
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
        # Metallic outer chassis ring
        draw_rounded_rect(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (180, 180, 180), 2, 10)
        draw_rounded_rect(frame, (self.x + 2, self.y + 2), (self.x + self.width - 2, self.y + self.height - 2), (40, 40, 40), -1, 8)
        
        # Inner track channel
        cy = self.y + self.height // 2
        cv2.rectangle(frame, (self.x + 10, cy - 3), (self.x + self.width - 10, cy + 3), (15, 15, 15), -1)

        # Fader Thumb
        tw = 16
        thumb_x = int(self.x + 10 + self.value * (self.width - 20))
        tx1 = max(self.x + 2, thumb_x - tw // 2)
        tx2 = min(self.x + self.width - 2, thumb_x + tw // 2)
        ty1 = self.y + 4
        ty2 = self.y + self.height - 4
        
        # Metallic thumb body
        draw_rounded_rect(frame, (tx1, ty1), (tx2, ty2), (200, 200, 200), -1, 3)
        # Thumb grip line
        cv2.line(frame, (thumb_x, ty1 + 4), (thumb_x, ty2 - 4), (50, 50, 50), 2)
        
        return frame


class BPMSlider(HorizontalSlider):
    """Slider that sets deck playback rate: 0 (left) = stop, 1 (right) = 2x avg BPM."""
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, value=0.5, **kwargs)
        self.selector = selector
        self.side = side
        self.selector.set_rate(side, self.value * 2.0)

    def draw(self, frame):
        self.value = self.selector.rate[self.side] / 2.0
        return super().draw(frame)

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
        # Metallic outer chassis ring
        draw_rounded_rect(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (180, 180, 180), 2, 10)
        draw_rounded_rect(frame, (self.x + 2, self.y + 2), (self.x + self.width - 2, self.y + self.height - 2), (40, 40, 40), -1, 8)
        
        # Inner track channel
        cx = self.x + self.width // 2
        cv2.rectangle(frame, (cx - 3, self.y + 10), (cx + 3, self.y + self.height - 10), (15, 15, 15), -1)

        # Fader Thumb
        th = 16
        thumb_y = int(self.y + 10 + (1.0 - self.value) * (self.height - 20))
        ty1 = max(self.y + 2, thumb_y - th // 2)
        ty2 = min(self.y + self.height - 2, thumb_y + th // 2)
        tx1 = self.x + 4
        tx2 = self.x + self.width - 4
        
        # Metallic thumb body
        draw_rounded_rect(frame, (tx1, ty1), (tx2, ty2), (200, 200, 200), -1, 3)
        # Thumb grip line
        cv2.line(frame, (tx1 + 4, thumb_y), (tx2 - 4, thumb_y), (50, 50, 50), 2)
        
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
        # Draw a metallic background screen for the waveform
        draw_rounded_rect(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (180, 180, 180), 2, 8)
        draw_rounded_rect(frame, (self.x + 2, self.y + 2), (self.x + self.width - 2, self.y + self.height - 2), (20, 20, 20), -1, 6)

        waveform = self.selector.waveforms[self.side]
        if not len(waveform):
            return frame

        # Map playback position to waveform index
        duration = self.selector.get_duration(self.side)
        position = self.selector.get_position(self.side)
        if duration <= 0:
            return frame

        center_idx = int((position / duration) * len(waveform))
        half_w = (self.width - 10) // 2

        # Draw bright green bars internally with padding
        for px in range(self.width - 10):
            idx = center_idx - half_w + px
            if idx < 0 or idx >= len(waveform):
                continue
            amp = waveform[idx]
            bar_h = int(amp * (self.height - 10))
            vx = self.x + 5 + px
            y_top = self.y + self.height // 2 - bar_h // 2
            y_bot = self.y + self.height // 2 + bar_h // 2
            cv2.line(frame, (vx, y_top), (vx, y_bot), (235, 99, 37), 1)

        # Playhead line (White)
        cx = self.x + self.width // 2
        cv2.line(frame, (cx, self.y + 4), (cx, self.y + self.height - 4), (255, 255, 255), 2)

        return frame

