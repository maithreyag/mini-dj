import os
import sounddevice as sd
import soundfile as sf
import numpy as np

STEMS = ["bass", "drums", "other", "vocals"]

def _read_bpm(song):
    path = f"songs/{song}/bpm.txt"
    if not os.path.isfile(path):
        return 120.0
    with open(path) as f:
        return float(f.read().strip())

def _resample_stem(stem, new_len):
    """Resample stem (samples, channels) to new_len samples. Preserves pitch (time-stretch)."""
    old_len = len(stem)
    if old_len == new_len:
        return stem
    indices = np.linspace(0, old_len - 1, new_len)
    out = np.empty((new_len, stem.shape[1]), dtype=stem.dtype)
    for ch in range(stem.shape[1]):
        out[:, ch] = np.interp(indices, np.arange(old_len), stem[:, ch])
    return out

class SongSelector:
    def __init__(self, sr=44100):
        self.sr = sr
        self.stems = {"left": [], "right": []}
        self.bpm = {"left": None, "right": None}
        self.playing = {"left": False, "right": False}
        self.volumes = {"left": [1.0] * 4, "right": [1.0] * 4}
        self.rate = {"left": 1.0, "right": 1.0}
        self.position = {"left": 0.0, "right": 0.0}
        self.waveforms = {"left": [], "right": []}

        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=2,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()

    def _sample_stem_at(self, stem, pos):
        """Linear interpolation at float position. pos in [0, len-1]."""
        if pos <= 0:
            return stem[0]
        if pos >= len(stem) - 1:
            return stem[-1]
        i0, i1 = int(pos), min(int(pos) + 1, len(stem) - 1)
        t = pos - i0
        return (1 - t) * stem[i0] + t * stem[i1]

    def _callback(self, outdata, frames, time, status):
        outdata[:] = 0
        for side in ["left", "right"]:
            if not self.playing[side] or not self.stems[side]:
                continue
            rate = self.rate[side]
            if rate <= 0:
                continue
            pos = self.position[side]
            max_len = max(len(s) for s in self.stems[side])
            if pos >= max_len:
                self.playing[side] = False
                continue
            # Vectorized: compute all read positions at once
            read_positions = pos + np.arange(frames, dtype=np.float64) * rate
            valid = read_positions < max_len
            if not np.any(valid):
                self.playing[side] = False
                continue
            rp = read_positions[valid]
            i0 = np.floor(rp).astype(np.int64)
            t = (rp - i0).astype(np.float32)
            n_valid = len(rp)
            for i, stem_data in enumerate(self.stems[side]):
                vol = self.volumes[side][i]
                if vol == 0.0:
                    continue
                slen = len(stem_data)
                mask = i0 < slen - 1
                idx = np.minimum(i0[mask], slen - 2)
                frac = t[mask]
                samples = stem_data[idx] * (1 - frac[:, None]) + stem_data[idx + 1] * frac[:, None]
                # Write into the valid portion of outdata
                out_slice = outdata[:n_valid]
                out_slice[mask] += samples * vol
            self.position[side] = pos + frames * rate
        outdata *= 0.5
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def play(self, side):
        self.playing[side] = True

    def pause(self, side):
        self.playing[side] = False

    def set_rate(self, side, rate):
        self.rate[side] = max(0.0, float(rate))

    def cue(self, side):
        self.position[side] = 0.0

    def mute(self, side, stem_index):
        self.volumes[side][stem_index] = 0.0

    def unmute(self, side, stem_index):
        self.volumes[side][stem_index] = 1.0

    def select(self, side, song):
        self.playing[side] = False
        self.bpm[side] = _read_bpm(song)
        loaded = []
        for stem in STEMS:
            data, sr = sf.read(f"songs/{song}/{stem}.mp3", dtype='float32')
            if data.ndim == 1:
                data = np.column_stack([data, data])
            loaded.append(data)

        self.stems[side] = loaded
        self.position[side] = 0.0
        self._build_waveform(side)

    def _build_waveform(self, side):
        if not self.stems[side]:
            self.waveforms[side] = []
            return
        combined = sum(stem[:, 0] for stem in self.stems[side])
        chunk_size = 1000
        self.waveforms[side] = [np.max(np.abs(combined[i:i+chunk_size])) for i in range(0, len(combined), chunk_size)]

    def apply_bpm_sync(self):
        """Resample both decks so they play at average BPM. Call after both select()s."""
        if self.bpm["left"] is None or self.bpm["right"] is None:
            return
        avg_bpm = (self.bpm["left"] + self.bpm["right"]) / 2.0
        for side in ["left", "right"]:
            if not self.stems[side]:
                continue
            ratio = self.bpm[side] / avg_bpm
            resampled = []
            for stem in self.stems[side]:
                new_len = int(len(stem) * ratio)
                resampled.append(_resample_stem(stem, new_len))
            self.stems[side] = resampled
            self.position[side] = 0.0
            self._build_waveform(side)

    def seek(self, side, ds):
        self.position[side] += ds * self.sr
        max_len = max(len(s) for s in self.stems[side]) if self.stems[side] else 0
        self.position[side] = max(0.0, min(self.position[side], max_len - 1e-6))

    def get_position(self, side):
        if not self.stems[side]:
            return 0.0
        return float(self.position[side]) / self.sr

    def get_duration(self, side):
        if not self.stems[side]:
            return 0.0
        return max(len(s) for s in self.stems[side]) / self.sr

    def close(self):
        self.stream.stop()
        self.stream.close()
