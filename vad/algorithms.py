# Time-domain VAD algorithms: Energy, ZCR, and a Combined adaptive method.
# The implementations are intentionally simple and explicit for learning.

from __future__ import annotations
import numpy as np

def ema(prev, new, alpha):
    """Exponential moving average helper."""
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

class EnergyVAD:
    """
    Energy-only VAD with either a fixed threshold or an adaptive noise floor.
    Use 'mode' in {'fixed', 'adaptive'}.
    When adaptive, 'on_ratio' and 'off_ratio' specify how many times above the
    estimated noise energy we need to turn on/off speech (hysteresis).

    Hangover keeps a few frames 'on' after last detection to avoid chopping words.
    """
    def __init__(self, mode="adaptive", fixed_threshold=1e-3,
                 on_ratio=3.0, off_ratio=1.5, hangover_frames=15, ema_alpha=0.05):
        self.mode = mode
        self.fixed_threshold = float(fixed_threshold)
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray) -> np.ndarray:
        """Return 0/1 decisions per frame from STE array."""
        out = np.zeros_like(energy, dtype=np.int32)
        noise_floor = None
        on = False
        hang = 0
        for i, e in enumerate(energy):
            # update noise estimate when we're likely in non-speech OR always in adaptive mode
            if self.mode == "adaptive":
                # update toward smaller values faster by using min(e, current_estimate) logic
                if noise_floor is None:
                    noise_floor = e
                else:
                    target = min(e, noise_floor)
                    noise_floor = ema(noise_floor, target, self.ema_alpha)
                th_on = noise_floor * self.on_ratio
                th_off = noise_floor * self.off_ratio
            else:
                th_on = th_off = self.fixed_threshold

            if on:
                # stay on unless clearly below off threshold and no hangover
                if e < th_off and hang <= 0:
                    on = False
                else:
                    on = True
            else:
                if e > th_on:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)
        return out


class ZCRVAD:
    """
    ZCR-only VAD.
    Lower ZCR often correlates with voiced speech; we simply threshold max ZCR.
    For simplicity we treat 'zcr < max_allowed' as speech.
    """
    def __init__(self, zcr_max=0.12, hangover_frames=15):
        self.zcr_max = float(zcr_max)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, zcr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(zcr, dtype=np.int32)
        on = False
        hang = 0
        for i, z in enumerate(zcr):
            # ON if z is below threshold; OFF otherwise (with hangover)
            if on:
                if z > self.zcr_max and hang <= 0:
                    on = False
                else:
                    on = True
            else:
                if z < self.zcr_max:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)
        return out


class ComboVAD:
    """
    Combined VAD using energy + zcr with adaptive thresholds, hysteresis, and hangover.
    Turn ON when:
        energy > noise_floor * on_ratio  AND  zcr < zcr_max
    Turn OFF when:
        energy < noise_floor * off_ratio OR   zcr > zcr_max  (with hangover)
    """
    def __init__(self, on_ratio=3.0, off_ratio=1.5, zcr_max=0.12,
                 ema_alpha=0.05, hangover_frames=15):
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.zcr_max = float(zcr_max)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray, zcr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(energy, dtype=np.int32)
        noise_floor = None
        on = False
        hang = 0
        for i, (e, z) in enumerate(zip(energy, zcr)):
            # adaptive noise floor (track lower envelope via EMA toward min)
            if noise_floor is None:
                noise_floor = e
            else:
                target = min(e, noise_floor)
                noise_floor = (1 - self.ema_alpha) * noise_floor + self.ema_alpha * target

            th_on = noise_floor * self.on_ratio
            th_off = noise_floor * self.off_ratio

            if on:
                # OFF if clearly below energy off threshold or zcr too high (and no hangover)
                if (e < th_off or z > self.zcr_max) and hang <= 0:
                    on = False
                else:
                    on = True
            else:
                # ON requires BOTH energy high and zcr small
                if (e > th_on) and (z < self.zcr_max):
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)
        return out
