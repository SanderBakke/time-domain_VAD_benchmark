# vad/algorithms.py
from __future__ import annotations
import numpy as np


def ema(prev, new, alpha):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new


class EnergyVAD:
    def __init__(self, mode="adaptive", fixed_threshold=1e-3,
                 on_ratio=3.0, off_ratio=1.5, hangover_frames=15, ema_alpha=0.05):
        self.mode = mode
        self.fixed_threshold = float(fixed_threshold)
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray) -> np.ndarray:
        out = np.zeros_like(energy, dtype=np.int32)
        noise_floor = None
        on = False
        hang = 0

        for i, e in enumerate(energy):
            if self.mode == "adaptive":
                if noise_floor is None:
                    noise_floor = float(e)
                else:
                    target = min(float(e), float(noise_floor))
                    noise_floor = (1 - self.ema_alpha) * float(noise_floor) + self.ema_alpha * float(target)
                th_on = float(noise_floor) * self.on_ratio
                th_off = float(noise_floor) * self.off_ratio
            else:
                th_on = th_off = self.fixed_threshold

            if on:
                if float(e) < th_off and hang <= 0:
                    on = False
            else:
                if float(e) > th_on:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class ZCRVAD:
    def __init__(self, zcr_max=0.12, hangover_frames=15):
        self.zcr_max = float(zcr_max)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, zcr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(zcr, dtype=np.int32)
        on = False
        hang = 0

        for i, z in enumerate(zcr):
            if on:
                if float(z) > self.zcr_max and hang <= 0:
                    on = False
            else:
                if float(z) < self.zcr_max:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class EnergyZCRVAD:
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
            e = float(e)
            z = float(z)

            if noise_floor is None:
                noise_floor = e
            else:
                target = min(e, float(noise_floor))
                noise_floor = (1 - self.ema_alpha) * float(noise_floor) + self.ema_alpha * float(target)

            th_on = float(noise_floor) * self.on_ratio
            th_off = float(noise_floor) * self.off_ratio

            if on:
                if (e < th_off or z > self.zcr_max) and hang <= 0:
                    on = False
            else:
                if (e > th_on) and (z < self.zcr_max):
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class EnergyLogVarVAD:
    def __init__(self, on_ratio=3.0, off_ratio=1.5,
                 v_on=0.06, v_off=0.04,
                 ema_alpha=0.05, hangover_frames=15):
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.v_on = float(v_on)
        self.v_off = float(v_off)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        out = np.zeros_like(energy, dtype=np.int32)
        noise_floor = None
        on = False
        hang = 0

        for i, (e, v) in enumerate(zip(energy, logvar)):
            e = float(e)
            v = float(v)

            if noise_floor is None:
                noise_floor = e
            else:
                target = min(e, float(noise_floor))
                noise_floor = (1 - self.ema_alpha) * float(noise_floor) + self.ema_alpha * float(target)

            th_on = float(noise_floor) * self.on_ratio
            th_off = float(noise_floor) * self.off_ratio

            if on:
                if (e < th_off or v < self.v_off) and hang <= 0:
                    on = False
            else:
                if (e > th_on) and (v > self.v_on):
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class EnergyZERVAD:
    def __init__(self, on_ratio=3.0, off_ratio=1.5,
                 zer_on=0.10, zer_off=0.14,
                 ema_alpha=0.05, hangover_frames=15):
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.zer_on = float(zer_on)
        self.zer_off = float(zer_off)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray, zer: np.ndarray) -> np.ndarray:
        out = np.zeros_like(energy, dtype=np.int32)
        noise_floor = None
        on = False
        hang = 0

        for i, (e, z) in enumerate(zip(energy, zer)):
            e = float(e)
            z = float(z)

            if noise_floor is None:
                noise_floor = e
            else:
                target = min(e, float(noise_floor))
                noise_floor = (1 - self.ema_alpha) * float(noise_floor) + self.ema_alpha * float(target)

            th_on = float(noise_floor) * self.on_ratio
            th_off = float(noise_floor) * self.off_ratio

            if on:
                if (e < th_off or z > self.zer_off) and hang <= 0:
                    on = False
            else:
                if (e > th_on) and (z < self.zer_on):
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class ZRMSEVAD:
    def __init__(self, k=2.0, gamma=1.0, hangover_frames=15, eps=1e-5):
        self.k = float(k)
        self.gamma = float(gamma)
        self.hangover_frames = int(hangover_frames)
        self.eps = float(eps)

    def predict_frames(self, energy: np.ndarray, zcr: np.ndarray) -> np.ndarray:
        zrmse = np.sqrt(np.asarray(energy, np.float32) / (np.asarray(zcr, np.float32) + self.eps))
        scores = 1.0 / (1.0 + np.exp(-self.k * (zrmse - self.gamma)))
        out = np.zeros_like(scores, dtype=np.int32)

        on = False
        hang = 0
        for i, p in enumerate(scores):
            p = float(p)
            if on:
                if p < 0.5 and hang <= 0:
                    on = False
            else:
                if p > 0.5:
                    on = True
                    hang = self.hangover_frames
            if on:
                out[i] = 1
                hang = max(hang - 1, 0)
        return out


class NRMSEVAD:
    def __init__(self, z_on=3.0, z_off=1.5, ema_alpha=0.05,
                 hangover_frames=15, eps=1e-5, init_frames=10):
        self.z_on = float(z_on)
        self.z_off = float(z_off)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)
        self.eps = float(eps)
        self.init_frames = int(init_frames)

    def predict_frames(self, rms: np.ndarray) -> np.ndarray:
        rms = np.asarray(rms, dtype=np.float32)
        out = np.zeros_like(rms, dtype=np.int32)

        mu = None
        var = None
        on = False
        hang = 0

        for i, r in enumerate(rms):
            r = float(r)
            if mu is None:
                mu = r
                var = 0.0

            sigma = float(np.sqrt(max(var, 0.0)) + self.eps)
            z = (r - mu) / sigma if sigma > 0 else 0.0

            noise_like = (not on) and (z < self.z_off)
            if noise_like or (i < self.init_frames):
                mu_new = (1 - self.ema_alpha) * mu + self.ema_alpha * r
                m2 = var + mu * mu
                m2_new = (1 - self.ema_alpha) * m2 + self.ema_alpha * (r * r)
                var_new = max(0.0, m2_new - mu_new * mu_new)
                mu, var = mu_new, var_new

                sigma = float(np.sqrt(max(var, 0.0)) + self.eps)
                z = (r - mu) / sigma if sigma > 0 else 0.0

            if on:
                if z < self.z_off and hang <= 0:
                    on = False
            else:
                if z > self.z_on:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class ZERVAD:
    def __init__(self, threshold=0.1, eps=0.02, hangover_frames=15):
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float32)
        zer = np.mean(np.abs(frames) < self.eps, axis=1).astype(np.float32)

        out = np.zeros_like(zer, dtype=np.int32)
        on = False
        hang = 0

        for i, z in enumerate(zer):
            z = float(z)
            if on:
                if z > self.threshold and hang <= 0:
                    on = False
            else:
                if z < self.threshold:
                    on = True
                    hang = self.hangover_frames
            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class PARVAD:
    def __init__(self, threshold=5.0, hangover_frames=15):
        self.threshold = float(threshold)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float32)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        peak = np.max(np.abs(frames), axis=1)
        par = peak / (rms + 1e-12)

        out = np.zeros_like(par, dtype=np.int32)
        on = False
        hang = 0

        for i, p in enumerate(par):
            p = float(p)
            if on:
                if p < self.threshold and hang <= 0:
                    on = False
            else:
                if p > self.threshold:
                    on = True
                    hang = self.hangover_frames
            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


class LogVarVAD:
    def __init__(self, window=5, threshold=0.05, hangover_frames=15):
        self.window = int(window)
        self.threshold = float(threshold)
        self.hangover_frames = int(hangover_frames)

    def predict_frames(self, energy: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy, dtype=np.float32)
        logE = np.log(energy + 1e-12)

        var_logE = np.zeros_like(logE, dtype=np.float32)
        for i in range(len(logE)):
            start = max(0, i - self.window // 2)
            end = min(len(logE), i + self.window // 2 + 1)
            var_logE[i] = float(np.var(logE[start:end], ddof=0))

        out = np.zeros_like(var_logE, dtype=np.int32)
        on = False
        hang = 0

        for i, v in enumerate(var_logE):
            v = float(v)
            if on:
                if v < self.threshold and hang <= 0:
                    on = False
            else:
                if v > self.threshold:
                    on = True
                    hang = self.hangover_frames
            if on:
                out[i] = 1
                hang = max(hang - 1, 0)

        return out


# --------------------------
# Frequency-domain algorithms
# --------------------------

class BandSNRVAD:
    def __init__(self, thr_on_db=6.0, thr_off_db=4.0, ema_alpha=0.05, hangover_frames=15, eps=1e-12):
        self.thr_on_db = float(thr_on_db)
        self.thr_off_db = float(thr_off_db)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)
        self.eps = float(eps)
        self.last_statistic = None

    def predict_frames(self, Eb: np.ndarray) -> np.ndarray:
        Eb = np.asarray(Eb, dtype=np.float32)
        M, B = Eb.shape
        out = np.zeros(M, dtype=np.int32)
        T = np.zeros(M, dtype=np.float32)

        noise = None
        on = False
        hang = 0

        for m in range(M):
            e = Eb[m]
            if noise is None:
                noise = e.copy()
            else:
                target = np.minimum(e, noise)
                noise = (1 - self.ema_alpha) * noise + self.ema_alpha * target

            snr_lin = e / (noise + self.eps)
            t_db = 10.0 * np.log10(np.mean(snr_lin) + self.eps)
            T[m] = float(t_db)

            if on:
                if (t_db < self.thr_off_db) and hang <= 0:
                    on = False
            else:
                if t_db > self.thr_on_db:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[m] = 1
                hang = max(hang - 1, 0)

        self.last_statistic = T
        return out


class LSFMVAD:
    def __init__(self, window_frames=5, thr_on=0.55, thr_off=0.60, hangover_frames=15):
        self.window_frames = int(window_frames)
        self.thr_on = float(thr_on)
        self.thr_off = float(thr_off)
        self.hangover_frames = int(hangover_frames)
        self.last_statistic = None

    def predict_frames(self, sfm: np.ndarray) -> np.ndarray:
        sfm = np.asarray(sfm, dtype=np.float32)
        M = sfm.shape[0]
        out = np.zeros(M, dtype=np.int32)
        T = np.zeros(M, dtype=np.float32)

        L = max(self.window_frames, 1)
        buf = np.zeros(L, dtype=np.float32)
        n = 0
        idx = 0

        on = False
        hang = 0

        for m in range(M):
            buf[idx] = sfm[m]
            idx = (idx + 1) % L
            n = min(n + 1, L)

            lsfm = float(buf[:n].mean())
            T[m] = lsfm

            if on:
                if (lsfm > self.thr_off) and hang <= 0:
                    on = False
            else:
                if lsfm < self.thr_on:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[m] = 1
                hang = max(hang - 1, 0)

        self.last_statistic = T
        return out


class LTSVVAD:
    def __init__(self, window_frames=5, thr_on=0.20, thr_off=0.15, hangover_frames=15, eps=1e-12):
        self.window_frames = int(window_frames)
        self.thr_on = float(thr_on)
        self.thr_off = float(thr_off)
        self.hangover_frames = int(hangover_frames)
        self.eps = float(eps)
        self.last_statistic = None

    def predict_frames(self, Eb: np.ndarray) -> np.ndarray:
        Eb = np.asarray(Eb, dtype=np.float32)
        M, B = Eb.shape
        out = np.zeros(M, dtype=np.int32)
        T = np.zeros(M, dtype=np.float32)

        L = max(self.window_frames, 2)
        buf = np.zeros((L, B), dtype=np.float32)
        n = 0
        idx = 0

        on = False
        hang = 0

        for m in range(M):
            logE = np.log(Eb[m] + self.eps).astype(np.float32)
            buf[idx] = logE
            idx = (idx + 1) % L
            n = min(n + 1, L)

            X = buf[:n]
            v = float(np.var(X, axis=0, ddof=0).mean())
            T[m] = v

            if on:
                if (v < self.thr_off) and hang <= 0:
                    on = False
            else:
                if v > self.thr_on:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[m] = 1
                hang = max(hang - 1, 0)

        self.last_statistic = T
        return out


class LTSDVAD:
    def __init__(self, window_frames=5, thr_on_db=6.0, thr_off_db=4.0, ema_alpha=0.05,
                 hangover_frames=15, eps=1e-12):
        self.window_frames = int(window_frames)
        self.thr_on_db = float(thr_on_db)
        self.thr_off_db = float(thr_off_db)
        self.ema_alpha = float(ema_alpha)
        self.hangover_frames = int(hangover_frames)
        self.eps = float(eps)
        self.last_statistic = None

    def predict_frames(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float32)
        M, K = P.shape
        out = np.zeros(M, dtype=np.int32)
        T = np.zeros(M, dtype=np.float32)

        L = max(self.window_frames, 1)
        buf = np.zeros((L, K), dtype=np.float32)
        n = 0
        idx = 0

        noise = None
        on = False
        hang = 0

        for m in range(M):
            buf[idx] = P[m]
            idx = (idx + 1) % L
            n = min(n + 1, L)

            Pmax = buf[:n].max(axis=0)

            if noise is None:
                noise = Pmax.copy()
            else:
                if not on:
                    target = np.minimum(Pmax, noise)
                    noise = (1 - self.ema_alpha) * noise + self.ema_alpha * target

            ltsd_db = float(np.mean(10.0 * np.log10((Pmax + self.eps) / (noise + self.eps))))
            T[m] = ltsd_db

            if on:
                if (ltsd_db < self.thr_off_db) and hang <= 0:
                    on = False
            else:
                if ltsd_db > self.thr_on_db:
                    on = True
                    hang = self.hangover_frames

            if on:
                out[m] = 1
                hang = max(hang - 1, 0)

        self.last_statistic = T
        return out
