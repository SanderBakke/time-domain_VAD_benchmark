# vad/features.py
# Lightweight signal utilities for time-domain and frequency-domain VAD.
# Designed to be simple and explicit.

from __future__ import annotations
import numpy as np


def frame_signal(x, fs, frame_ms=32.0, hop_ms=16.0, frame_size=None, hop_size=None):
    """
    Returns frames as shape [M, N].
    Supports either ms-based args (frame_ms/hop_ms) or sample-based (frame_size/hop_size).
    """
    x = np.asarray(x, dtype=np.float32)

    if frame_size is None:
        frame_size = int(round(frame_ms * fs / 1000.0))
    if hop_size is None:
        hop_size = int(round(hop_ms * fs / 1000.0))

    frame_size = max(1, int(frame_size))
    hop_size = max(1, int(hop_size))

    if x.size < frame_size:
        pad = frame_size - x.size
        x = np.pad(x, (0, pad), mode="constant")

    n_frames = 1 + (x.size - frame_size) // hop_size
    idx = (np.arange(frame_size)[None, :] + hop_size * np.arange(n_frames)[:, None])
    return x[idx]


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Short-term energy per frame."""
    frames = np.asarray(frames, dtype=np.float32)
    return (frames * frames).sum(axis=1) + 1e-12


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Proportion of sample-to-sample sign flips inside the frame."""
    frames = np.asarray(frames, dtype=np.float32)
    signs = np.sign(frames)
    signs[signs == 0] = -1.0
    flips = (signs[:, 1:] * signs[:, :-1]) < 0
    return flips.mean(axis=1).astype(np.float32)


def rms_energy(frames: np.ndarray) -> np.ndarray:
    """RMS per frame."""
    frames = np.asarray(frames, dtype=np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)


def zero_energy_ratio(frames: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """
    ZER: fraction of samples in a frame with |x| < eps.
    Lower => more likely speech (typically).
    """
    frames = np.asarray(frames, dtype=np.float32)
    return (np.abs(frames) < float(eps)).mean(axis=1).astype(np.float32)


def log_energy_variance(energy: np.ndarray, window: int = 5, eps: float = 1e-12) -> np.ndarray:
    """
    Per-frame variance of log-energy over a sliding window centered on each frame.
    Matches the LogVarVAD-style feature, but exported as a feature vector for combo models.
    """
    energy = np.asarray(energy, dtype=np.float32)
    logE = np.log(energy + float(eps)).astype(np.float32)

    w = int(max(1, window))
    half = w // 2
    out = np.zeros_like(logE, dtype=np.float32)

    for i in range(len(logE)):
        s = max(0, i - half)
        e = min(len(logE), i + half + 1)
        out[i] = float(np.var(logE[s:e], ddof=0))
    return out


# -----------------------------
# Frequency-domain feature utils
# -----------------------------

def _hann_window(N: int) -> np.ndarray:
    return np.hanning(N).astype(np.float32)


def stft_power(frames: np.ndarray, n_fft: int = 256, window: str = "hann") -> np.ndarray:
    """
    Per-frame power spectrum using rFFT.
    frames: [M, N]
    returns P: [M, K] where K = n_fft//2 + 1
    """
    frames = np.asarray(frames, dtype=np.float32)
    M, N = frames.shape

    if window == "hann":
        w = _hann_window(N)
    elif window is None or window == "rect":
        w = np.ones(N, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported window '{window}'")

    xw = frames * w[None, :]

    if N < n_fft:
        pad = np.zeros((M, n_fft - N), dtype=np.float32)
        xw = np.concatenate([xw, pad], axis=1)
    elif N > n_fft:
        xw = xw[:, :n_fft]

    X = np.fft.rfft(xw, n=n_fft, axis=1)
    P = (X.real * X.real + X.imag * X.imag).astype(np.float32)
    return P + 1e-20


def make_linear_bands(n_fft: int, n_bands: int, drop_dc: bool = True) -> list[np.ndarray]:
    """
    Create simple linear bands as lists of FFT-bin indices.
    drop_dc=True excludes bin 0 from banding.
    """
    K = n_fft // 2 + 1
    bins = np.arange(K, dtype=np.int32)
    if drop_dc and K > 1:
        bins = bins[1:]
    band_splits = np.array_split(bins, n_bands)
    band_bins = [b for b in band_splits if len(b) > 0]
    return band_bins


def band_energies(P: np.ndarray, n_bands: int = 16, band_bins: list[np.ndarray] | None = None) -> np.ndarray:
    """
    Pool power spectrum into band energies.
    P: [M, K]
    returns Eb: [M, B]
    """
    P = np.asarray(P, dtype=np.float32)
    M, K = P.shape

    if band_bins is None:
        n_fft = 2 * (K - 1)
        band_bins = make_linear_bands(n_fft=n_fft, n_bands=n_bands, drop_dc=True)

    B = len(band_bins)
    Eb = np.zeros((M, B), dtype=np.float32)
    for b, idx in enumerate(band_bins):
        Eb[:, b] = P[:, idx].sum(axis=1)
    return Eb + 1e-20


def spectral_flatness(P: np.ndarray, eps: float = 1e-12, drop_dc: bool = True) -> np.ndarray:
    """
    Spectral flatness per frame:
      SFM = exp(mean(log(P))) / mean(P)
    """
    P = np.asarray(P, dtype=np.float32)
    if drop_dc and P.shape[1] > 1:
        Puse = P[:, 1:]
    else:
        Puse = P

    logP = np.log(Puse + eps)
    gm = np.exp(logP.mean(axis=1))
    am = Puse.mean(axis=1) + eps
    return (gm / am).astype(np.float32)
