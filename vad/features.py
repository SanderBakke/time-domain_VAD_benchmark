# Lightweight signal utilities for time-domain VAD.
# Highly commented for learning purposes.

from __future__ import annotations
import numpy as np

def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    """
    Slice a 1-D waveform into overlapping frames (rows).
    x: float32 waveform in range [-1, 1] preferably
    Returns: 2-D array of shape [num_frames, frame_length]
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_ms/hop_ms too small for the sample rate.")
    if len(x) < frame_len:
        # pad with zeros to at least one frame
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad), mode="constant")

    # compute number of frames
    n_frames = 1 + (len(x) - frame_len) // hop_len
    # build a 2-D view by stride trick
    strides = (x.strides[0]*hop_len, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, frame_len), strides=strides)
    return frames.copy()  # copy to avoid accidental modification of original buffer

def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Sum of squares per frame (STE)."""
    # add a tiny epsilon to avoid exact zeros
    return (frames * frames).sum(axis=1) + 1e-12

def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Proportion of sample-to-sample sign flips inside the frame."""
    # sign: positive -> 1, negative/zero -> -1 (zero considered as negative side to avoid flicker)
    signs = np.sign(frames)
    signs[signs == 0] = -1.0
    flips = (signs[:, 1:] * signs[:, :-1]) < 0  # True where sign changes
    # normalize by number of transitions
    return flips.mean(axis=1).astype(np.float32)
