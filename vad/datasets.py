# vad/datasets.py
# Robust dataset iterator supporting:
#  (A) Subset layout: <root>/speech/*.wav and <root>/noise/*.wav
#  (B) Speech Commands layout: <root>/<word>/*.wav and <root>/_background_noise_/*.wav (sliced)
#
# IMPORTANT: iter_dataset() yields (source, x, fs, label) to match run_vad.py expectations.

from __future__ import annotations

import os
import glob
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

SR_TARGET = 16000

# Speech Commands special dirs to exclude as "words"
SPEECH_LABELS_EXCLUDE = {"_background_noise_", "_silence_", "silence", "noise"}


def _read_wav_mono(fp: str) -> Tuple[np.ndarray, int]:
    """Read wav to float32 mono, returning (x, fs)."""
    x, fs = sf.read(fp, dtype="float32", always_2d=False)
    if isinstance(fs, np.generic):
        fs = int(fs)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
    return x, int(fs)


def _ensure_1s(x: np.ndarray, fs: int) -> np.ndarray:
    """Pad/trim to exactly 1 second at fs."""
    n = fs
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = n - x.shape[0]
    return np.pad(x, (0, pad), mode="constant")


def _list_wavs(dirpath: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dirpath, "*.wav")))


def _has_subset_layout(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "speech")) and os.path.isdir(os.path.join(root, "noise"))


# ---------- Speech Commands fallback (word dirs + _background_noise_) ----------

def list_word_dirs(root: str) -> List[str]:
    entries = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted([d for d in entries if d not in SPEECH_LABELS_EXCLUDE])


def iter_speech_files_speechcommands(root: str) -> Iterator[str]:
    for wd in list_word_dirs(root):
        for fp in glob.glob(os.path.join(root, wd, "*.wav")):
            yield fp


def list_background_noise_files(root: str) -> List[str]:
    bgdir = os.path.join(root, "_background_noise_")
    if not os.path.isdir(bgdir):
        return []
    return sorted(glob.glob(os.path.join(bgdir, "*.wav")))


def slice_background_into_seconds(wav_path: str, sr_target: int = SR_TARGET) -> List[np.ndarray]:
    x, sr = _read_wav_mono(wav_path)
    if sr != sr_target:
        raise ValueError(f"Expected {sr_target} Hz, got {sr} Hz in {wav_path}.")
    seg_len = sr_target
    n_full = len(x) // seg_len
    if n_full == 0:
        return []
    return [x[i * seg_len : (i + 1) * seg_len] for i in range(n_full)]


# ---------- Unified iterator ----------

def iter_dataset(root: str, max_files: Optional[int] = None, seed: int = 0) -> Iterator[Tuple[str, np.ndarray, int, int]]:
    """
    Yield (source_path, waveform, fs, label) where:
      label=1 for speech, label=0 for non-speech.

    Supported roots:
      A) Subset layout: root/speech/*.wav and root/noise/*.wav  (preferred for your project)
      B) Speech Commands: root/<word>/*.wav and root/_background_noise_/*.wav (noise sliced to 1s)
    """
    rng = random.Random(seed)

    # --- Case A: subset layout (speech/ + noise/) ---
    if _has_subset_layout(root):
        speech_dir = os.path.join(root, "speech")
        noise_dir = os.path.join(root, "noise")

        speech_files = _list_wavs(speech_dir)
        noise_files = _list_wavs(noise_dir)

        rng.shuffle(speech_files)
        rng.shuffle(noise_files)

        items: List[Tuple[str, int]] = [(fp, 1) for fp in speech_files] + [(fp, 0) for fp in noise_files]
        rng.shuffle(items)

        if max_files is not None:
            items = items[:max_files]

        for fp, label in items:
            x, fs = _read_wav_mono(fp)
            if fs != SR_TARGET:
                raise ValueError(f"Expected {SR_TARGET} Hz, got {fs} Hz in {fp}.")
            x = _ensure_1s(x, fs)
            yield (fp, x, fs, int(label))

        return

    # --- Case B: Speech Commands fallback ---
    speech_files = list(iter_speech_files_speechcommands(root))
    rng.shuffle(speech_files)

    bg_files = list_background_noise_files(root)
    nonspeech_segs: List[np.ndarray] = []
    for bfp in bg_files:
        nonspeech_segs.extend(slice_background_into_seconds(bfp, SR_TARGET))

    # Balance counts (at most as many nonspeech as speech)
    n = len(speech_files)
    if len(nonspeech_segs) > n:
        rng.shuffle(nonspeech_segs)
        nonspeech_segs = nonspeech_segs[:n]

    items2: List[Tuple[str, int]] = [(fp, 1) for fp in speech_files] + [(f"<bg:{i}>", 0) for i in range(len(nonspeech_segs))]
    rng.shuffle(items2)

    if max_files is not None:
        items2 = items2[:max_files]

    bg_idx = 0
    for source, label in items2:
        if label == 1:
            x, fs = _read_wav_mono(source)
            if fs != SR_TARGET:
                raise ValueError(f"Expected {SR_TARGET} Hz, got {fs} Hz in {source}.")
            x = _ensure_1s(x, fs)
            yield (source, x, fs, 1)
        else:
            x = nonspeech_segs[bg_idx]
            fs = SR_TARGET
            x = _ensure_1s(x, fs)
            yield (source, x, fs, 0)
            bg_idx += 1
