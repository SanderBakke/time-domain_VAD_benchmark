# Simple file-system loader for Speech Commands v0.02.
# We rely only on soundfile and numpy to keep dependencies light.

from __future__ import annotations
import os, glob, random
import numpy as np
import soundfile as sf
from typing import List, Tuple, Iterator, Optional

SPEECH_LABELS_EXCLUDE = set(["_background_noise_", "_silence_"])

def list_word_dirs(root: str) -> List[str]:
    # Word folders contain wav files. Excludes special folders.
    entries = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted([d for d in entries if d not in SPEECH_LABELS_EXCLUDE])

def iter_speech_files(root: str) -> Iterator[str]:
    # Iterate wavs inside word folders (speech clips)
    for wd in list_word_dirs(root):
        for fp in glob.glob(os.path.join(root, wd, "*.wav")):
            yield fp

def list_background_noise_files(root: str) -> List[str]:
    bgdir = os.path.join(root, "_background_noise_")
    if not os.path.isdir(bgdir):
        return []
    return sorted(glob.glob(os.path.join(bgdir, "*.wav")))

def slice_background_into_seconds(wav_path: str, sr_target: int = 16000) -> List[np.ndarray]:
    """Load a long background-noise wav and slice into 1-second segments (no overlap)."""
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)  # mono mixdown
    if sr != sr_target:
        raise ValueError(f"Expected {sr_target} Hz, got {sr} Hz in {wav_path}.")
    seg_len = sr_target
    n_full = len(x) // seg_len
    if n_full == 0:
        return []
    return [x[i*seg_len:(i+1)*seg_len] for i in range(n_full)]

def load_clip(fp: str, sr_target: int = 16000) -> np.ndarray:
    x, sr = sf.read(fp, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sr_target:
        raise ValueError(f"Expected {sr_target} Hz, got {sr} Hz in {fp}.")
    return x

def iter_dataset(root: str, max_files: Optional[int] = None, seed: int = 0) -> Iterator[Tuple[np.ndarray, int, str]]:
    """
    Yield (waveform, label, source_path) where label=1 for speech (word folders), 0 for non-speech (background noise).
    We create 1-second non-speech segments by slicing background noise files.
    """
    rng = random.Random(seed)
    # Gather speech files
    speech_files = list(iter_speech_files(root))
    rng.shuffle(speech_files)
    # Gather non-speech segments from background noise
    bg_files = list_background_noise_files(root)
    nonspeech_segs = []
    for bfp in bg_files:
        nonspeech_segs.extend(slice_background_into_seconds(bfp, 16000))

    # Balance counts: take at most len(speech_files) non-speech segments
    n = len(speech_files)
    if len(nonspeech_segs) > n:
        rng.shuffle(nonspeech_segs)
        nonspeech_segs = nonspeech_segs[:n]

    items = []
    # Label=1 for speech clips
    for fp in speech_files:
        items.append((fp, 1))
    # Label=0 for non-speech synthetic "files" carried as arrays (we store them inline, path shown as "<bg:idx>")
    for i, seg in enumerate(nonspeech_segs):
        items.append((f"<bg:{i}>", 0))

    rng.shuffle(items)
    if max_files is not None:
        items = items[:max_files]

    # Now yield data
    bg_cache = nonspeech_segs  # indexable by order of creation
    bg_idx = 0
    for fp, label in items:
        if label == 1:
            x = load_clip(fp)
            yield (x, 1, fp)
        else:
            x = bg_cache[bg_idx]
            yield (x, 0, fp)
            bg_idx += 1
