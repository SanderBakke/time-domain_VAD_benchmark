#!/usr/bin/env python3
"""
augment_subset.py
Create a small augmented subset of a Speech Commands–style dataset for realistic VAD benchmarking.

What it does (and why):
- Picks a subset of speech files + splits noise into 1s clips (balances classes; faster iteration).
- Applies realistic augmentations: gain, SNR mixing, time shift, short reverb, safe band-limiting.
- Presets (--preset light|heavy|custom) control augmentation strength and probabilities.
- Saves augmented WAVs into <out>/{speech,noise} and logs manifest.csv + config.json (reproducibility).
- Plots histograms for SNR and gain to verify augmentation distributions.
- Uses numerically-stable SOS filters with safe cutoffs (fixes "0 < Wn < 1" SciPy errors).

Usage (defaults okay):
    python scripts/augment_subset.py
or with options/presets:
    python scripts/augment_subset.py --preset light --out data_aug_light --n_speech 4000 --n_noise 4000
    python scripts/augment_subset.py --preset heavy --out data_aug_heavy --n_speech 4000 --n_noise 4000
"""

import argparse
import csv
import json
import random
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is not installed. Run inside your venv:\n"
        "    python -m pip install librosa soundfile scipy matplotlib\n"
    )

import soundfile as sf
import scipy.signal as sig


# ----------------------- Helpers -----------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def normalize_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")
    return x

def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return x * (10.0 ** (gain_db / 20.0))

def _safe_norm(freq_hz: float, sr: int, eps: float = 1e-6) -> float:
    """Normalize frequency to (0,1) for digital filters; clamp away from 0 and Nyquist."""
    nyq = sr / 2.0
    f = max(eps, min(float(freq_hz), nyq - eps))
    return f / nyq

def _butter_sos(order: int, wn, btype: str):
    """Stable SOS butterworth helper."""
    return sig.butter(order, wn, btype=btype, output="sos")

def random_bandlimit(x: np.ndarray, sr: int, p_conf: dict) -> np.ndarray:
    """
    Random simple coloration with safe cutoffs and SOS filtering:
      - bp_prob: band-pass in a speechy band
      - hp_prob: high-pass (remove lows)
      - lp_prob: low-pass (remove highs)
    """
    r = random.random()
    bp_prob = p_conf["bp_prob"]
    hp_prob = p_conf["hp_prob"]
    # lp_prob is implicitly 1 - bp_prob - hp_prob
    nyq = sr / 2.0

    if r < bp_prob:
        # Band-pass: either phone band or a randomized band
        if random.random() < 0.5:
            low_hz, high_hz = 300.0, min(3400.0, nyq - 50.0)
        else:
            low_hz = random.uniform(150.0, 1200.0)
            high_hz = random.uniform(max(low_hz + 200.0, 1200.0), nyq - 50.0)
        wn = (_safe_norm(low_hz, sr), _safe_norm(high_hz, sr))
        sos = _butter_sos(4, wn, btype="bandpass")
        return sig.sosfiltfilt(sos, x)

    elif r < bp_prob + hp_prob:
        # High-pass
        cut_hz = random.uniform(p_conf["hp_min"], p_conf["hp_max"])
        wn = _safe_norm(cut_hz, sr)
        sos = _butter_sos(4, wn, btype="highpass")
        return sig.sosfiltfilt(sos, x)

    else:
        # Low-pass
        cut_hz = random.uniform(p_conf["lp_min"], nyq - 200.0)  # margin from Nyquist
        wn = _safe_norm(cut_hz, sr)
        sos = _butter_sos(4, wn, btype="lowpass")
        return sig.sosfiltfilt(sos, x)

def apply_reverb(x: np.ndarray, sr: int, t60_range=(0.2, 0.4)) -> np.ndarray:
    """Short 'roomy' reverb via exponential IR; normalized safely to avoid div-zero."""
    decay = random.uniform(*t60_range)        # seconds
    ir_len = max(8, int(sr * decay))
    ir = np.exp(-np.linspace(0, 3, ir_len))
    mx = float(np.max(np.abs(ir)))
    if mx > 0:
        ir = ir / mx
    y = sig.fftconvolve(x, ir)[:len(x)]
    my = float(np.max(np.abs(y)))
    if my > 0:
        y = y / my
    return y

def shift_signal(x: np.ndarray, sr: int, max_shift_ms: float):
    """Zero-padded shift to keep length constant."""
    max_shift = int((max_shift_ms / 1000.0) * sr)
    if max_shift <= 0:
        return x, 0
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        x = np.pad(x, (shift, 0), mode="constant")[:len(x)]
    elif shift < 0:
        x = np.pad(x, (0, -shift), mode="constant")[-shift:]
    return x, shift

def mix_with_noise_multi(speech: np.ndarray, noises: list[np.ndarray], snr_db: float) -> np.ndarray:
    """
    Mix one or more noise tracks to reach target SNR wrt speech RMS.
    1) Sum noises -> combined noise
    2) Scale combined noise to achieve desired SNR
    3) Add and peak-normalize
    """
    sp = speech
    ns = np.zeros_like(speech)
    for n in noises:
        ns = ns + n[:len(ns)]
    sp_rms = rms(sp)
    ns_rms = rms(ns)
    if ns_rms == 0.0:
        y = sp
    else:
        desired_ns_rms = sp_rms / (10.0 ** (snr_db / 20.0))
        scale = desired_ns_rms / ns_rms
        ns_scaled = ns * scale
        y = sp + ns_scaled
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y

def load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr)
    return y.astype(np.float32, copy=False)

def load_random_noise(noise_seg_files, sr: int, target_len: int) -> np.ndarray:
    nfile = random.choice(noise_seg_files)
    n = load_audio(nfile, sr)
    return normalize_length(n, target_len)

def get_speech_files(root: Path, classes):
    files = []
    for c in classes:
        folder = root / c
        if folder.exists():
            files += list(folder.glob("*.wav"))
    return files

def get_noise_files(root: Path):
    folder = root / "_background_noise_"
    if not folder.exists():
        return []
    return list(folder.glob("*.wav"))

def split_noise_to_one_sec(noise_files, out_dir: Path, sr: int, target_len: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = []
    for nf in noise_files:
        x = load_audio(nf, sr)
        nseg = len(x) // target_len
        for i in range(nseg):
            seg = x[i*target_len:(i+1)*target_len]
            seg_file = out_dir / f"{nf.stem}_{i:03d}.wav"
            sf.write(seg_file, seg, sr)
            segs.append(seg_file)
    return segs


# ----------------------- Preset configuration -----------------------

def preset_config(name: str) -> dict:
    """
    Returns a dictionary with augmentation strengths/probabilities.
    - 'light'  : mild, realistic phone/mic variability
    - 'heavy'  : stress test in noisy/colored conditions
    - 'custom' : baseline close to your previous script ("medium-heavy")
    """
    if name == "light":
        return {
            "snr_choices": [0, 3, 5, 8, 10],
            "gain_speech_db": (-6, 6),
            "gain_noise_db": (-4, 4),
            "shift_ms_max": 100.0,
            "reverb_prob_speech": 0.20,
            "reverb_prob_noise": 0.15,
            "bandlimit_prob_speech": 0.20,
            "bandlimit_prob_noise": 0.20,
            "bp_prob": 0.35, "hp_prob": 0.35, "lp_min": 2200.0, "hp_min": 100.0, "hp_max": 300.0,
            "t60_range": (0.18, 0.30),
            "double_noise_prob": 0.0,   # single noise layer
        }
    if name == "heavy":
        return {
            "snr_choices": [-10, -7, -5, -3, 0, 3],
            "gain_speech_db": (-12, 12),
            "gain_noise_db": (-8, 8),
            "shift_ms_max": 200.0,
            "reverb_prob_speech": 0.40,
            "reverb_prob_noise": 0.30,
            "bandlimit_prob_speech": 0.40,
            "bandlimit_prob_noise": 0.40,
            "bp_prob": 0.40, "hp_prob": 0.30, "lp_min": 1800.0, "hp_min": 80.0, "hp_max": 400.0,
            "t60_range": (0.25, 0.40),
            "double_noise_prob": 0.35,  # sometimes add a 2nd noise layer
        }
    # custom = your previous defaults ("medium-heavy")
    return {
        "snr_choices": [-10, -5, 0, 5, 10],
        "gain_speech_db": (-12, 12),
        "gain_noise_db": (-6, 6),
        "shift_ms_max": 200.0,
        "reverb_prob_speech": 0.30,
        "reverb_prob_noise": 0.20,
        "bandlimit_prob_speech": 0.30,
        "bandlimit_prob_noise": 0.30,
        "bp_prob": 0.40, "hp_prob": 0.30, "lp_min": 2500.0, "hp_min": 80.0, "hp_max": 400.0,
        "t60_range": (0.20, 0.40),
        "double_noise_prob": 0.0,
    }


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Augment a small, realistic subset for VAD benchmarking.")
    parser.add_argument("--src", type=str, default="data", help="Source dataset root")
    parser.add_argument("--out", type=str, default=None, help="Output root for augmented data (default depends on preset)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--n_speech", type=int, default=5000, help="Number of speech clips to augment")
    parser.add_argument("--n_noise", type=int, default=5000, help="Number of 1s noise clips to generate")
    parser.add_argument("--classes", type=str, nargs="*", default=[
        "bed","bird","cat","dog","down","eight","five","go"
    ], help="Speech classes to include in subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--preset", type=str, default="custom", choices=["light","heavy","custom"],
                        help="Augmentation strength preset")
    args = parser.parse_args()

    # RNG seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Config from preset
    cfg = preset_config(args.preset)

    SRC_ROOT = Path(args.src)
    # If --out omitted, choose default name by preset
    if args.out is None:
        OUT_ROOT = Path(f"data_aug_{args.preset}")
    else:
        OUT_ROOT = Path(args.out)
    SR = int(args.sr)
    TARGET_LEN = SR * 1  # 1 second

    # Output dirs
    (OUT_ROOT / "speech").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "noise").mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    cfg_snapshot = {
        "preset": args.preset,
        "seed": args.seed,
        "sr": SR,
        "n_speech": args.n_speech,
        "n_noise": args.n_noise,
        "classes": args.classes,
        "params": cfg
    }
    with open(OUT_ROOT / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, indent=2)

    # Manifest CSV
    manifest_path = OUT_ROOT / "manifest.csv"
    mf = open(manifest_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=[
        "outfile","source","class","gain_db","snr_db","shift_ms","reverb","bandlimit","preset"
    ])
    writer.writeheader()

    # Discover files
    speech_files = get_speech_files(SRC_ROOT, args.classes)
    noise_files = get_noise_files(SRC_ROOT)
    print(f"Found {len(speech_files)} speech, {len(noise_files)} noise source files")
    if len(noise_files) == 0:
        print("[warn] No files under '_background_noise_'. You can still run, but 'mix_with_noise' will fail.")
        print("       Please add noise WAVs into SRC/_background_noise_/")

    # Split noise into 1s segments
    split_noise_dir = OUT_ROOT / "noise_split"
    all_noise_segments = split_noise_to_one_sec(noise_files, split_noise_dir, SR, TARGET_LEN)
    print(f"Split noise into {len(all_noise_segments)} 1s segments")

    # Choose subsets
    n_speech = min(args.n_speech, len(speech_files))
    chosen_speech = random.sample(speech_files, n_speech)

    if args.n_noise <= len(all_noise_segments):
        chosen_noise = random.sample(all_noise_segments, args.n_noise)
    else:
        # sample with replacement to reach requested count
        chosen_noise = [random.choice(all_noise_segments) for _ in range(args.n_noise)]

    # Tracking for histograms
    snr_values, gain_values = [], []

    # Bandlimit sub-config for filters
    bl_conf = dict(
        bp_prob=cfg["bp_prob"], hp_prob=cfg["hp_prob"],
        lp_min=cfg["lp_min"], hp_min=cfg["hp_min"], hp_max=cfg["hp_max"]
    )

    # --- Augment speech ---
    for sfile in chosen_speech:
        x = load_audio(sfile, SR)
        x = normalize_length(x, TARGET_LEN)

        # Gain
        g_lo, g_hi = cfg["gain_speech_db"]
        gain_db = random.uniform(g_lo, g_hi)
        x = apply_gain(x, gain_db)
        gain_values.append(gain_db)

        # Shift
        shift_ms = 0.0
        if cfg["shift_ms_max"] > 0 and random.random() < 0.5:
            x, shift = shift_signal(x, SR, cfg["shift_ms_max"])
            shift_ms = 1000.0 * shift / SR

        # Reverb
        reverb_used = False
        if random.random() < cfg["reverb_prob_speech"]:
            x = apply_reverb(x, SR, t60_range=cfg["t60_range"])
            reverb_used = True

        # Band-limit
        band_used = False
        if random.random() < cfg["bandlimit_prob_speech"]:
            x = random_bandlimit(x, SR, bl_conf)
            band_used = True

        # Mix with 1 or 2 noise layers
        snr_db = random.choice(cfg["snr_choices"])
        noise1 = load_random_noise(all_noise_segments, SR, TARGET_LEN)
        noises = [noise1]
        if random.random() < cfg["double_noise_prob"]:
            noise2 = load_random_noise(all_noise_segments, SR, TARGET_LEN)
            noises.append(noise2)

        y = mix_with_noise_multi(x, noises, snr_db)
        snr_values.append(snr_db)

        out_file = OUT_ROOT / "speech" / f"{sfile.stem}_aug.wav"
        sf.write(out_file, y, SR)
        writer.writerow({
            "outfile": str(out_file),
            "source": str(sfile),
            "class": "speech",
            "gain_db": round(gain_db, 2),
            "snr_db": snr_db,
            "shift_ms": round(shift_ms, 1),
            "reverb": reverb_used,
            "bandlimit": band_used,
            "preset": args.preset
        })

    # --- Augment noise (light coloring) ---
    for nfile in chosen_noise:
        x = load_audio(nfile, SR)
        x = normalize_length(x, TARGET_LEN)

        g_lo_n, g_hi_n = cfg["gain_noise_db"]
        gain_db = random.uniform(g_lo_n, g_hi_n)
        x = apply_gain(x, gain_db)
        gain_values.append(gain_db)

        reverb_used = False
        if random.random() < cfg["reverb_prob_noise"]:
            x = apply_reverb(x, SR, t60_range=cfg["t60_range"])
            reverb_used = True

        band_used = False
        if random.random() < cfg["bandlimit_prob_noise"]:
            x = random_bandlimit(x, SR, bl_conf)
            band_used = True

        out_file = OUT_ROOT / "noise" / f"{Path(nfile).stem}_aug.wav"
        sf.write(out_file, x, SR)
        writer.writerow({
            "outfile": str(out_file),
            "source": str(nfile),
            "class": "noise",
            "gain_db": round(gain_db, 2),
            "snr_db": "",
            "shift_ms": "",
            "reverb": reverb_used,
            "bandlimit": band_used,
            "preset": args.preset
        })

    mf.close()
    print(f"Augmentation complete. Saved under {OUT_ROOT}")

    # --- Histograms ---
    if snr_values:
        plt.figure(figsize=(10,4))
        bins = np.arange(min(cfg["snr_choices"])-2, max(cfg["snr_choices"])+4, 2)
        plt.hist(snr_values, bins=bins, edgecolor="black")
        plt.title(f"SNR Distribution (Speech Augmentations) — preset={args.preset}")
        plt.xlabel("SNR [dB]"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "snr_hist.png")
        plt.close()

    if gain_values:
        plt.figure(figsize=(10,4))
        plt.hist(gain_values, bins=30, edgecolor="black")
        plt.title(f"Gain Distribution (All Augmentations) — preset={args.preset}")
        plt.xlabel("Gain [dB]"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "gain_hist.png")
        plt.close()

    print(f"Saved histogram plots to {OUT_ROOT}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
