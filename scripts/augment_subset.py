#!/usr/bin/env python3
"""
vad_subset_parallel.py

Create paired VAD test subsets (none / light / heavy) from Speech Commands–style data.
Designed for fair comparison of time-domain and frequency-domain VADs.

Key properties:
- One canonical base selection (speech + noise)
- Multiple augmentation presets applied in parallel
- Identical clips across presets (paired testing)
- No peak normalization (clip-guard only)
- No transient emphasis
- Visual diagnostics for augmentation realism
"""

import argparse
import csv
import json
import random
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import scipy.signal as sig


# =========================
# Basic utilities
# =========================

def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def normalize_length(x, target_len):
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x


def apply_gain(x, gain_db):
    return x * (10.0 ** (gain_db / 20.0))


def clip_guard(x, limit=0.99):
    peak = np.max(np.abs(x))
    if peak > limit:
        x = x * (limit / peak)
    return x


def shift_signal(x, sr, max_shift_ms):
    max_shift = int(sr * max_shift_ms / 1000)
    if max_shift <= 0:
        return x, 0
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        x = np.pad(x, (shift, 0))[:len(x)]
    elif shift < 0:
        x = np.pad(x, (0, -shift))[-shift:]
    return x, shift


# =========================
# Filtering / reverb
# =========================

def _safe_norm(freq, sr):
    nyq = sr / 2
    return max(1e-4, min(freq, nyq - 1e-4)) / nyq


def butter_sos(order, wn, btype):
    return sig.butter(order, wn, btype=btype, output="sos")


def random_bandlimit(x, sr, cfg):
    r = random.random()
    if r < cfg["bp_prob"]:
        lo = random.uniform(200, 800)
        hi = random.uniform(max(lo + 300, 1200), sr / 2 - 200)
        sos = butter_sos(4, (_safe_norm(lo, sr), _safe_norm(hi, sr)), "bandpass")
    elif r < cfg["bp_prob"] + cfg["hp_prob"]:
        cut = random.uniform(cfg["hp_min"], cfg["hp_max"])
        sos = butter_sos(4, _safe_norm(cut, sr), "highpass")
    else:
        cut = random.uniform(cfg["lp_min"], sr / 2 - 200)
        sos = butter_sos(4, _safe_norm(cut, sr), "lowpass")
    return sig.sosfiltfilt(sos, x)


def apply_reverb(x, sr, t60_range):
    t60 = random.uniform(*t60_range)
    ir_len = int(sr * t60)
    ir = np.exp(-np.linspace(0, 3, ir_len))
    ir /= np.max(np.abs(ir))
    y = sig.fftconvolve(x, ir)[:len(x)]
    return y / max(1e-6, np.max(np.abs(y)))


# =========================
# Mixing
# =========================

def mix_with_noise(speech, noise, snr_db):
    sp_rms = rms(speech)
    ns_rms = rms(noise)
    if ns_rms == 0:
        return speech
    desired_ns_rms = sp_rms / (10 ** (snr_db / 20))
    noise = noise * (desired_ns_rms / ns_rms)
    return speech + noise


# =========================
# Presets
# =========================

def preset_cfg(name):
    if name == "none":
        return dict(
            snr=None,
            gain_s=(0, 0),
            gain_n=(0, 0),
            shift=0,
            rev_p_s=0,
            rev_p_n=0,
            bl_p_s=0,
            bl_p_n=0,
        )
    if name == "light":
        return dict(
            snr=[0, 3, 5, 8, 10],
            gain_s=(-6, 6),
            gain_n=(-4, 4),
            shift=100,
            rev_p_s=0.2,
            rev_p_n=0.15,
            bl_p_s=0.2,
            bl_p_n=0.2,
        )
    if name == "heavy":
        return dict(
            snr=[-10, -7, -5, -3, 0, 3],
            gain_s=(-12, 12),
            gain_n=(-8, 8),
            shift=200,
            rev_p_s=0.4,
            rev_p_n=0.3,
            bl_p_s=0.4,
            bl_p_n=0.4,
        )
    raise ValueError(name)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_speech", type=int, default=7500)
    ap.add_argument("--n_noise", type=int, default=7500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--presets", nargs="+", default=["none", "light", "heavy"])
    ap.add_argument("--classes", nargs="+", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    SR = args.sr
    L = SR

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -------- base selection --------
    speech_files = []
    for d in src.iterdir():
        if d.is_dir() and d.name != "_background_noise_":
            if args.classes is None or d.name in args.classes:
                speech_files += list(d.glob("*.wav"))

    noise_src = list((src / "_background_noise_").glob("*.wav"))

    assert speech_files and noise_src

    speech_sel = random.sample(speech_files, args.n_speech)

    noise_segments = []
    for nf in noise_src:
        y, _ = librosa.load(nf, sr=SR)
        for i in range(len(y) // L):
            noise_segments.append((nf.name, y[i * L:(i + 1) * L]))

    if args.n_noise <= len(noise_segments):
        noise_sel = random.sample(noise_segments, args.n_noise)
    else:
        noise_sel = [random.choice(noise_segments) for _ in range(args.n_noise)]

    base_index = {
        "speech": [str(p) for p in speech_sel],
        "noise": [f"{n[0]}::{i}" for i, n in enumerate(noise_sel)],
        "seed": args.seed,
    }
    with open(out / "base_selection.json", "w") as f:
        json.dump(base_index, f, indent=2)

    # -------- generate presets --------
    for preset in args.presets:
        cfg = preset_cfg(preset)
        pdir = out / preset
        (pdir / "speech").mkdir(parents=True, exist_ok=True)
        (pdir / "noise").mkdir(parents=True, exist_ok=True)
        (pdir / "plots").mkdir(parents=True, exist_ok=True)

        snrs, gains, rms_s, rms_n = [], [], [], []

        with open(pdir / "manifest.csv", "w", newline="") as mf:
            wr = csv.writer(mf)
            wr.writerow(["id", "type", "preset", "snr_db", "gain_db", "shift"])

            # speech
            for i, sfp in enumerate(speech_sel):
                x, _ = librosa.load(sfp, sr=SR)
                x = normalize_length(x, L)

                g = random.uniform(*cfg["gain_s"])
                x = apply_gain(x, g)

                if cfg["shift"] > 0:
                    x, sh = shift_signal(x, SR, cfg["shift"])
                else:
                    sh = 0

                if random.random() < cfg["rev_p_s"]:
                    x = apply_reverb(x, SR, (0.2, 0.4))

                if random.random() < cfg["bl_p_s"]:
                    x = random_bandlimit(x, SR, dict(
                        bp_prob=0.4, hp_prob=0.3, lp_min=2000, hp_min=80, hp_max=400
                    ))

                if cfg["snr"] is not None:
                    snr = random.choice(cfg["snr"])
                    nsrc, n = noise_sel[i % len(noise_sel)]
                    y = mix_with_noise(x, n, snr)
                    snrs.append(snr)
                else:
                    y = x
                    snr = ""

                y = clip_guard(y)

                sf.write(pdir / "speech" / f"speech_{i:05d}.wav", y, SR)
                wr.writerow([i, "speech", preset, snr, g, sh])

                gains.append(g)
                rms_s.append(rms(y))

            # noise
            for i, (srcn, n) in enumerate(noise_sel):
                x = normalize_length(n, L)
                g = random.uniform(*cfg["gain_n"])
                x = apply_gain(x, g)

                if random.random() < cfg["rev_p_n"]:
                    x = apply_reverb(x, SR, (0.2, 0.4))
                if random.random() < cfg["bl_p_n"]:
                    x = random_bandlimit(x, SR, dict(
                        bp_prob=0.4, hp_prob=0.3, lp_min=2000, hp_min=80, hp_max=400
                    ))

                x = clip_guard(x)
                sf.write(pdir / "noise" / f"noise_{i:05d}.wav", x, SR)
                wr.writerow([i, "noise", preset, "", g, ""])

                gains.append(g)
                rms_n.append(rms(x))

        # -------- plots --------
        if snrs:
            plt.hist(snrs, bins=20)
            plt.title(f"SNR distribution ({preset})")
            plt.savefig(pdir / "plots" / "snr_hist.png")
            plt.close()

        plt.hist(gains, bins=30)
        plt.title(f"Gain distribution ({preset})")
        plt.savefig(pdir / "plots" / "gain_hist.png")
        plt.close()

        plt.hist(rms_s, bins=30, alpha=0.7, label="speech")
        plt.hist(rms_n, bins=30, alpha=0.7, label="noise")
        plt.legend()
        plt.title(f"RMS overlap ({preset})")
        plt.savefig(pdir / "plots" / "rms_overlap.png")
        plt.close()

    print("Done.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
