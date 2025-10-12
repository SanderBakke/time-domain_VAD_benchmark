# scripts/run_vad_fft.py
# FFT-band-energy VAD (adaptive noise floor + hysteresis + hangover)
# Now with optional per-frame score/probability output for ROC/AUC.

import argparse
import os
import time
import csv
import numpy as np

from vad.datasets import iter_dataset
from vad.features import frame_signal
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def band_power_spectrum(frames_fft_pow, freqs, f_lo, f_hi):
    """Sum power over [f_lo, f_hi] Hz per frame."""
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band_mask):
        return np.zeros(frames_fft_pow.shape[0])
    return frames_fft_pow[:, band_mask].sum(axis=1)

def vad_fft_band_energy(band_energy, on_ratio=3.0, off_ratio=1.5, ema_alpha=0.05, hangover_frames=20):
    """Adaptive VAD on band energy with hysteresis and hangover. Returns frame decisions (0/1)."""
    decisions = np.zeros_like(band_energy, dtype=np.int32)
    nf = max(1e-12, np.percentile(band_energy, 20) * 0.8)
    hang = 0
    is_on = False
    alpha_off = ema_alpha
    alpha_on = ema_alpha * 0.25

    for i, e in enumerate(band_energy):
        # dual-rate EMA for noise floor
        if not is_on:
            nf = (1 - alpha_off) * nf + alpha_off * e
        else:
            nf = (1 - alpha_on) * nf + alpha_on * min(e, nf)

        thr_on = on_ratio * nf
        thr_off = off_ratio * nf

        if is_on:
            if e >= thr_off or hang > 0:
                is_on = True
            else:
                is_on = False
        else:
            if e >= thr_on:
                is_on = True
                hang = hangover_frames

        if is_on:
            decisions[i] = 1
            hang = max(0, hang - 1)

    return decisions

def sec_per_hour(num_clips, wall_seconds):
    audio_hours = num_clips / 3600.0  # ~1 s per clip
    return wall_seconds / max(audio_hours, 1e-9)

def main():
    ap = argparse.ArgumentParser(description="FFT-band-energy VAD (adaptive) over Speech Commands.")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="outputs/clip_results_fft.csv")

    # Framing / FFT
    ap.add_argument("--frame_ms", type=float, default=20.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fft_size", type=int, default=512)          # 512 @16kHz ≈ 31.25 Hz bins
    ap.add_argument("--preemph", type=float, default=0.97)

    # Band of interest (speech)
    ap.add_argument("--band_lo", type=float, default=300.0)
    ap.add_argument("--band_hi", type=float, default=3000.0)

    # VAD thresholds / adaptation
    ap.add_argument("--on_ratio", type=float, default=3.0)
    ap.add_argument("--off_ratio", type=float, default=1.5)
    ap.add_argument("--ema_alpha", type=float, default=0.05)
    ap.add_argument("--hangover_ms", type=float, default=200.0)

    # Scores / probabilities
    ap.add_argument("--emit_scores", action="store_true")
    ap.add_argument("--scores_csv", type=str, default="outputs/frame_scores_fft.csv")
    ap.add_argument("--score_gain", type=float, default=2.0)

    # Misc
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    sr = 16000
    frames_per_sec = 1000.0 / args.hop_ms
    hang_frames = int(round(args.hangover_ms / (1000.0 / frames_per_sec)))

    y_true, y_pred = [], []
    n_clips = 0

    # optional scores CSV
    score_writer = None
    score_f = None
    if args.emit_scores:
        os.makedirs(os.path.dirname(args.scores_csv) or ".", exist_ok=True)
        score_f = open(args.scores_csv, "w", newline="", encoding="utf-8")
        score_writer = csv.writer(score_f)
        score_writer.writerow(["source", "frame_idx", "label_frame", "score", "prob"])

    t0 = time.perf_counter()
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["source", "label", "pred"])

        for x, label, src in iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed):
            # Pre-emphasis
            if args.preemph and args.preemph > 0:
                x = np.append(x[0], x[1:] - args.preemph * x[:-1])

            # Frame + window
            frames = frame_signal(x, sr=sr, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
            if frames.size == 0:
                pred = 0
                y_true.append(label); y_pred.append(pred)
                w.writerow([src, label, pred])
                n_clips += 1
                continue

            win = np.hanning(frames.shape[1]).astype(frames.dtype)
            frames_win = frames * win[None, :]

            # FFT power spectrum and band energy
            spec = np.fft.rfft(frames_win, n=args.fft_size, axis=1)
            pow_spec = (spec.real**2 + spec.imag**2) / args.fft_size
            freqs = np.fft.rfftfreq(args.fft_size, d=1.0/sr)
            band_e = band_power_spectrum(pow_spec, freqs, args.band_lo, args.band_hi)

            # Build per-frame spectral score = band_e / noise_floor (dual-rate EMA)
            eps = 1e-12
            nf = max(eps, np.percentile(band_e, 20) * 0.8)
            sF = np.zeros_like(band_e, dtype=float)
            is_on = False
            alpha_off = args.ema_alpha
            alpha_on = args.ema_alpha * 0.25
            for i, e in enumerate(band_e):
                if not is_on:
                    nf = (1 - alpha_off) * nf + alpha_off * e
                else:
                    nf = (1 - alpha_on) * nf + alpha_on * min(e, nf)
                sF[i] = e / max(nf, eps)

            # Probability mapping
            prob = 1.0 / (1.0 + np.exp(-args.score_gain * (sF - 1.0)))

            # Decisions (same as before)
            decisions = vad_fft_band_energy(
                band_e,
                on_ratio=args.on_ratio,
                off_ratio=args.off_ratio,
                ema_alpha=args.ema_alpha,
                hangover_frames=hang_frames,
            )

            pred = int(decisions.max() > 0)
            y_true.append(label); y_pred.append(pred)
            w.writerow([src, label, pred])
            n_clips += 1

            if score_writer is not None:
                lbl_frame = int(label)
                for fi, (sc, pr) in enumerate(zip(sF, prob)):
                    score_writer.writerow([src, fi, lbl_frame, float(sc), float(pr)])

    wall = time.perf_counter() - t0

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    print("\n=== Clip-level metrics (FFT band energy) ===")
    print(f"  accuracy: {acc:.4f}")
    print(f" precision: {prec:.4f}")
    print(f"    recall: {rec:.4f}")
    print(f"        f1: {f1:.4f}")
    print(f"sec_per_hour: {sec_per_hour(n_clips, wall):.2f} (wall-clock sec to process 1 hour of audio)")
    print(f"Saved per-clip results to: {args.out_csv}")

    if score_writer is not None:
        score_f.close()
        print(f"Saved per-frame scores to: {args.scores_csv}")

if __name__ == "__main__":
    main()
