#!/usr/bin/env python
# CLI to evaluate time-domain VAD variants on Speech Commands v0.02
# Now with optional per-frame score/probability output for ROC/AUC.

from __future__ import annotations
import os, csv, argparse, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from tqdm import tqdm

from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD
from vad.datasets import iter_dataset
from vad.metrics import evaluate_clip_level, Timer

def parse_args():
    p = argparse.ArgumentParser(description="Time-domain VAD benchmark (clip-level + optional frame scores).")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root of Speech Commands v0.02 (must contain word folders and _background_noise_)")
    p.add_argument("--algo", type=str, default="energy_adaptive",
                   choices=["energy_fixed","energy_adaptive","zcr","combo"],
                   help="Which VAD to run.")
    p.add_argument("--frame_ms", type=float, default=20.0)
    p.add_argument("--hop_ms", type=float, default=10.0)

    # Energy/ZCR/Combo knobs
    p.add_argument("--fixed_threshold", type=float, default=1e-3, help="Energy threshold if using energy_fixed.")
    p.add_argument("--on_ratio", type=float, default=3.0, help="Energy ON threshold ratio vs noise floor (adaptive).")
    p.add_argument("--off_ratio", type=float, default=1.5, help="Energy OFF threshold ratio vs noise floor (adaptive).")
    p.add_argument("--ema_alpha", type=float, default=0.05, help="EMA smoothing for noise floor (adaptive).")
    p.add_argument("--zcr_max", type=float, default=0.12, help="ZCR threshold (lower than this = speech)")
    p.add_argument("--hangover_ms", type=float, default=200.0, help="Hangover duration in milliseconds.")

    # Score / probability export
    p.add_argument("--emit_scores", action="store_true",
                   help="Write per-frame score/probability CSV for ROC/AUC.")
    p.add_argument("--scores_csv", type=str, default="outputs/frame_scores.csv",
                   help="Path for per-frame scores CSV.")
    p.add_argument("--score_gain", type=float, default=2.0,
                   help="Sigmoid gain for mapping score->prob around score=1.0.")
    p.add_argument("--combo_gamma", type=float, default=1.0,
                   help="Scaling for ZCR score inside combo fusion: s = min(sE, gamma*sZ).")

    # Misc
    p.add_argument("--max_files", type=int, default=None, help="Debug: cap number of files/segments processed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=str, default="outputs/clip_level_results.csv")
    return p.parse_args()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dual_rate_ema(prev, x, alpha_off=0.05, alpha_on=0.0125, is_on=False):
    a = float(alpha_on if is_on else alpha_off)
    return (1.0 - a) * prev + a * x

def compute_scores(energy, zcr, algo, on_mask=None, k=2.0, gamma=1.0):
    """
    Build per-frame 'speech-likeliness' scores and mapped probabilities in [0,1].
    energy, zcr: arrays per-frame
    algo: "energy_adaptive" | "zcr" | "combo" | "energy_fixed"
    on_mask: boolean array (detected speech frames) to slow EMA while ON
    k: sigmoid gain for prob mapping around score=1.0
    gamma: scaling for sZ inside combo fusion
    Returns dict with keys: score, prob, score_E, score_Z
    """
    eps = 1e-12
    n = len(energy)
    if on_mask is None:
        on_mask = np.zeros(n, dtype=bool)

    # --- Energy score: ratio to dynamic noise floor ---
    nf_e = max(eps, np.percentile(energy, 20) * 0.8)
    sE = np.zeros(n, dtype=float)
    for i, e in enumerate(energy):
        nf_e = dual_rate_ema(nf_e, e, is_on=on_mask[i])
        sE[i] = e / max(nf_e, eps)

    # --- ZCR score: invert normalized ZCR so 'higher=more speechy' ---
    zcr_ref = max(eps, np.percentile(zcr, 95))
    sZ = np.zeros(n, dtype=float)
    for i, z in enumerate(zcr):
        zn = min(1.0, z / zcr_ref)
        sZ[i] = 1.0 - zn  # low ZCR -> near 1.0 (speech-like), high ZCR -> near 0.0

    if algo in ("energy_adaptive", "energy_fixed"):
        s = sE
    elif algo == "zcr":
        s = sZ
    else:  # combo
        s = np.minimum(sE, gamma * sZ)

    p = sigmoid(k * (s - 1.0))
    return {"score": s, "prob": p, "score_E": sE, "score_Z": sZ}

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Configure algorithm
    fps = 1000.0 / args.hop_ms
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0/fps))))

    if args.algo == "energy_fixed":
        vad = EnergyVAD(mode="fixed", fixed_threshold=args.fixed_threshold, hangover_frames=hang_frames)
    elif args.algo == "energy_adaptive":
        vad = EnergyVAD(mode="adaptive", on_ratio=args.on_ratio, off_ratio=args.off_ratio,
                        ema_alpha=args.ema_alpha, hangover_frames=hang_frames)
    elif args.algo == "zcr":
        vad = ZCRVAD(zcr_max=args.zcr_max, hangover_frames=hang_frames)
    elif args.algo == "combo":
        vad = ComboVAD(on_ratio=args.on_ratio, off_ratio=args.off_ratio, zcr_max=args.zcr_max,
                       ema_alpha=args.ema_alpha, hangover_frames=hang_frames)
    else:
        raise ValueError("Unknown algo")

    y_true, y_pred, rows = [], [], []

    # Optional frame-score CSV
    score_writer = None
    score_f = None
    if args.emit_scores:
        os.makedirs(os.path.dirname(args.scores_csv) or ".", exist_ok=True)
        score_f = open(args.scores_csv, "w", newline="", encoding="utf-8")
        score_writer = csv.writer(score_f)
        score_writer.writerow(["source", "frame_idx", "label_frame", "score", "prob"])

    with Timer() as t_total:
        it = iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed)
        for x, label, src in tqdm(it, desc="Processing clips"):
            # Frame
            frames = frame_signal(x, sr=16000, frame_ms=args.frame_ms, hop_ms=args.hop_ms)

            # Features
            energy = short_time_energy(frames)
            zcr = zero_crossing_rate(frames)

            # VAD decision per frame
            if isinstance(vad, ComboVAD):
                decisions = vad.predict_frames(energy, zcr)
            elif isinstance(vad, ZCRVAD):
                decisions = vad.predict_frames(zcr)
            else:
                decisions = vad.predict_frames(energy)

            # Build on/off mask and per-frame scores/probabilities
            on_mask = decisions.astype(bool)
            scores = compute_scores(energy, zcr, algo=args.algo,
                                    on_mask=on_mask, k=args.score_gain, gamma=args.combo_gamma)

            # Clip-level decision: any speech within the clip?
            pred = int(decisions.max() > 0)

            y_true.append(label)
            y_pred.append(pred)
            rows.append({"source": src, "label": label, "pred": pred})

            # Optional per-frame scores CSV
            if score_writer is not None:
                lbl_frame = int(label)  # all frames inherit clip label in this dataset protocol
                for fi, (sc, pr) in enumerate(zip(scores["score"], scores["prob"])):
                    score_writer.writerow([src, fi, lbl_frame, float(sc), float(pr)])

    metrics = evaluate_clip_level(y_true, y_pred)
    audio_hours = len(y_true) * 1.0 / 3600.0  # each clip is ~1s
    wall_seconds = t_total.dt
    sec_per_audio_hour = wall_seconds / max(audio_hours, 1e-9)

    print("\n=== Clip-level metrics ===")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.4f}")
    print(f"{'sec_per_hour':>10s}: {sec_per_audio_hour:.2f} (wall-clock sec to process 1 hour of audio)")

    # Save clip-level CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","label","pred"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved per-clip results to: {args.out_csv}")

    if score_writer is not None:
        score_f.close()
        print(f"Saved per-frame scores to: {args.scores_csv}")

if __name__ == "__main__":
    main()
