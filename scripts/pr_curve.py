#!/usr/bin/env python
# scripts/pr_curve.py
# Make Precision–Recall curves and FP/hour vs Recall from per-frame VAD score CSVs.
#
# Input files are produced by run_vad.py / run_vad_fft.py with --emit_scores:
#   columns: source,frame_idx,label_frame,score,prob
#
# Example:
#   python -m scripts.pr_curve \
#     --scores_csv outputs/frame_scores_energy.csv \
#     --scores_csv outputs/frame_scores_zcr.csv \
#     --scores_csv outputs/frame_scores_combo.csv \
#     --scores_csv outputs/frame_scores_fft.csv \
#     --labels Energy ZCR Combo FFT \
#     --use_prob \
#     --outdir outputs/pr \
#     --fp_target 18 \
#     --frame_hop_ms 10
#
# Produces:
#   outputs/pr/pr_all.png                (PR curves + AP)
#   outputs/pr/fp_per_hour_vs_recall.png (FP/hour vs Recall)
#   outputs/pr/pr_<label>.csv            (precision, recall, threshold, fp_per_hour)
#   outputs/pr/summary.csv               (per-method AP and selected operating point)

import argparse
import csv
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def load_scores_csv(path: Path, use_prob: bool) -> Tuple[np.ndarray, np.ndarray]:
    y = []
    s = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                y.append(int(r["label_frame"]))
                val = float(r["prob"]) if use_prob else float(r["score"])
                if not np.isfinite(val):
                    continue
                s.append(val)
            except Exception:
                continue
    if not y:
        raise RuntimeError(f"No rows parsed from {path}")
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)

def fp_per_hour_from_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float, fps: float) -> float:
    y_pred = (y_score >= thr).astype(np.int32)
    noise_mask = (y_true == 0)
    fp = int(((y_pred == 1) & noise_mask).sum())
    noise_frames = int(noise_mask.sum())
    if noise_frames <= 0:
        return float("nan")
    frames_per_hour = fps * 3600.0
    return (fp / noise_frames) * frames_per_hour

def main():
    ap = argparse.ArgumentParser(description="Precision–Recall and FP/hour vs Recall from VAD frame scores.")
    ap.add_argument("--scores_csv", action="append", required=True,
                    help="One or more per-frame score CSVs.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Legend labels matching --scores_csv order.")
    ap.add_argument("--use_prob", action="store_true", help="Use 'prob' column instead of 'score'.")
    ap.add_argument("--outdir", type=str, default="outputs/pr")
    ap.add_argument("--filename_pr", type=str, default="pr_all.png")
    ap.add_argument("--filename_fp", type=str, default="fp_per_hour_vs_recall.png")
    ap.add_argument("--frame_hop_ms", type=float, default=10.0, help="Hop size (ms) to compute frames/sec.")
    ap.add_argument("--fp_target", type=float, default=None,
                    help="Optional FP/hour target for selecting operating point (e.g., 18).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.labels is None or len(args.labels) != len(args.scores_csv):
        labels = [Path(p).stem for p in args.scores_csv]
    else:
        labels = args.labels

    fps = 1000.0 / float(args.frame_hop_ms)

    # Prepare plots
    plt.figure(figsize=(6,5))
    summary_rows: List[Dict[str, float]] = []

    # For FP/hour vs Recall
    plt_fp = plt.figure(figsize=(6,5))
    ax_fp = plt_fp.gca()

    for path_str, lab in zip(args.scores_csv, labels):
        y, s = load_scores_csv(Path(path_str), use_prob=args.use_prob)

        # Precision–Recall
        precision, recall, thresholds = precision_recall_curve(y, s)
        ap_score = average_precision_score(y, s)

        # Align threshold array to precision/recall length (+1 element)
        thr_ext = np.concatenate([thresholds, [thresholds[-1] if thresholds.size else 0.0]])
        # Compute FP/hour for each threshold (skip last if redundant)
        fp_per_hour = np.array([fp_per_hour_from_threshold(y, s, t, fps) for t in thresholds])
        if fp_per_hour.size + 1 == precision.size:
            fp_per_hour = np.concatenate([fp_per_hour, [fp_per_hour[-1] if fp_per_hour.size else np.nan]])

        # Save per-curve CSV
        with open(outdir / f"pr_{lab}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["precision", "recall", "threshold", "fp_per_hour"])
            for p, r, t, fph in zip(precision, recall, thr_ext, fp_per_hour):
                w.writerow([float(p), float(r), float(t), float(fph)])

        # Plot PR
        plt.plot(recall, precision, label=f"{lab} (AP={ap_score:.3f})")

        # Plot FP/hour vs Recall (use same recall grid)
        ax_fp.plot(recall[:-1] if recall.size>1 else recall, fp_per_hour, label=lab)

        # Choose an operating point if requested
        sel = {}
        if args.fp_target is not None and fp_per_hour.size > 0:
            # pick highest recall with fp/hour <= target
            ok = np.where(fp_per_hour <= args.fp_target)[0]
            if ok.size > 0:
                idx = int(ok[np.argmax(recall[ok])])
                thr_sel = thresholds[idx]
                sel = {
                    "threshold": float(thr_sel),
                    "recall": float(recall[idx]),
                    "precision": float(precision[idx]),
                    "fp_per_hour": float(fp_per_hour[idx]),
                }
            else:
                # choose minimum fp/hour (most conservative)
                idx = int(np.argmin(fp_per_hour))
                sel = {
                    "threshold": float(thresholds[idx]),
                    "recall": float(recall[idx]),
                    "precision": float(precision[idx]),
                    "fp_per_hour": float(fp_per_hour[idx]),
                }

        summary_rows.append({
            "label": lab,
            "average_precision": float(ap_score),
            **({} if not sel else sel)
        })

    # Finalize PR plot
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outdir / args.filename_pr, dpi=150)
    plt.close()

    # Finalize FP/hour vs Recall
    ax_fp.set_xlabel("Recall")
    ax_fp.set_ylabel("FP per hour (on noise)")
    ax_fp.set_title("FP/hour vs Recall")
    ax_fp.legend(loc="upper right")
    plt_fp.tight_layout()
    plt_fp.savefig(outdir / args.filename_fp, dpi=150)
    plt_fp.close()

    # Save summary
    with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        if any("threshold" in r for r in summary_rows):
            fields = ["label","average_precision","threshold","recall","precision","fp_per_hour"]
        else:
            fields = ["label","average_precision"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"Saved PR figure to: {outdir / args.filename_pr}")
    print(f"Saved FP/hour vs Recall to: {outdir / args.filename_fp}")
    print(f"Wrote per-curve CSVs and summary to: {outdir}")
    print("Done.")

if __name__ == "__main__":
    main()
