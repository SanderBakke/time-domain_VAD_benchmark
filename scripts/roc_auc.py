#!/usr/bin/env python
# scripts/roc_auc.py
# Compute ROC + AUC from per-frame score/prob CSVs and pick FP-target operating points.

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

def load_scores_csv(path: Path, use_prob: bool):
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

def compute_fp_hour(y_true, y_score, thr, frames_per_second):
    y_pred = (y_score >= thr).astype(np.int32)
    mask_noise = (y_true == 0)
    fp = int(((y_pred == 1) & mask_noise).sum())
    noise_frames = int(mask_noise.sum())
    if noise_frames <= 0:
        return float("nan")
    frames_per_hour = frames_per_second * 3600.0
    return (fp / noise_frames) * frames_per_hour

def pick_operating_point(y_true, y_score, fpr_target, frames_per_second):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    candidates = [(t, f, r) for t, f, r in zip(thr, fpr, tpr) if f <= fpr_target]
    if not candidates:
        idx = int(np.argmin(fpr))
        best_thr, best_fpr, best_tpr = thr[idx], float(fpr[idx]), float(tpr[idx])
    else:
        best_thr, best_fpr, best_tpr = max(candidates, key=lambda x: (x[2], -x[1]))

    y_pred = (y_score >= best_thr).astype(np.int32)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fp_per_hour = compute_fp_hour(y_true, y_score, best_thr, frames_per_second)

    return {
        "auc": float(roc_auc),
        "threshold": float(best_thr),
        "fpr": float(best_fpr),
        "tpr": float(best_tpr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fp_per_hour": float(fp_per_hour),
        "fpr_curve": fpr,
        "tpr_curve": tpr,
        "thr_curve": thr,
    }

def main():
    ap = argparse.ArgumentParser(description="ROC/AUC from per-frame VAD scores; pick FP-target operating points.")
    ap.add_argument("--scores_csv", action="append", required=True,
                    help="One or more per-frame score CSV files (from run_vad/run_vad_fft with --emit_scores).")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels for legend (same length/order as --scores_csv).")
    ap.add_argument("--use_prob", action="store_true", help="Use 'prob' column instead of 'score' (default: score).")
    ap.add_argument("--outdir", type=str, default="outputs/roc")
    ap.add_argument("--filename", type=str, default="roc_all.png")
    ap.add_argument("--fpr_target", type=float, default=0.005, help="Target false-positive rate (e.g., 0.005 = 0.5%).")
    ap.add_argument("--frame_hop_ms", type=float, default=10.0, help="Frame hop in ms to convert FP/hour.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.labels is None or len(args.labels) != len(args.scores_csv):
        labels = [Path(p).stem for p in args.scores_csv]
    else:
        labels = args.labels

    fps = 1000.0 / float(args.frame_hop_ms)

    summary_rows = []

    plt.figure(figsize=(6, 5))

    for path_str, lab in zip(args.scores_csv, labels):
        path = Path(path_str)
        y, s = load_scores_csv(path, use_prob=args.use_prob)

        res = pick_operating_point(y, s, fpr_target=args.fpr_target, frames_per_second=fps)

        plt.plot(res["fpr_curve"], res["tpr_curve"], label=f"{lab} (AUC={res['auc']:.4f})")

        roc_csv = outdir / f"roc_{lab}.csv"
        with open(roc_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fpr", "tpr", "threshold"])
            for fpr_v, tpr_v, thr_v in zip(res["fpr_curve"], res["tpr_curve"], res["thr_curve"]):
                w.writerow([float(fpr_v), float(tpr_v), float(thr_v)])

        summary_rows.append({
            "label": lab,
            "auc": res["auc"],
            "threshold": res["threshold"],
            "fpr_at_thr": res["fpr"],
            "tpr_at_thr": res["tpr"],
            "precision_at_thr": res["precision"],
            "recall_at_thr": res["recall"],
            "f1_at_thr": res["f1"],
            "fp_per_hour_at_thr": res["fp_per_hour"],
        })

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title(f"VAD ROC Curves (target FPR ≤ {args.fpr_target*100:.2f}%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png = outdir / args.filename
    plt.savefig(out_png, dpi=150)
    plt.close()

    summary_csv = outdir / "summary_operating_points.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "auc", "threshold", "fpr_at_thr", "tpr_at_thr",
            "precision_at_thr", "recall_at_thr", "f1_at_thr", "fp_per_hour_at_thr"
        ])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"Saved ROC figure to: {out_png}")
    print(f"Saved per-curve CSVs and summary to: {outdir}")
    print("Done.")

if __name__ == "__main__":
    main()
