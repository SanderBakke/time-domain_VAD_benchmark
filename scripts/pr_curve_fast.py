#!/usr/bin/env python
# scripts/pr_curve_fast.py
# Fast PR + FP/hour vs Recall using pandas (C engine).

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# def load_scores_fast(path: Path, use_prob: bool):
#     usecols = ["label_frame", "prob" if use_prob else "score"]
#     df = pd.read_csv(path, usecols=usecols)
#     y = df["label_frame"].astype(np.int32).to_numpy()
#     s = (df["prob"] if use_prob else df["score"]).astype(np.float64).to_numpy()
#     # drop NaNs (just in case)
#     mask = np.isfinite(s)
#     return y[mask], s[mask]

# replace the existing load_scores_fast with this:
def load_scores_fast(path: Path, use_prob: bool):
    col = "prob" if use_prob else "score"
    usecols = ["label_frame", col]
    dtypes = {"label_frame": "int8", col: "float32"}
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        memory_map=True,
        low_memory=False,
    )
    y = df["label_frame"].to_numpy()
    s = df[col].to_numpy()
    mask = np.isfinite(s)
    return y[mask], s[mask]


def fp_per_hour(y_true, y_score, thr, fps):
    y_pred = (y_score >= thr).astype(np.int32)
    noise = (y_true == 0)
    fp = int(((y_pred == 1) & noise).sum())
    noise_frames = int(noise.sum())
    if noise_frames <= 0:
        return float("nan")
    return (fp / noise_frames) * (fps * 3600.0)

def main():
    ap = argparse.ArgumentParser(description="Fast PR & FP/hour vs Recall from frame scores.")
    ap.add_argument("--scores_csv", action="append", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--use_prob", action="store_true")
    ap.add_argument("--outdir", type=str, default="outputs/pr")
    ap.add_argument("--filename_pr", type=str, default="pr_all.png")
    ap.add_argument("--filename_fp", type=str, default="fp_per_hour_vs_recall.png")
    ap.add_argument("--frame_hop_ms", type=float, default=10.0)
    ap.add_argument("--fp_target", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    labels = args.labels if (args.labels and len(args.labels)==len(args.scores_csv)) \
             else [Path(p).stem for p in args.scores_csv]
    fps = 1000.0 / float(args.frame_hop_ms)

    plt.figure(figsize=(6,5))
    fig_fp, ax_fp = plt.subplots(figsize=(6,5))

    summary_rows = []
    for pth, lab in zip(args.scores_csv, labels):
        path = Path(pth)
        if not path.exists():
            print(f"[WARN] Missing: {path}"); continue
        print(f"[INFO] Loading: {lab}  ({path})")
        y, s = load_scores_fast(path, use_prob=args.use_prob)

        precision, recall, thresholds = precision_recall_curve(y, s)
        ap_score = float(average_precision_score(y, s))

        if thresholds.size == 0:
            # Degenerate case: constant scores
            with open(outdir / f"pr_{lab}.csv", "w", encoding="utf-8") as f:
                f.write("precision,recall,threshold,fp_per_hour\n")
                f.write(f"{precision[-1]:.6f},{recall[-1]:.6f},,\n")
            summary_rows.append({"label": lab, "average_precision": ap_score})
            continue

        thr_ext = np.concatenate([thresholds, [thresholds[-1]]])
        fp_hour = np.array([fp_per_hour(y, s, t, fps) for t in thresholds])
        if fp_hour.size + 1 == precision.size:
            fp_hour = np.concatenate([fp_hour, [fp_hour[-1]]])

        # Save per-curve CSV
        pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": thr_ext,
            "fp_per_hour": fp_hour
        }).to_csv(outdir / f"pr_{lab}.csv", index=False)

        # Plots
        plt.plot(recall, precision, label=f"{lab} (AP={ap_score:.3f})")
        ax_fp.plot(recall[:-1] if recall.size>1 else recall, fp_hour, label=lab)

        # Operating point selection
        sel = {}
        if args.fp_target is not None and fp_hour.size > 0:
            ok = np.where(np.isfinite(fp_hour) & (fp_hour <= args.fp_target))[0]
            if ok.size > 0:
                idx = int(ok[np.argmax(recall[ok])])
            else:
                idx = int(np.nanargmin(fp_hour))
            sel = {
                "threshold": float(thresholds[idx]),
                "recall": float(recall[idx]),
                "precision": float(precision[idx]),
                "fp_per_hour": float(fp_hour[idx]),
            }

        summary_rows.append({"label": lab, "average_precision": ap_score, **sel})

    # Finalize
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curves")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(outdir / args.filename_pr, dpi=args.dpi); plt.close()

    ax_fp.set_xlabel("Recall"); ax_fp.set_ylabel("FP per hour (on noise)")
    ax_fp.set_title("FP/hour vs Recall"); ax_fp.legend(loc="upper right")
    fig_fp.tight_layout(); fig_fp.savefig(outdir / args.filename_fp, dpi=args.dpi); plt.close(fig_fp)

    # Summary
    import csv
    fields = ["label","average_precision","threshold","recall","precision","fp_per_hour"]
    with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in summary_rows:
            for k in fields:
                r.setdefault(k, "")
            w.writerow(r)

    print(f"[OK] Wrote PR figure → {outdir / args.filename_pr}")
    print(f"[OK] Wrote FP/hour vs Recall → {outdir / args.filename_fp}")
    print(f"[OK] Summary → {outdir / 'summary.csv'}")

if __name__ == "__main__":
    main()
