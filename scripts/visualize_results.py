#!/usr/bin/env python

"""
Visualize VAD clip-level results with file-name tags per algorithm and
histograms of most-missed words.

Usage examples:
  # Single CSV (energy)
  python scripts/visualize_results.py --csv outputs/clip_results_energy.csv --title "Energy VAD (clip-level)" --labels Energy

  # Compare multiple
  python scripts/visualize_results.py \
    --csv outputs/clip_results_energy.csv \
    --csv outputs/clip_results_zcr.csv \
    --csv outputs/clip_results_combo.csv \
    --labels Energy ZCR Combo \
    --title "VAD Comparison (clip-level)"

Outputs go under outputs/plots/ and are tagged using the provided label for each CSV
(e.g., _energy, _zcr, _combo). If a label is not supplied, a tag is inferred from
the CSV filename.

Generated artifacts per CSV:
- confusion_matrix_<tag>.png
- per_class_recall_or_specificity_<tag>.png
- per_class_error_breakdown_<tag>.png
- fp_by_noise_file_<tag>.png
- summary_panel_<tag>.png
- top_misclassified_hist_<tag>.png        (NEW: histogram of most-missed words)
- top_misclassified_counts_<tag>.csv      (NEW: table of counts for the histogram)

If multiple CSVs are passed, a cross-algorithm plot is also produced:
- comparison_per_class_recall.png
"""
import argparse
import csv
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple


import numpy as np
import matplotlib.pyplot as plt

# ------------------ Helpers ------------------

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def read_results(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"source": r["source"], "label": int(r["label"]), "pred": int(r["pred"])})
    return rows

def extract_class_from_source(source: str) -> str:
    # Determine if this is background noise or a word class
    parts = source.replace("\\", "/").split("/")
    for p in parts:
        if p == "_background_noise_":
            return "_background_noise_"
    # Otherwise, the class is parent folder of the wav
    for i in range(len(parts)-2, -1, -1):
        if parts[i].strip():
            return parts[i]
    return "unknown"

def compute_global_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    y_true = np.array([r["label"] for r in rows], dtype=int)
    y_pred = np.array([r["pred"] for r in rows], dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=accuracy, precision=precision, recall=recall, f1=f1)

def per_class_counts(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    stats = defaultdict(lambda: dict(tp=0, tn=0, fp=0, fn=0, total=0, speech_total=0, nonspeech_total=0))
    for r in rows:
        cls = extract_class_from_source(r["source"])
        y = r["label"]
        p = r["pred"]
        stats[cls]["total"] += 1
        if y == 1:
            stats[cls]["speech_total"] += 1
        else:
            stats[cls]["nonspeech_total"] += 1
        if y == 1 and p == 1:
            stats[cls]["tp"] += 1
        elif y == 0 and p == 0:
            stats[cls]["tn"] += 1
        elif y == 0 and p == 1:
            stats[cls]["fp"] += 1
        elif y == 1 and p == 0:
            stats[cls]["fn"] += 1
    return stats

def top_missed_word_counts(rows: List[Dict[str, str]], topn: int = 20) -> List[Tuple[str, int]]:
    """Return list of (word_class, FN_count) for the most-missed speech words."""
    counter = Counter()
    for r in rows:
        if r["label"] == 1 and r["pred"] == 0:
            cls = extract_class_from_source(r["source"])
            if cls != "_background_noise_":
                counter[cls] += 1
    return counter.most_common(topn)

def infer_tag(csv_path: Path) -> str:
    tag = csv_path.stem.lower()
    for bad in ["results", "clip", "level", "vad", "csv"]:
        tag = tag.replace(bad, "")
    tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in tag).strip("_-")
    return tag or "run"

# ------------------ Plots ------------------

def plot_confusion(global_metrics: Dict[str, float], outpath: Path, title: str):
    tp, tn, fp, fn = global_metrics["tp"], global_metrics["tn"], global_metrics["fp"], global_metrics["fn"]
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title + "\nConfusion matrix (Speech vs Non-speech)")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-speech", "Speech"]); ax.set_yticklabels(["Non-speech", "Speech"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_per_class_recall_specificity(stats: Dict[str, Dict[str, int]], outpath: Path, title: str):
    classes = sorted(stats.keys(), key=lambda c: c.lower())
    values = []
    for c in classes:
        s = stats[c]
        if c == "_background_noise_":
            denom = s["nonspeech_total"]
            val = (s["tn"] / denom) if denom > 0 else 0.0
        else:
            denom = s["speech_total"]
            val = (s["tp"] / denom) if denom > 0 else 0.0
        values.append(val)
    order = np.argsort(values)
    classes = [classes[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(classes)), values)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (speech classes) / Specificity (background)")
    ax.set_title(title + "\nPer-class recall/specificity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_per_class_error_breakdown(stats: Dict[str, Dict[str, int]], outpath: Path, title: str):
    classes = sorted(stats.keys(), key=lambda c: c.lower())
    tp = [stats[c]["tp"] for c in classes]
    fn = [stats[c]["fn"] for c in classes]
    tn = [stats[c]["tn"] for c in classes]
    fp = [stats[c]["fp"] for c in classes]

    correct = []
    error = []
    labels = []
    for c, tp_i, fn_i, tn_i, fp_i in zip(classes, tp, fn, tn, fp):
        if c == "_background_noise_":
            correct.append(tn_i)
            error.append(fp_i)
            labels.append(f"{c} (TN/FP)")
        else:
            correct.append(tp_i)
            error.append(fn_i)
            labels.append(f"{c} (TP/FN)")

    order = np.argsort(error)[::-1]
    labels = [labels[i] for i in order]
    correct = [correct[i] for i in order]
    error = [error[i] for i in order]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, correct, label="Correct", align="center")
    ax.bar(x, error, bottom=correct, label="Error", align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_title(title + "\nPer-class error breakdown (stacked)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_fp_by_noise_file(rows: List[Dict[str, str]], outpath: Path, title: str):
    fp_files = []
    for r in rows:
        if r["label"] == 0 and r["pred"] == 1:
            parts = r["source"].replace("\\", "/").split("/")
            if "_background_noise_" in parts:
                fp_files.append(parts[-1])
    if not fp_files:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No false positives on background noise", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        return

    counts = Counter(fp_files)
    names, vals = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("False positives")
    ax.set_title(title + "\nFalse positives by background-noise file")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_summary_panel(global_metrics: Dict[str, float], outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    txt = (
        f"Accuracy: {global_metrics['accuracy']:.4f}\n"
        f"Precision: {global_metrics['precision']:.4f}\n"
        f"Recall: {global_metrics['recall']:.4f}\n"
        f"F1: {global_metrics['f1']:.4f}\n\n"
        f"TP: {global_metrics['tp']}  FP: {global_metrics['fp']}\n"
        f"FN: {global_metrics['fn']}  TN: {global_metrics['tn']}"
    )
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=11)
    ax.set_title(title + "\nSummary metrics")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_top_missed_words_hist(rows: List[Dict[str, str]], outpath_png: Path, outpath_csv: Path, title: str, topn: int = 20):
    counts = top_missed_word_counts(rows, topn=topn)
    # Save CSV
    with open(outpath_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word_class", "missed_count"])
        for cls, cnt in counts:
            w.writerow([cls, cnt])

    # Plot histogram
    if not counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No missed speech words (FN=0)", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath_png, dpi=150)
        plt.close(fig)
        return

    classes, vals = zip(*counts)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(classes))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_ylabel("Missed count (FN)")
    ax.set_title(title + "\nTop misclassified words (speech FNs)")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=150)
    plt.close(fig)

def plot_comparison_per_class_recall(stats_list: List[Dict[str, Dict[str, int]]], labels: List[str], outpath: Path, title: str):
    classes = sorted(set().union(*[set(s.keys()) for s in stats_list]), key=lambda c: c.lower())
    mat = []
    for stats in stats_list:
        vals = []
        for c in classes:
            s = stats.get(c, dict(tp=0, tn=0, fp=0, fn=0, speech_total=0, nonspeech_total=0))
            if c == "_background_noise_":
                denom = s.get("nonspeech_total", 0)
                val = (s.get("tn", 0) / denom) if denom > 0 else 0.0
            else:
                denom = s.get("speech_total", 0)
                val = (s.get("tp", 0) / denom) if denom > 0 else 0.0
            vals.append(val)
        mat.append(vals)
    mat = np.array(mat)
    A, C = mat.shape
    x = np.arange(C)
    width = 0.8 / max(A, 1)

    fig, ax = plt.subplots(figsize=(max(10, C*0.5), 5))
    for a in range(A):
        ax.bar(x + a*width, mat[a], width=width, label=labels[a] if a < len(labels) else f"CSV{a+1}")
    ax.set_xticks(x + width*(A-1)/2)
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (speech) / Specificity (background)")
    ax.set_title(title + "\nPer-class comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize VAD clip-level results CSV(s).")
    ap.add_argument("--csv", action="append", required=True, help="Path to a results CSV (source,label,pred). Pass multiple to compare.")
    ap.add_argument("--labels", nargs="*", default=None, help="Labels for each CSV (e.g., Energy ZCR Combo).")
    ap.add_argument("--title", type=str, default="VAD Results")
    ap.add_argument("--topn", type=int, default=20, help="Top-N misclassified speech words (FN) to include in the histogram/CSV.")
    ap.add_argument("--outdir", type=str, default="outputs/plots", help="Directory to save figures/tables.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Read CSVs
    rows_list = []
    stats_list = []
    globals_list = []
    tags = []

    for idx, csv_path in enumerate(args.csv):
        rows = read_results(Path(csv_path))
        for r in rows:
            r["class"] = extract_class_from_source(r["source"])
        rows_list.append(rows)
        stats_list.append(per_class_counts(rows))
        globals_list.append(compute_global_metrics(rows))

        # Determine tag: from labels if provided, else from filename
        if args.labels and idx < len(args.labels):
            tag = args.labels[idx].lower().replace(" ", "_")
        else:
            tag = infer_tag(Path(csv_path))
        tags.append(tag)

    # Generate per-CSV plots
    for rows, stats, gm, tag in zip(rows_list, stats_list, globals_list, tags):
        prefix = lambda name: outdir / f"{name}_{tag}.png"
        plot_confusion(gm, prefix("confusion_matrix"), args.title)
        plot_per_class_recall_specificity(stats, prefix("per_class_recall_or_specificity"), args.title)
        plot_per_class_error_breakdown(stats, prefix("per_class_error_breakdown"), args.title)
        plot_fp_by_noise_file(rows, prefix("fp_by_noise_file"), args.title)
        plot_summary_panel(gm, prefix("summary_panel"), args.title)
        # Top misclassified words (speech FNs)
        png_path = prefix("top_misclassified_hist")
        csv_path = outdir / f"top_misclassified_counts_{tag}.csv"
        plot_top_missed_words_hist(rows, png_path, csv_path, args.title, topn=args.topn)

    # If multiple CSVs: also comparison plot
    if len(rows_list) > 1:
        labels = args.labels if args.labels and len(args.labels) == len(rows_list) else tags
        plot_comparison_per_class_recall(stats_list, labels, outdir / "comparison_per_class_recall.png", args.title)

if __name__ == "__main__":
    main()
