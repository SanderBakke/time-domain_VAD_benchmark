# #!/usr/bin/env python
# # scripts/evaluate_vad.py
# # Unified evaluator: reads clip-level results and (optionally) runtime logs.
# # Produces summary table + confusion bar. Supports a second timing-only runtime file.

# import argparse, csv
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt


# def load_clip_results(path: Path):
#     lab, pred = [], []
#     with open(path, "r", encoding="utf-8") as f:
#         rd = csv.DictReader(f)
#         for r in rd:
#             lab.append(int(r["label"]))
#             pred.append(int(r["pred"]))
#     y = np.asarray(lab, dtype=np.int32)
#     p = np.asarray(pred, dtype=np.int32)
#     return y, p


# def confusion(y, p):
#     tn = int(((y == 0) & (p == 0)).sum())
#     tp = int(((y == 1) & (p == 1)).sum())
#     fp = int(((y == 0) & (p == 1)).sum())
#     fn = int(((y == 1) & (p == 0)).sum())
#     return tp, fp, fn, tn


# def metrics_from_conf(tp, fp, fn, tn):
#     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
#     acc = (tp + tn) / max(tp + fp + fn + tn, 1)
#     return prec, rec, f1, acc


# def read_runtime(runtime_csv: Path):
#     speeds = {}
#     if not runtime_csv or not runtime_csv.exists():
#         return speeds
#     with open(runtime_csv, "r", encoding="utf-8") as f:
#         rd = csv.DictReader(f)
#         for r in rd:
#             m = (r.get("model", "") or "").strip()
#             try:
#                 speeds[m] = float(r.get("sec_per_hour", ""))
#             except Exception:
#                 pass
#     return speeds


# def norm_key(s: str) -> str:
#     return s.lower().replace("-", "_").replace(" ", "")


# def match_speed(label: str, speeds: dict):
#     """Robust label→speed lookup: case/space/hyphen insensitive; allows partial matches."""
#     if not speeds:
#         return ""
#     lab_norm = norm_key(label)
#     # direct alias for common models
#     alias = {
#         "webrtc_l0": "webrtc_l0", "webrtc-l0": "webrtc_l0",
#         "webrtc_l1": "webrtc_l1", "webrtc-l1": "webrtc_l1",
#         "webrtc_l2": "webrtc_l2", "webrtc-l2": "webrtc_l2",
#         "webrtc_l3": "webrtc_l3", "webrtc-l3": "webrtc_l3",
#         "energy": "energy",
#         "energy-fixed": "energy-fixed",
#         "zcr": "zcr",
#         "combo": "combo",
#     }
#     if lab_norm in alias and alias[lab_norm] in speeds:
#         return speeds[alias[lab_norm]]
#     # relaxed search
#     for k in speeds.keys():
#         if norm_key(k) == lab_norm or lab_norm in norm_key(k) or norm_key(k) in lab_norm:
#             return speeds[k]
#     return ""


# def main():
#     ap = argparse.ArgumentParser(description="Unified VAD evaluation (clip-level).")
#     ap.add_argument("--clips", nargs="+", required=True, help="Clip-level CSVs (source,label,pred).")
#     ap.add_argument("--labels", nargs="*", default=None, help="Display labels matching --clips order.")
#     ap.add_argument("--runtime", type=str, default=None, help="Optional end-to-end runtime_summary.csv")
#     ap.add_argument("--runtime_timing", type=str, default=None, help="Optional timing-only runtime_summary.csv")
#     ap.add_argument("--outdir", type=str, default="outputs/eval")
#     ap.add_argument("--fp_target", type=float, default=None, help="Project FP/hour target (for table note).")
#     args = ap.parse_args()

#     outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
#     paths = [Path(p) for p in args.clips]
#     labels = args.labels if (args.labels and len(args.labels) == len(paths)) else \
#              [p.stem.replace("clip_results_", "") for p in paths]

#     speeds_end2end = read_runtime(Path(args.runtime)) if args.runtime else {}
#     speeds_timing  = read_runtime(Path(args.runtime_timing)) if args.runtime_timing else {}

#     rows = []
#     bars = {"TP": [], "FP": [], "FN": [], "TN": [], "label": []}

#     for p, lab in zip(paths, labels):
#         y, pred = load_clip_results(p)
#         tp, fp, fn, tn = confusion(y, pred)
#         prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

#         sec_per_hour = match_speed(lab, speeds_end2end)
#         sec_per_hour_timing = match_speed(lab, speeds_timing)

#         rows.append({
#             "model": lab,
#             "clips": int(y.size),
#             "TP": tp, "FP": fp, "FN": fn, "TN": tn,
#             "precision": round(prec, 4),
#             "recall":    round(rec,  4),
#             "f1":        round(f1,   4),
#             "accuracy":  round(acc,  4),
#             "sec_per_hour": sec_per_hour,
#             "sec_per_hour_timing": sec_per_hour_timing,
#         })

#         bars["TP"].append(tp); bars["FP"].append(fp); bars["FN"].append(fn); bars["TN"].append(tn); bars["label"].append(lab)

#     # Write summary CSV
#     fields = ["model","clips","TP","FP","FN","TN","precision","recall","f1","accuracy","sec_per_hour","sec_per_hour_timing"]
#     with open(outdir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
#         for r in rows: w.writerow(r)

#     # Confusion bar (stacked)
#     x = np.arange(len(bars["label"]))
#     width = 0.6
#     plt.figure(figsize=(9,5))
#     bottom = np.zeros_like(x, dtype=float)
#     for name, color in (("TP","#4caf50"), ("FP","#f44336"), ("FN","#ff9800"), ("TN","#2196f3")):
#         vals = np.array(bars[name], dtype=float)
#         plt.bar(x, vals, width, bottom=bottom, label=name, color=color)
#         bottom += vals
#     plt.xticks(x, bars["label"], rotation=15, ha="right")
#     plt.ylabel("Count (clips)")
#     plt.title("Confusion matrix counts per model (clip-level)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outdir / "confusion_bar.png", dpi=150)
#     plt.close()

#     # Pretty table image
#     fig, ax = plt.subplots(figsize=(12, 0.6 + 0.38*len(rows)))
#     ax.axis("off")
#     col_labels = ["Model","Clips","TP","FP","FN","TN","Precision","Recall","F1","Accuracy","sec/hour","sec/hour (timing)"]
#     table_data = [[r["model"], r["clips"], r["TP"], r["FP"], r["FN"], r["TN"],
#                    r["precision"], r["recall"], r["f1"], r["accuracy"], r["sec_per_hour"], r["sec_per_hour_timing"]] for r in rows]
#     table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
#     table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.3)
#     plt.tight_layout()
#     plt.savefig(outdir / "summary_table.png", dpi=150)
#     plt.close()

#     note = f"(Target FP/hour: {args.fp_target})" if args.fp_target is not None else ""
#     print(f"Saved: {outdir/'summary_metrics.csv'}, {outdir/'confusion_bar.png'}, {outdir/'summary_table.png'} {note}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# scripts/evaluate_vad.py
# Unified evaluator: reads clip-level results and runtime logs.
# Shows end-to-end sec/hour and timing-only mean ± std (if provided).

import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_clip_results(path: Path):
    lab, pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            lab.append(int(r["label"]))
            pred.append(int(r["pred"]))
    return np.asarray(lab, np.int32), np.asarray(pred, np.int32)


def confusion(y, p):
    tn = int(((y == 0) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return tp, fp, fn, tn


def metrics_from_conf(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    return prec, rec, f1, acc


def read_runtime(csv_path: Path):
    if not csv_path or not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            m = (r.get("model", "") or "").strip()
            out[m] = {k: v for k, v in r.items()}
    return out


def norm_key(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "")


def match_runtime(label: str, rt_dict: dict):
    if not rt_dict:
        return None
    ln = norm_key(label)
    for k, row in rt_dict.items():
        kn = norm_key(k)
        if kn == ln or ln in kn or kn in ln:
            return row
    return None


def get_speed_fields(row: dict, prefer_mean=True):
    if not row:
        return "", ""
    if prefer_mean and "sec_per_hour_mean" in row:
        mean = row.get("sec_per_hour_mean", "")
        std  = row.get("sec_per_hour_std", "")
        return mean, std
    val = row.get("sec_per_hour", "")
    return val, ""


def main():
    ap = argparse.ArgumentParser(description="Unified VAD evaluation (clip-level).")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--runtime", type=str, default=None, help="End-to-end runtime_summary.csv")
    ap.add_argument("--runtime_timing", type=str, default=None, help="Timing-only runtime_summary.csv")
    ap.add_argument("--outdir", type=str, default="outputs/eval")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.clips]
    labels = (
        args.labels
        if (args.labels and len(args.labels) == len(paths))
        else [p.stem.replace("clip_results_", "") for p in paths]
    )

    rt_end = read_runtime(Path(args.runtime)) if args.runtime else {}
    rt_tim = read_runtime(Path(args.runtime_timing)) if args.runtime_timing else {}

    rows = []
    bars = {"TP": [], "FP": [], "FN": [], "TN": [], "label": []}

    for p, lab in zip(paths, labels):
        y, pred = load_clip_results(p)
        tp, fp, fn, tn = confusion(y, pred)
        prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

        end_row = match_runtime(lab, rt_end)
        tim_row = match_runtime(lab, rt_tim)

        end_speed, _ = get_speed_fields(end_row, prefer_mean=False)
        tim_mean, tim_std = get_speed_fields(tim_row, prefer_mean=True)

        rows.append(
            {
                "model": lab,
                "clips": int(y.size),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "sec_per_hour": end_speed,
                "sec_per_hour_timing_mean": tim_mean,
                "sec_per_hour_timing_std": tim_std,
            }
        )

        # FIX: append the actual lowercase variables (not locals()["TP"])
        bars["TP"].append(tp)
        bars["FP"].append(fp)
        bars["FN"].append(fn)
        bars["TN"].append(tn)
        bars["label"].append(lab)

    # Write summary CSV
    fields = [
        "model",
        "clips",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "sec_per_hour",
        "sec_per_hour_timing_mean",
        "sec_per_hour_timing_std",
    ]
    with open(outdir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Confusion bar (stacked)
    x = np.arange(len(bars["label"]))
    width = 0.6
    plt.figure(figsize=(10, 5))
    bottom = np.zeros_like(x, dtype=float)
    for name, color in (("TP", "#4caf50"), ("FP", "#f44336"), ("FN", "#ff9800"), ("TN", "#2196f3")):
        vals = np.array(bars[name], dtype=float)
        plt.bar(x, vals, width, bottom=bottom, label=name, color=color)
        bottom += vals
    plt.xticks(x, bars["label"], rotation=15, ha="right")
    plt.ylabel("Count (clips)")
    plt.title("Confusion matrix counts (clip-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "confusion_bar.png", dpi=150)
    plt.close()

    # Pretty table image
    # fig, ax = plt.subplots(figsize=(13, 0.6 + 0.38 * len(rows)))
    # ax.axis("off")
    # col_labels = [
    #     "Model",
    #     "Clips",
    #     "TP",
    #     "FP",
    #     "FN",
    #     "TN",
    #     "Precision",
    #     "Recall",
    #     "F1",
    #     "Accuracy",
    #     "sec/hour",
    #     "sec/hour (timing mean)",
    #     "± std",
    # ]
    # table_data = [
    #     [
    #         r["model"],
    #         r["clips"],
    #         r["TP"],
    #         r["FP"],
    #         r["FN"],
    #         r["TN"],
    #         r["precision"],
    #         r["recall"],
    #         r["f1"],
    #         r["accuracy"],
    #         r["sec_per_hour"],
    #         r["sec_per_hour_timing_mean"],
    #         r["sec_per_hour_timing_std"],
    #     ]
    #     for r in rows
    # ]
    # table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
    # table.auto_set_font_size(False)
    # table.set_fontsize(9)
    # table.scale(1, 1.3)
    # plt.tight_layout()
    # plt.savefig(outdir / "summary_table.png", dpi=150)
    # plt.close()

        # Pretty table image — autosize columns by content length
    # ------------------------------------------------------
    # Build column labels (short & clear)
    col_labels = [
        "Model", "Clips", "TP", "FP", "FN", "TN",
        "Precision", "Recall", "F1", "Accuracy",
        "sec/hour", "timing mean", "± std",
    ]

    # Build table row strings (so we can measure widths reliably)
    str_rows = []
    for r in rows:
        str_rows.append([
            str(r["model"]),
            f'{r["clips"]}',
            f'{r["TP"]}', f'{r["FP"]}', f'{r["FN"]}', f'{r["TN"]}',
            f'{r["precision"]}', f'{r["recall"]}', f'{r["f1"]}', f'{r["accuracy"]}',
            str(r["sec_per_hour"]),
            str(r["sec_per_hour_timing_mean"]),
            str(r["sec_per_hour_timing_std"]),
        ])

    # Compute per-column max string length across header + rows
    def col_len(col_idx):
        head = len(col_labels[col_idx])
        body = max((len(r[col_idx]) for r in str_rows), default=0)
        return max(head, body)

    col_lengths = [col_len(i) for i in range(len(col_labels))]

    # Convert lengths to column width fractions
    # Add a small minimum so narrow columns remain clickable/legible
    min_width = 0.06
    total = float(sum(col_lengths)) or 1.0
    col_widths = [max(min_width, L / total) for L in col_lengths]

    # Choose figure size proportional to total text mass (wider when more content)
    # Height scales with number of rows
    fig_w = max(10.0, 1.0 * sum(col_widths) * 12.0)   # tune factor as desired
    fig_h = 0.8 + 0.38 * len(str_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=str_rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    plt.tight_layout()
    plt.savefig(outdir / "summary_table.png", dpi=150)
    plt.close()


    print(f"Saved: {outdir/'summary_metrics.csv'}, {outdir/'confusion_bar.png'}, {outdir/'summary_table.png'}")


if __name__ == "__main__":
    main()
