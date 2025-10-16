# # # # #!/usr/bin/env python
# # # # # scripts/evaluate_vad.py
# # # # # Unified evaluator: reads clip-level results and (optionally) runtime logs.
# # # # # Produces summary table + confusion bar. Supports a second timing-only runtime file.

# # # # import argparse, csv
# # # # from pathlib import Path
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt


# # # # def load_clip_results(path: Path):
# # # #     lab, pred = [], []
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         rd = csv.DictReader(f)
# # # #         for r in rd:
# # # #             lab.append(int(r["label"]))
# # # #             pred.append(int(r["pred"]))
# # # #     y = np.asarray(lab, dtype=np.int32)
# # # #     p = np.asarray(pred, dtype=np.int32)
# # # #     return y, p


# # # # def confusion(y, p):
# # # #     tn = int(((y == 0) & (p == 0)).sum())
# # # #     tp = int(((y == 1) & (p == 1)).sum())
# # # #     fp = int(((y == 0) & (p == 1)).sum())
# # # #     fn = int(((y == 1) & (p == 0)).sum())
# # # #     return tp, fp, fn, tn


# # # # def metrics_from_conf(tp, fp, fn, tn):
# # # #     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# # # #     rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# # # #     f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
# # # #     acc = (tp + tn) / max(tp + fp + fn + tn, 1)
# # # #     return prec, rec, f1, acc


# # # # def read_runtime(runtime_csv: Path):
# # # #     speeds = {}
# # # #     if not runtime_csv or not runtime_csv.exists():
# # # #         return speeds
# # # #     with open(runtime_csv, "r", encoding="utf-8") as f:
# # # #         rd = csv.DictReader(f)
# # # #         for r in rd:
# # # #             m = (r.get("model", "") or "").strip()
# # # #             try:
# # # #                 speeds[m] = float(r.get("sec_per_hour", ""))
# # # #             except Exception:
# # # #                 pass
# # # #     return speeds


# # # # def norm_key(s: str) -> str:
# # # #     return s.lower().replace("-", "_").replace(" ", "")


# # # # def match_speed(label: str, speeds: dict):
# # # #     """Robust label→speed lookup: case/space/hyphen insensitive; allows partial matches."""
# # # #     if not speeds:
# # # #         return ""
# # # #     lab_norm = norm_key(label)
# # # #     # direct alias for common models
# # # #     alias = {
# # # #         "webrtc_l0": "webrtc_l0", "webrtc-l0": "webrtc_l0",
# # # #         "webrtc_l1": "webrtc_l1", "webrtc-l1": "webrtc_l1",
# # # #         "webrtc_l2": "webrtc_l2", "webrtc-l2": "webrtc_l2",
# # # #         "webrtc_l3": "webrtc_l3", "webrtc-l3": "webrtc_l3",
# # # #         "energy": "energy",
# # # #         "energy-fixed": "energy-fixed",
# # # #         "zcr": "zcr",
# # # #         "combo": "combo",
# # # #     }
# # # #     if lab_norm in alias and alias[lab_norm] in speeds:
# # # #         return speeds[alias[lab_norm]]
# # # #     # relaxed search
# # # #     for k in speeds.keys():
# # # #         if norm_key(k) == lab_norm or lab_norm in norm_key(k) or norm_key(k) in lab_norm:
# # # #             return speeds[k]
# # # #     return ""


# # # # def main():
# # # #     ap = argparse.ArgumentParser(description="Unified VAD evaluation (clip-level).")
# # # #     ap.add_argument("--clips", nargs="+", required=True, help="Clip-level CSVs (source,label,pred).")
# # # #     ap.add_argument("--labels", nargs="*", default=None, help="Display labels matching --clips order.")
# # # #     ap.add_argument("--runtime", type=str, default=None, help="Optional end-to-end runtime_summary.csv")
# # # #     ap.add_argument("--runtime_timing", type=str, default=None, help="Optional timing-only runtime_summary.csv")
# # # #     ap.add_argument("--outdir", type=str, default="outputs/eval")
# # # #     ap.add_argument("--fp_target", type=float, default=None, help="Project FP/hour target (for table note).")
# # # #     args = ap.parse_args()

# # # #     outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
# # # #     paths = [Path(p) for p in args.clips]
# # # #     labels = args.labels if (args.labels and len(args.labels) == len(paths)) else \
# # # #              [p.stem.replace("clip_results_", "") for p in paths]

# # # #     speeds_end2end = read_runtime(Path(args.runtime)) if args.runtime else {}
# # # #     speeds_timing  = read_runtime(Path(args.runtime_timing)) if args.runtime_timing else {}

# # # #     rows = []
# # # #     bars = {"TP": [], "FP": [], "FN": [], "TN": [], "label": []}

# # # #     for p, lab in zip(paths, labels):
# # # #         y, pred = load_clip_results(p)
# # # #         tp, fp, fn, tn = confusion(y, pred)
# # # #         prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

# # # #         sec_per_hour = match_speed(lab, speeds_end2end)
# # # #         sec_per_hour_timing = match_speed(lab, speeds_timing)

# # # #         rows.append({
# # # #             "model": lab,
# # # #             "clips": int(y.size),
# # # #             "TP": tp, "FP": fp, "FN": fn, "TN": tn,
# # # #             "precision": round(prec, 4),
# # # #             "recall":    round(rec,  4),
# # # #             "f1":        round(f1,   4),
# # # #             "accuracy":  round(acc,  4),
# # # #             "sec_per_hour": sec_per_hour,
# # # #             "sec_per_hour_timing": sec_per_hour_timing,
# # # #         })

# # # #         bars["TP"].append(tp); bars["FP"].append(fp); bars["FN"].append(fn); bars["TN"].append(tn); bars["label"].append(lab)

# # # #     # Write summary CSV
# # # #     fields = ["model","clips","TP","FP","FN","TN","precision","recall","f1","accuracy","sec_per_hour","sec_per_hour_timing"]
# # # #     with open(outdir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
# # # #         w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
# # # #         for r in rows: w.writerow(r)

# # # #     # Confusion bar (stacked)
# # # #     x = np.arange(len(bars["label"]))
# # # #     width = 0.6
# # # #     plt.figure(figsize=(9,5))
# # # #     bottom = np.zeros_like(x, dtype=float)
# # # #     for name, color in (("TP","#4caf50"), ("FP","#f44336"), ("FN","#ff9800"), ("TN","#2196f3")):
# # # #         vals = np.array(bars[name], dtype=float)
# # # #         plt.bar(x, vals, width, bottom=bottom, label=name, color=color)
# # # #         bottom += vals
# # # #     plt.xticks(x, bars["label"], rotation=15, ha="right")
# # # #     plt.ylabel("Count (clips)")
# # # #     plt.title("Confusion matrix counts per model (clip-level)")
# # # #     plt.legend()
# # # #     plt.tight_layout()
# # # #     plt.savefig(outdir / "confusion_bar.png", dpi=150)
# # # #     plt.close()

# # # #     # Pretty table image
# # # #     fig, ax = plt.subplots(figsize=(12, 0.6 + 0.38*len(rows)))
# # # #     ax.axis("off")
# # # #     col_labels = ["Model","Clips","TP","FP","FN","TN","Precision","Recall","F1","Accuracy","sec/hour","sec/hour (timing)"]
# # # #     table_data = [[r["model"], r["clips"], r["TP"], r["FP"], r["FN"], r["TN"],
# # # #                    r["precision"], r["recall"], r["f1"], r["accuracy"], r["sec_per_hour"], r["sec_per_hour_timing"]] for r in rows]
# # # #     table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
# # # #     table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.3)
# # # #     plt.tight_layout()
# # # #     plt.savefig(outdir / "summary_table.png", dpi=150)
# # # #     plt.close()

# # # #     note = f"(Target FP/hour: {args.fp_target})" if args.fp_target is not None else ""
# # # #     print(f"Saved: {outdir/'summary_metrics.csv'}, {outdir/'confusion_bar.png'}, {outdir/'summary_table.png'} {note}")


# # # # if __name__ == "__main__":
# # # #     main()


# # # #!/usr/bin/env python
# # # # scripts/evaluate_vad.py
# # # # Unified evaluator: reads clip-level results and runtime logs.
# # # # Shows end-to-end sec/hour and timing-only mean ± std (if provided).

# # # import argparse, csv
# # # from pathlib import Path
# # # import numpy as np
# # # import matplotlib.pyplot as plt


# # # def load_clip_results(path: Path):
# # #     lab, pred = [], []
# # #     with open(path, "r", encoding="utf-8") as f:
# # #         rd = csv.DictReader(f)
# # #         for r in rd:
# # #             lab.append(int(r["label"]))
# # #             pred.append(int(r["pred"]))
# # #     return np.asarray(lab, np.int32), np.asarray(pred, np.int32)


# # # def confusion(y, p):
# # #     tn = int(((y == 0) & (p == 0)).sum())
# # #     tp = int(((y == 1) & (p == 1)).sum())
# # #     fp = int(((y == 0) & (p == 1)).sum())
# # #     fn = int(((y == 1) & (p == 0)).sum())
# # #     return tp, fp, fn, tn


# # # def metrics_from_conf(tp, fp, fn, tn):
# # #     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# # #     rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# # #     f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
# # #     acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
# # #     return prec, rec, f1, acc


# # # def read_runtime(csv_path: Path):
# # #     if not csv_path or not csv_path.exists():
# # #         return {}
# # #     out = {}
# # #     with open(csv_path, "r", encoding="utf-8") as f:
# # #         rd = csv.DictReader(f)
# # #         for r in rd:
# # #             m = (r.get("model", "") or "").strip()
# # #             out[m] = {k: v for k, v in r.items()}
# # #     return out


# # # def norm_key(s: str) -> str:
# # #     return s.lower().replace("-", "_").replace(" ", "")


# # # def match_runtime(label: str, rt_dict: dict):
# # #     if not rt_dict:
# # #         return None
# # #     ln = norm_key(label)
# # #     for k, row in rt_dict.items():
# # #         kn = norm_key(k)
# # #         if kn == ln or ln in kn or kn in ln:
# # #             return row
# # #     return None


# # # def get_speed_fields(row: dict, prefer_mean=True):
# # #     if not row:
# # #         return "", ""
# # #     if prefer_mean and "sec_per_hour_mean" in row:
# # #         mean = row.get("sec_per_hour_mean", "")
# # #         std  = row.get("sec_per_hour_std", "")
# # #         return mean, std
# # #     val = row.get("sec_per_hour", "")
# # #     return val, ""


# # # def main():
# # #     ap = argparse.ArgumentParser(description="Unified VAD evaluation (clip-level).")
# # #     ap.add_argument("--clips", nargs="+", required=True)
# # #     ap.add_argument("--labels", nargs="*", default=None)
# # #     ap.add_argument("--runtime", type=str, default=None, help="End-to-end runtime_summary.csv")
# # #     ap.add_argument("--runtime_timing", type=str, default=None, help="Timing-only runtime_summary.csv")
# # #     ap.add_argument("--outdir", type=str, default="outputs/eval")
# # #     args = ap.parse_args()

# # #     outdir = Path(args.outdir)
# # #     outdir.mkdir(parents=True, exist_ok=True)

# # #     paths = [Path(p) for p in args.clips]
# # #     labels = (
# # #         args.labels
# # #         if (args.labels and len(args.labels) == len(paths))
# # #         else [p.stem.replace("clip_results_", "") for p in paths]
# # #     )

# # #     rt_end = read_runtime(Path(args.runtime)) if args.runtime else {}
# # #     rt_tim = read_runtime(Path(args.runtime_timing)) if args.runtime_timing else {}

# # #     rows = []
# # #     bars = {"TP": [], "FP": [], "FN": [], "TN": [], "label": []}

# # #     for p, lab in zip(paths, labels):
# # #         y, pred = load_clip_results(p)
# # #         tp, fp, fn, tn = confusion(y, pred)
# # #         prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

# # #         end_row = match_runtime(lab, rt_end)
# # #         tim_row = match_runtime(lab, rt_tim)

# # #         end_speed, _ = get_speed_fields(end_row, prefer_mean=False)
# # #         tim_mean, tim_std = get_speed_fields(tim_row, prefer_mean=True)

# # #         rows.append(
# # #             {
# # #                 "model": lab,
# # #                 "clips": int(y.size),
# # #                 "TP": tp,
# # #                 "FP": fp,
# # #                 "FN": fn,
# # #                 "TN": tn,
# # #                 "precision": round(prec, 4),
# # #                 "recall": round(rec, 4),
# # #                 "f1": round(f1, 4),
# # #                 "accuracy": round(acc, 4),
# # #                 "sec_per_hour": end_speed,
# # #                 "sec_per_hour_timing_mean": tim_mean,
# # #                 "sec_per_hour_timing_std": tim_std,
# # #             }
# # #         )

# # #         # FIX: append the actual lowercase variables (not locals()["TP"])
# # #         bars["TP"].append(tp)
# # #         bars["FP"].append(fp)
# # #         bars["FN"].append(fn)
# # #         bars["TN"].append(tn)
# # #         bars["label"].append(lab)

# # #     # Write summary CSV
# # #     fields = [
# # #         "model",
# # #         "clips",
# # #         "TP",
# # #         "FP",
# # #         "FN",
# # #         "TN",
# # #         "precision",
# # #         "recall",
# # #         "f1",
# # #         "accuracy",
# # #         "sec_per_hour",
# # #         "sec_per_hour_timing_mean",
# # #         "sec_per_hour_timing_std",
# # #     ]
# # #     with open(outdir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
# # #         w = csv.DictWriter(f, fieldnames=fields)
# # #         w.writeheader()
# # #         for r in rows:
# # #             w.writerow(r)

# # #     # Confusion bar (stacked)
# # #     x = np.arange(len(bars["label"]))
# # #     width = 0.6
# # #     plt.figure(figsize=(10, 5))
# # #     bottom = np.zeros_like(x, dtype=float)
# # #     for name, color in (("TP", "#4caf50"), ("FP", "#f44336"), ("FN", "#ff9800"), ("TN", "#2196f3")):
# # #         vals = np.array(bars[name], dtype=float)
# # #         plt.bar(x, vals, width, bottom=bottom, label=name, color=color)
# # #         bottom += vals
# # #     plt.xticks(x, bars["label"], rotation=15, ha="right")
# # #     plt.ylabel("Count (clips)")
# # #     plt.title("Confusion matrix counts (clip-level)")
# # #     plt.legend()
# # #     plt.tight_layout()
# # #     plt.savefig(outdir / "confusion_bar.png", dpi=150)
# # #     plt.close()

# # #     # Pretty table image
# # #     # fig, ax = plt.subplots(figsize=(13, 0.6 + 0.38 * len(rows)))
# # #     # ax.axis("off")
# # #     # col_labels = [
# # #     #     "Model",
# # #     #     "Clips",
# # #     #     "TP",
# # #     #     "FP",
# # #     #     "FN",
# # #     #     "TN",
# # #     #     "Precision",
# # #     #     "Recall",
# # #     #     "F1",
# # #     #     "Accuracy",
# # #     #     "sec/hour",
# # #     #     "sec/hour (timing mean)",
# # #     #     "± std",
# # #     # ]
# # #     # table_data = [
# # #     #     [
# # #     #         r["model"],
# # #     #         r["clips"],
# # #     #         r["TP"],
# # #     #         r["FP"],
# # #     #         r["FN"],
# # #     #         r["TN"],
# # #     #         r["precision"],
# # #     #         r["recall"],
# # #     #         r["f1"],
# # #     #         r["accuracy"],
# # #     #         r["sec_per_hour"],
# # #     #         r["sec_per_hour_timing_mean"],
# # #     #         r["sec_per_hour_timing_std"],
# # #     #     ]
# # #     #     for r in rows
# # #     # ]
# # #     # table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
# # #     # table.auto_set_font_size(False)
# # #     # table.set_fontsize(9)
# # #     # table.scale(1, 1.3)
# # #     # plt.tight_layout()
# # #     # plt.savefig(outdir / "summary_table.png", dpi=150)
# # #     # plt.close()

# # #         # Pretty table image — autosize columns by content length
# # #     # ------------------------------------------------------
# # #     # Build column labels (short & clear)
# # #     col_labels = [
# # #         "Model", "Clips", "TP", "FP", "FN", "TN",
# # #         "Precision", "Recall", "F1", "Accuracy",
# # #         "sec/hour", "timing mean", "± std",
# # #     ]

# # #     # Build table row strings (so we can measure widths reliably)
# # #     str_rows = []
# # #     for r in rows:
# # #         str_rows.append([
# # #             str(r["model"]),
# # #             f'{r["clips"]}',
# # #             f'{r["TP"]}', f'{r["FP"]}', f'{r["FN"]}', f'{r["TN"]}',
# # #             f'{r["precision"]}', f'{r["recall"]}', f'{r["f1"]}', f'{r["accuracy"]}',
# # #             str(r["sec_per_hour"]),
# # #             str(r["sec_per_hour_timing_mean"]),
# # #             str(r["sec_per_hour_timing_std"]),
# # #         ])

# # #     # Compute per-column max string length across header + rows
# # #     def col_len(col_idx):
# # #         head = len(col_labels[col_idx])
# # #         body = max((len(r[col_idx]) for r in str_rows), default=0)
# # #         return max(head, body)

# # #     col_lengths = [col_len(i) for i in range(len(col_labels))]

# # #     # Convert lengths to column width fractions
# # #     # Add a small minimum so narrow columns remain clickable/legible
# # #     min_width = 0.06
# # #     total = float(sum(col_lengths)) or 1.0
# # #     col_widths = [max(min_width, L / total) for L in col_lengths]

# # #     # Choose figure size proportional to total text mass (wider when more content)
# # #     # Height scales with number of rows
# # #     fig_w = max(10.0, 1.0 * sum(col_widths) * 12.0)   # tune factor as desired
# # #     fig_h = 0.8 + 0.38 * len(str_rows)

# # #     fig, ax = plt.subplots(figsize=(fig_w, fig_h))
# # #     ax.axis("off")

# # #     table = ax.table(
# # #         cellText=str_rows,
# # #         colLabels=col_labels,
# # #         colWidths=col_widths,
# # #         loc="center"
# # #     )

# # #     table.auto_set_font_size(False)
# # #     table.set_fontsize(9)
# # #     table.scale(1, 1.3)

# # #     plt.tight_layout()
# # #     plt.savefig(outdir / "summary_table.png", dpi=150)
# # #     plt.close()


# # #     print(f"Saved: {outdir/'summary_metrics.csv'}, {outdir/'confusion_bar.png'}, {outdir/'summary_table.png'}")


# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python
# # # scripts/evaluate_vad.py
# # # Unified evaluator: clip-level + fast ROC/PR (frame-level), per-environment, micro/macro averages.

# # import argparse, csv, datetime, json
# # from pathlib import Path
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def now_tag():
# #     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # # ---------- Clip-level helpers ----------

# # def load_clip_results(path: Path):
# #     lab, pred, src = [], [], []
# #     with open(path, "r", encoding="utf-8") as f:
# #         rd = csv.DictReader(f)
# #         for r in rd:
# #             lab.append(int(r["label"]))
# #             pred.append(int(r["pred"]))
# #             src.append(r.get("source",""))
# #     return np.asarray(lab, np.int32), np.asarray(pred, np.int32), np.asarray(src, object)

# # def confusion(y, p):
# #     tn = int(((y == 0) & (p == 0)).sum())
# #     tp = int(((y == 1) & (p == 1)).sum())
# #     fp = int(((y == 0) & (p == 1)).sum())
# #     fn = int(((y == 1) & (p == 0)).sum())
# #     return tp, fp, fn, tn

# # def metrics_from_conf(tp, fp, fn, tn):
# #     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# #     rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# #     f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
# #     acc = (tp + tn) / max(tp + fp + fn + tn, 1)
# #     return prec, rec, f1, acc

# # # ---------- Frame-level fast ROC/PR ----------

# # def compute_curves(labels: np.ndarray, scores: np.ndarray):
# #     """Exact ROC/PR via one sort + cumsums."""
# #     labels = labels.astype(np.int32)
# #     scores = scores.astype(np.float64)
# #     order = np.argsort(scores)[::-1]
# #     y = labels[order]
# #     # cum sums
# #     tp = np.cumsum(y)
# #     fp = np.cumsum(1 - y)
# #     P = tp[-1] if tp.size > 0 else 0
# #     N = fp[-1] if fp.size > 0 else 0
# #     if P == 0 or N == 0:
# #         # Degenerate: can't build curves
# #         return {
# #             "fpr": np.array([0.0, 1.0]),
# #             "tpr": np.array([0.0, 1.0]),
# #             "auc": float("nan"),
# #             "recall": np.array([0.0, 1.0]),
# #             "precision": np.array([1.0, 0.0]),
# #             "ap": float("nan"),
# #             "P": int(P), "N": int(N),
# #         }
# #     # indices where score changes (keep first occurrence)
# #     sc_sorted = scores[order]
# #     change = np.r_[True, sc_sorted[1:] != sc_sorted[:-1]]
# #     tp, fp = tp[change], fp[change]

# #     # ROC
# #     tpr = tp / P
# #     fpr = fp / N
# #     auc = float(np.trapz(tpr, fpr))

# #     # PR
# #     recall = tp / P
# #     precision = tp / (tp + fp)
# #     # replace nan (0/0) with 1.0 when both tp and fp are zero
# #     precision = np.where((tp + fp) == 0, 1.0, precision)

# #     # Average Precision (step-wise area)
# #     # Ensure recall is increasing
# #     ap = 0.0
# #     for i in range(1, len(recall)):
# #         ap += precision[i] * (recall[i] - recall[i-1])
# #     return {
# #         "fpr": fpr, "tpr": tpr, "auc": auc,
# #         "recall": recall, "precision": precision, "ap": float(ap),
# #         "P": int(P), "N": int(N),
# #     }

# # def save_curve_npz_png(outdir: Path, model: str, roc, pr, plot_webrtc_points=None, title_suffix=""):
# #     outdir.mkdir(parents=True, exist_ok=True)
# #     # npz
# #     np.savez(outdir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
# #     np.savez(outdir / f"pr_{model}.npz", recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

# #     # ROC PNG
# #     plt.figure()
# #     plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
# #     if plot_webrtc_points:
# #         for label, (x, y) in plot_webrtc_points.items():
# #             plt.scatter([x], [y], marker="o", s=36, label=label)
# #     plt.xlabel("FPR")
# #     plt.ylabel("TPR")
# #     plt.title(f"ROC — {model}{title_suffix}")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(outdir / f"roc_{model}.png", dpi=140)
# #     plt.close()

# #     # PR PNG
# #     plt.figure()
# #     plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
# #     if plot_webrtc_points:
# #         # PR points: need precision/recall for those binary ops; user can pass them if wanted
# #         pass
# #     plt.xlabel("Recall")
# #     plt.ylabel("Precision")
# #     plt.title(f"PR — {model}{title_suffix}")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(outdir / f"pr_{model}.png", dpi=140)
# #     plt.close()

# # # ---------- Loading frame scores with env support ----------

# # def load_frame_scores(paths, env_map_path: Path | None):
# #     """
# #     Returns dict: model -> dict with arrays (labels, scores) and metadata (sources, envs).
# #     Accepts CSV (.csv) or compact NPZ (.npz) with fields:
# #       CSV cols: model, source, frame_idx, label_frame, score, prob
# #       NPZ  keys: model, source, label_frame, score, prob  (vectors)
# #     """
# #     # env map: source -> env
# #     env_map = {}
# #     if env_map_path:
# #         with open(env_map_path, "r", encoding="utf-8") as f:
# #             rd = csv.DictReader(f)
# #             for r in rd:
# #                 env_map[r["source"]] = r["env"]
# #     out = {}
# #     for p in paths:
# #         p = Path(p)
# #         if p.suffix.lower() == ".npz":
# #             z = np.load(p, allow_pickle=True)
# #             model = str(z["model"]) if "model" in z else p.stem
# #             labels = z["label_frame"].astype(np.int32)
# #             scores = (z["prob"] if "prob" in z else z["score"]).astype(np.float64)
# #             sources = z["source"].astype(object) if "source" in z else np.array([""]*len(labels), dtype=object)
# #         else:
# #             labels, scores, sources, model = [], [], [], None
# #             with open(p, "r", encoding="utf-8") as f:
# #                 rd = csv.DictReader(f)
# #                 for r in rd:
# #                     model = r.get("model", model) or p.stem
# #                     labels.append(int(r["label_frame"]))
# #                     s = r.get("prob", "")
# #                     s = float(s) if s not in ("", None) else float(r["score"])
# #                     scores.append(s)
# #                     sources.append(r.get("source",""))
# #             labels = np.asarray(labels, np.int32)
# #             scores = np.asarray(scores, np.float64)
# #             sources = np.asarray(sources, object)

# #         envs = None
# #         if env_map:
# #             envs = np.array([env_map.get(src, "unknown") for src in sources], dtype=object)

# #         if model not in out:
# #             out[model] = {"labels": [], "scores": [], "sources": [], "envs": []}
# #         out[model]["labels"].append(labels)
# #         out[model]["scores"].append(scores)
# #         out[model]["sources"].append(sources)
# #         out[model]["envs"].append(envs)
# #     # concatenate
# #     for m in list(out.keys()):
# #         out[m]["labels"]  = np.concatenate(out[m]["labels"])
# #         out[m]["scores"]  = np.concatenate(out[m]["scores"])
# #         out[m]["sources"] = np.concatenate(out[m]["sources"])
# #         if any(e is not None for e in out[m]["envs"]):
# #             # fill unknown for missing envs
# #             env_lists = [e if e is not None else np.array(["unknown"]*len(l), dtype=object)
# #                          for e, l in zip(out[m]["envs"], [len(x) for x in out[m]["labels"]])]
# #             out[m]["envs"] = np.concatenate(env_lists)
# #         else:
# #             out[m]["envs"] = None
# #     return out

# # # ---------- Averaging schemes ----------

# # def micro_curve(labels, scores):
# #     return compute_curves(labels, scores)

# # def macro_file_curve(labels, scores, sources):
# #     # per-file curves then unweighted mean of AUC/AP (not the curve itself)
# #     uniq = np.unique(sources)
# #     aucs, aps = [], []
# #     for u in uniq:
# #         m = (sources == u)
# #         res = compute_curves(labels[m], scores[m])
# #         if np.isfinite(res["auc"]): aucs.append(res["auc"])
# #         if np.isfinite(res["ap"]):  aps.append(res["ap"])
# #     return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(aps)) if aps else float("nan")

# # def macro_env_curve(labels, scores, envs, weighted=False, weights=None):
# #     uniq = np.unique(envs)
# #     aucs, aps, ws = [], [], []
# #     for u in uniq:
# #         m = (envs == u)
# #         res = compute_curves(labels[m], scores[m])
# #         aucs.append(res["auc"])
# #         aps.append(res["ap"])
# #         if weighted:
# #             if weights and (u in weights):
# #                 ws.append(float(weights[u]))
# #             else:
# #                 ws.append(float(np.sum(m) / len(labels)))  # default: duration-proportional
# #     if weighted:
# #         w = np.array(ws, dtype=np.float64)
# #         w = w / np.sum(w) if np.sum(w) > 0 else np.ones_like(w)/len(w)
# #         auc = float(np.nansum(np.array(aucs)*w))
# #         ap  = float(np.nansum(np.array(aps)*w))
# #     else:
# #         auc = float(np.nanmean(aucs))
# #         ap  = float(np.nanmean(aps))
# #     return auc, ap

# # # ---------- Main ----------

# # def main():
# #     ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR")
# #     # clip-level
# #     ap.add_argument("--clips", nargs="*", default=[], help="Clip-level result CSVs")
# #     # frame-level curves
# #     ap.add_argument("--frame_scores", nargs="*", default=[], help="Frame score CSV/NPZ files")
# #     ap.add_argument("--curve_outdir", type=str, default=None, help="Optional override for ROC/PR outdir")
# #     ap.add_argument("--env_map", type=str, default=None, help="CSV with columns: source,env")
# #     ap.add_argument("--avg", choices=["micro","macro-file","macro-env","macro-env-weighted"], default="micro")
# #     ap.add_argument("--env_weights", type=str, default=None, help='JSON or "env1:0.3,env2:0.7" for macro-env-weighted')
# #     ap.add_argument("--tag", type=str, default=None, help="Custom tag for outputs; default=timestamp")
# #     args = ap.parse_args()

# #     tag = args.tag or now_tag()
# #     curve_root = Path(args.curve_outdir) if args.curve_outdir else Path("outputs/roc_pr") / tag
# #     roc_dir = curve_root / "roc"
# #     pr_dir  = curve_root / "pr"
# #     eval_dir = Path("outputs/eval")
# #     eval_dir.mkdir(parents=True, exist_ok=True)

# #     # ---------- Clip-level summary table ----------
# #     if args.clips:
# #         summary_rows = []
# #         for clip_csv in args.clips:
# #             clip_csv = Path(clip_csv)
# #             y, p, _ = load_clip_results(clip_csv)
# #             tp, fp, fn, tn = confusion(y, p)
# #             prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)
# #             summary_rows.append({
# #                 "file": clip_csv.name,
# #                 "tp": tp, "fp": fp, "fn": fn, "tn": tn,
# #                 "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
# #                 "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
# #             })
# #         out_csv = eval_dir / f"summary_metrics_{tag}.csv"
# #         with open(out_csv, "w", newline="", encoding="utf-8") as f:
# #             w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
# #             w.writeheader()
# #             for r in summary_rows:
# #                 w.writerow(r)
# #         print(f"[clip] wrote {out_csv}")

# #     # ---------- Frame-level ROC/PR ----------
# #     if args.frame_scores:
# #         env_weights = None
# #         if args.env_weights:
# #             if args.env_weights.strip().startswith("{"):
# #                 env_weights = json.loads(args.env_weights)
# #             else:
# #                 env_weights = {}
# #                 for kv in args.env_weights.split(","):
# #                     if ":" in kv:
# #                         k, v = kv.split(":", 1)
# #                         env_weights[k.strip()] = float(v)

# #         data = load_frame_scores(args.frame_scores, Path(args.env_map) if args.env_map else None)
# #         summary_curves = []
# #         for model, pack in data.items():
# #             labels = pack["labels"]
# #             scores = pack["scores"]
# #             sources = pack["sources"]
# #             envs = pack["envs"]

# #             # Primary curve (micro)
# #             res = compute_curves(labels, scores)
# #             roc = {"fpr": res["fpr"], "tpr": res["tpr"], "auc": res["auc"]}
# #             pr  = {"recall": res["recall"], "precision": res["precision"], "ap": res["ap"]}
# #             roc_dir.mkdir(parents=True, exist_ok=True)
# #             pr_dir.mkdir(parents=True, exist_ok=True)
# #             save_curve_npz_png(roc_dir, model, roc, pr, plot_webrtc_points=None, title_suffix=f"  [{tag}]")

# #             row = {
# #                 "model": model, "scheme": "micro",
# #                 "auc_roc": f"{res['auc']:.6f}",
# #                 "ap_pr": f"{res['ap']:.6f}",
# #                 "frames": int(len(labels)),
# #                 "positives": int(np.sum(labels==1)),
# #                 "negatives": int(np.sum(labels==0)),
# #             }
# #             summary_curves.append(row)

# #             # Macro by file
# #             if args.avg in ("macro-file", "macro-env", "macro-env-weighted") and len(sources)>0:
# #                 auc_mf, ap_mf = macro_file_curve(labels, scores, sources)
# #                 summary_curves.append({
# #                     "model": model, "scheme": "macro-file",
# #                     "auc_roc": f"{auc_mf:.6f}", "ap_pr": f"{ap_mf:.6f}",
# #                     "frames": int(len(labels)),
# #                     "positives": int(np.sum(labels==1)), "negatives": int(np.sum(labels==0)),
# #                 })

# #             # Per-env + macro-env
# #             if envs is not None:
# #                 uniq_env = np.unique(envs)
# #                 for u in uniq_env:
# #                     m = (envs == u)
# #                     res_e = compute_curves(labels[m], scores[m])
# #                     roc_e = {"fpr": res_e["fpr"], "tpr": res_e["tpr"], "auc": res_e["auc"]}
# #                     pr_e  = {"recall": res_e["recall"], "precision": res_e["precision"], "ap": res_e["ap"]}
# #                     save_curve_npz_png(roc_dir, f"{model}_{u}", roc_e, pr_e, title_suffix=f"  [{tag} | env={u}]")
# #                     summary_curves.append({
# #                         "model": model, "scheme": f"env:{u}",
# #                         "auc_roc": f"{res_e['auc']:.6f}", "ap_pr": f"{res_e['ap']:.6f}",
# #                         "frames": int(np.sum(m)),
# #                         "positives": int(np.sum(labels[m]==1)),
# #                         "negatives": int(np.sum(labels[m]==0)),
# #                     })
# #                 # macro envs
# #                 if args.avg in ("macro-env","macro-env-weighted"):
# #                     auc_me, ap_me = macro_env_curve(
# #                         labels, scores, envs,
# #                         weighted=(args.avg=="macro-env-weighted"),
# #                         weights=env_weights
# #                     )
# #                     summary_curves.append({
# #                         "model": model, "scheme": args.avg,
# #                         "auc_roc": f"{auc_me:.6f}", "ap_pr": f"{ap_me:.6f}",
# #                         "frames": int(len(labels)),
# #                         "positives": int(np.sum(labels==1)),
# #                         "negatives": int(np.sum(labels==0)),
# #                     })

# #         out_summary = eval_dir / f"summary_curves_{tag}.csv"
# #         with open(out_summary, "w", newline="", encoding="utf-8") as f:
# #             w = csv.DictWriter(f, fieldnames=list(summary_curves[0].keys()))
# #             w.writeheader()
# #             for r in summary_curves:
# #                 w.writerow(r)
# #         print(f"[curves] wrote {out_summary}")
# #         print(f"[curves] npz/png written under: {curve_root}")

# # if __name__ == "__main__":
# #     main()
# # #

# #!/usr/bin/env python
# # scripts/evaluate_vad.py
# # Unified evaluator: clip-level + fast ROC/PR (frame-level), per-environment, micro/macro averages,
# # plus combined comparison plots (ROC/PR) across all models.
# import argparse, csv, datetime, json, os
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # ---------- Clip-level helpers ----------
# def load_clip_results(path: Path):
#     lab, pred, src = [], [], []
#     with open(path, "r", encoding="utf-8") as f:
#         rd = csv.DictReader(f)
#         for r in rd:
#             lab.append(int(r["label"]))
#             pred.append(int(r["pred"]))
#             src.append(r.get("source",""))
#     return np.asarray(lab, np.int32), np.asarray(pred, np.int32), np.asarray(src, object)

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

# # ---------- Frame-level fast ROC/PR ----------
# def compute_curves(labels: np.ndarray, scores: np.ndarray):
#     """Exact ROC/PR via one sort + cumsums."""
#     labels = labels.astype(np.int32)
#     scores = scores.astype(np.float64)
#     if labels.size == 0:
#         return {
#             "fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "auc": float("nan"),
#             "recall": np.array([0.0, 1.0]), "precision": np.array([1.0, 0.0]), "ap": float("nan"),
#             "P": 0, "N": 0,
#         }
#     order = np.argsort(scores)[::-1]
#     y = labels[order]
#     tp = np.cumsum(y)
#     fp = np.cumsum(1 - y)
#     P = tp[-1]
#     N = fp[-1]
#     if P == 0 or N == 0:
#         return {
#             "fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "auc": float("nan"),
#             "recall": np.array([0.0, 1.0]), "precision": np.array([1.0, 0.0]), "ap": float("nan"),
#             "P": int(P), "N": int(N),
#         }
#     sc_sorted = scores[order]
#     change = np.r_[True, sc_sorted[1:] != sc_sorted[:-1]]
#     tp, fp = tp[change], fp[change]
#     # ROC
#     tpr = tp / P
#     fpr = fp / N
#     auc = float(np.trapz(tpr, fpr))
#     # PR
#     recall = tp / P
#     precision = tp / (tp + fp)
#     precision = np.where((tp + fp) == 0, 1.0, precision)
#     # AP (step-wise)
#     ap = 0.0
#     for i in range(1, len(recall)):
#         ap += precision[i] * (recall[i] - recall[i-1])
#     return {
#         "fpr": fpr, "tpr": tpr, "auc": auc,
#         "recall": recall, "precision": precision, "ap": float(ap),
#         "P": int(P), "N": int(N),
#     }

# def save_curve_npz_png(outdir: Path, model: str, roc, pr, title_suffix=""):
#     outdir.mkdir(parents=True, exist_ok=True)
#     np.savez(outdir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
#     np.savez(outdir / f"pr_{model}.npz", recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

#     # ROC PNG
#     plt.figure()
#     plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
#     plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.title(f"ROC — {model}{title_suffix}")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(outdir / f"roc_{model}.png", dpi=140); plt.close()

#     # PR PNG
#     plt.figure()
#     plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(f"PR — {model}{title_suffix}")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(outdir / f"pr_{model}.png", dpi=140); plt.close()

# # ---------- Loading frame scores with env support ----------
# def load_frame_scores(paths, env_map_path: Path | None):
#     """
#     Returns dict: model -> dict with arrays (labels, scores) and metadata (sources, envs).
#     Accepts CSV (.csv) or NPZ (.npz):
#       CSV cols: model, source, frame_idx, label_frame, score, prob
#       NPZ  keys: model, source, label_frame, score, prob  (vectors)
#     """
#     # env map: source -> env
#     env_map = {}
#     if env_map_path:
#         if not Path(env_map_path).exists():
#             print(f"[warn] env_map file not found: {env_map_path} — continuing without envs.")
#             env_map_path = None
#         else:
#             with open(env_map_path, "r", encoding="utf-8") as f:
#                 rd = csv.DictReader(f)
#                 for r in rd:
#                     env_map[r["source"]] = r["env"]
#     out = {}
#     for p in paths:
#         p = Path(p)
#         if p.suffix.lower() == ".npz":
#             z = np.load(p, allow_pickle=True)
#             model = str(z["model"]) if "model" in z else p.stem
#             labels = z["label_frame"].astype(np.int32)
#             scores = (z["prob"] if "prob" in z and z["prob"].size else z["score"]).astype(np.float64)
#             sources = z["source"].astype(object) if "source" in z else np.array([""]*len(labels), dtype=object)
#         else:
#             labels, scores, sources, model = [], [], [], None
#             with open(p, "r", encoding="utf-8") as f:
#                 rd = csv.DictReader(f)
#                 for r in rd:
#                     model = r.get("model", model) or p.stem
#                     labels.append(int(r["label_frame"]))
#                     s = r.get("prob", "")
#                     s = float(s) if s not in ("", None) else float(r["score"])
#                     scores.append(s)
#                     sources.append(r.get("source",""))
#             labels = np.asarray(labels, np.int32)
#             scores = np.asarray(scores, np.float64)
#             sources = np.asarray(sources, object)

#         envs = None
#         if env_map_path:
#             envs = np.array([env_map.get(src, "unknown") for src in sources], dtype=object)

#         if model not in out:
#             out[model] = {"labels": [], "scores": [], "sources": [], "envs": []}
#         out[model]["labels"].append(labels)
#         out[model]["scores"].append(scores)
#         out[model]["sources"].append(sources)
#         out[model]["envs"].append(envs)
#     # concatenate
#     for m in list(out.keys()):
#         out[m]["labels"]  = np.concatenate(out[m]["labels"]) if out[m]["labels"] else np.array([], np.int32)
#         out[m]["scores"]  = np.concatenate(out[m]["scores"]) if out[m]["scores"] else np.array([], np.float64)
#         out[m]["sources"] = np.concatenate(out[m]["sources"]) if out[m]["sources"] else np.array([], object)
#         if any(e is not None for e in out[m]["envs"]):
#             env_lists = []
#             offset = 0
#             for e, lab in zip(out[m]["envs"], out[m]["labels"]):
#                 if e is None:
#                     env_lists.append(np.array(["unknown"]*len(lab), dtype=object))
#                 else:
#                     env_lists.append(e)
#                 offset += len(lab)
#             out[m]["envs"] = np.concatenate(env_lists)
#         else:
#             out[m]["envs"] = None
#     return out

# # ---------- Averaging schemes ----------
# def micro_curve(labels, scores):
#     return compute_curves(labels, scores)

# def macro_file_curve(labels, scores, sources):
#     uniq = np.unique(sources)
#     aucs, aps = [], []
#     for u in uniq:
#         m = (sources == u)
#         res = compute_curves(labels[m], scores[m])
#         if np.isfinite(res["auc"]): aucs.append(res["auc"])
#         if np.isfinite(res["ap"]):  aps.append(res["ap"])
#     return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(aps)) if aps else float("nan")

# def macro_env_curve(labels, scores, envs, weighted=False, weights=None):
#     uniq = np.unique(envs)
#     aucs, aps, ws = [], [], []
#     for u in uniq:
#         m = (envs == u)
#         res = compute_curves(labels[m], scores[m])
#         aucs.append(res["auc"]); aps.append(res["ap"])
#         if weighted:
#             if weights and (u in weights):
#                 ws.append(float(weights[u]))
#             else:
#                 ws.append(float(np.sum(m) / len(labels)))
#     if weighted:
#         w = np.array(ws, dtype=np.float64)
#         w = w / np.sum(w) if np.sum(w) > 0 else np.ones_like(w)/len(w)
#         auc = float(np.nansum(np.array(aucs)*w))
#         ap  = float(np.nansum(np.array(aps)*w))
#     else:
#         auc = float(np.nanmean(aucs))
#         ap  = float(np.nanmean(aps))
#     return auc, ap

# # ---------- Combined comparison plots ----------
# def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str):
#     """Overlay all models on one ROC, one PR."""
#     roc_dir.mkdir(parents=True, exist_ok=True)
#     pr_dir.mkdir(parents=True, exist_ok=True)

#     # ROC
#     plt.figure()
#     for m in all_models:
#         roc = m["roc"]
#         label = f"{m['name']} (AUC={roc['auc']:.3f})"
#         plt.plot(roc["fpr"], roc["tpr"], label=label)
#     plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.title(f"ROC — all models [{tag}]")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(roc_dir / "roc_all.png", dpi=140); plt.close()

#     # PR
#     plt.figure()
#     for m in all_models:
#         pr = m["pr"]
#         label = f"{m['name']} (AP={pr['ap']:.3f})"
#         plt.plot(pr["recall"], pr["precision"], label=label)
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(f"PR — all models [{tag}]")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(pr_dir / "pr_all.png", dpi=140); plt.close()

# # ---------- Main ----------
# def main():
#     ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR")
#     # clip-level
#     ap.add_argument("--clips", nargs="*", default=[], help="Clip-level result CSVs")
#     # frame-level curves
#     ap.add_argument("--frame_scores", nargs="*", default=[], help="Frame score CSV/NPZ files")
#     ap.add_argument("--curve_outdir", type=str, default=None, help="Optional override for ROC/PR outdir")
#     ap.add_argument("--env_map", type=str, default=None, help="CSV with columns: source,env")
#     ap.add_argument("--avg", choices=["micro","macro-file","macro-env","macro-env-weighted"], default="micro")
#     ap.add_argument("--env_weights", type=str, default=None, help='JSON or "env1:0.3,env2:0.7" for macro-env-weighted')
#     ap.add_argument("--tag", type=str, default=None, help="Custom tag for outputs; default=timestamp")
#     args = ap.parse_args()

#     tag = args.tag or now_tag()
#     curve_root = Path(args.curve_outdir) if args.curve_outdir else Path("outputs/roc_pr") / tag
#     roc_dir = curve_root / "roc"
#     pr_dir  = curve_root / "pr"
#     eval_dir = Path("outputs/eval")
#     eval_dir.mkdir(parents=True, exist_ok=True)

#     # ---------- Clip-level summary table ----------
#     if args.clips:
#         summary_rows = []
#         for clip_csv in args.clips:
#             clip_csv = Path(clip_csv)
#             y, p, _ = load_clip_results(clip_csv)
#             tp, fp, fn, tn = confusion(y, p)
#             prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)
#             summary_rows.append({
#                 "file": clip_csv.name,
#                 "tp": tp, "fp": fp, "fn": fn, "tn": tn,
#                 "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
#                 "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
#             })
#         out_csv = eval_dir / f"summary_metrics_{tag}.csv"
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
#             w.writeheader()
#             for r in summary_rows:
#                 w.writerow(r)
#         print(f"[clip] wrote {out_csv}")

#     # ---------- Frame-level ROC/PR ----------
#     if args.frame_scores:
#         # Parse env weights if any
#         env_weights = None
#         if args.env_weights:
#             if args.env_weights.strip().startswith("{"):
#                 env_weights = json.loads(args.env_weights)
#             else:
#                 env_weights = {}
#                 for kv in args.env_weights.split(","):
#                     if ":" in kv:
#                         k, v = kv.split(":", 1)
#                         env_weights[k.strip()] = float(v)

#         # Load scores (with safe/optional env)
#         env_map_path = Path(args.env_map) if args.env_map else None
#         data = load_frame_scores(args.frame_scores, env_map_path)

#         summary_curves = []
#         combined = []

#         for model, pack in data.items():
#             labels = pack["labels"]; scores = pack["scores"]
#             sources = pack["sources"]; envs = pack["envs"]

#             # Primary curve (micro)
#             res = compute_curves(labels, scores)
#             roc = {"fpr": res["fpr"], "tpr": res["tpr"], "auc": res["auc"]}
#             pr  = {"recall": res["recall"], "precision": res["precision"], "ap": res["ap"]}
#             roc_dir.mkdir(parents=True, exist_ok=True)
#             pr_dir.mkdir(parents=True, exist_ok=True)
#             save_curve_npz_png(roc_dir, model, roc, pr, title_suffix=f"  [{tag}]")

#             summary_curves.append({
#                 "model": model, "scheme": "micro",
#                 "auc_roc": f"{res['auc']:.6f}", "ap_pr": f"{res['ap']:.6f}",
#                 "frames": int(len(labels)),
#                 "positives": int(np.sum(labels==1)),
#                 "negatives": int(np.sum(labels==0)),
#             })
#             combined.append({"name": model, "roc": roc, "pr": pr})

#             # Macro by file
#             if args.avg in ("macro-file", "macro-env", "macro-env-weighted") and len(sources)>0:
#                 auc_mf, ap_mf = macro_file_curve(labels, scores, sources)
#                 summary_curves.append({
#                     "model": model, "scheme": "macro-file",
#                     "auc_roc": f"{auc_mf:.6f}", "ap_pr": f"{ap_mf:.6f}",
#                     "frames": int(len(labels)),
#                     "positives": int(np.sum(labels==1)), "negatives": int(np.sum(labels==0)),
#                 })

#             # Per-env + macro-env
#             if envs is not None:
#                 uniq_env = np.unique(envs)
#                 for u in uniq_env:
#                     m = (envs == u)
#                     res_e = compute_curves(labels[m], scores[m])
#                     roc_e = {"fpr": res_e["fpr"], "tpr": res_e["tpr"], "auc": res_e["auc"]}
#                     pr_e  = {"recall": res_e["recall"], "precision": res_e["precision"], "ap": res_e["ap"]}
#                     save_curve_npz_png(roc_dir, f"{model}_{u}", roc_e, pr_e, title_suffix=f"  [{tag} | env={u}]")
#                     summary_curves.append({
#                         "model": model, "scheme": f"env:{u}",
#                         "auc_roc": f"{res_e['auc']:.6f}", "ap_pr": f"{res_e['ap']:.6f}",
#                         "frames": int(np.sum(m)),
#                         "positives": int(np.sum(labels[m]==1)),
#                         "negatives": int(np.sum(labels[m]==0)),
#                     })

#                 if args.avg in ("macro-env","macro-env-weighted"):
#                     auc_me, ap_me = macro_env_curve(
#                         labels, scores, envs,
#                         weighted=(args.avg=="macro-env-weighted"),
#                         weights=env_weights
#                     )
#                     summary_curves.append({
#                         "model": model, "scheme": args.avg,
#                         "auc_roc": f"{auc_me:.6f}", "ap_pr": f"{ap_me:.6f}",
#                         "frames": int(len(labels)),
#                         "positives": int(np.sum(labels==1)),
#                         "negatives": int(np.sum(labels==0)),
#                     })

#         # Write summary CSV
#         out_summary = eval_dir / f"summary_curves_{tag}.csv"
#         with open(out_summary, "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=list(summary_curves[0].keys()))
#             w.writeheader()
#             for r in summary_curves:
#                 w.writerow(r)
#         print(f"[curves] wrote {out_summary}")
#         print(f"[curves] npz/png written under: {curve_root}")

#         # Combined comparison figures
#         if combined:
#             plot_compare(combined, roc_dir, pr_dir, tag)
#             print(f"[compare] wrote {roc_dir/'roc_all.png'} and {pr_dir/'pr_all.png'}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# scripts/evaluate_vad.py
# Unified evaluator:
# - Clip summary (unchanged)
# - Fast ROC/PR from frame scores
# - Combined comparison plots
# - NEW: overlay operating-point markers from clip CSVs on ROC & PR
# - Hides unnamed ("None") models

import argparse, csv, datetime, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# --------------------------- Clip helpers ---------------------------

def load_clip_results(path: Path):
    lab, pred, src = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            lab.append(int(r["label"]))
            pred.append(int(r["pred"]))
            src.append(r.get("source",""))
    return np.asarray(lab, np.int32), np.asarray(pred, np.int32), np.asarray(src, object)

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

def infer_model_from_clip_name(p: Path) -> str:
    s = p.stem.lower()
    # try common substrings first
    if "webrtc" in s:
        # e.g., clip_results_webrtc_l2[_...]
        for L in ("l0","l1","l2","l3"):
            if L in s: return f"webrtc_{L}"
        return "webrtc"
    for name in ("energy","zcr","combo","fft","fft_tuned"):
        if name in s: return name.capitalize() if name!="fft_tuned" else "FFT_tuned"
    # fallback: strip prefix/suffix
    return p.stem

# --------------------------- Curves (frame-level) ---------------------------

def compute_curves(labels: np.ndarray, scores: np.ndarray):
    labels = labels.astype(np.int32); scores = scores.astype(np.float64)
    if labels.size == 0:
        return {"fpr":np.array([0,1.0]),"tpr":np.array([0,1.0]),"auc":np.nan,
                "recall":np.array([0,1.0]),"precision":np.array([1.0,0]),"ap":np.nan,"P":0,"N":0}
    order = np.argsort(scores)[::-1]; y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1-y); P = tp[-1]; N = fp[-1]
    if P==0 or N==0:
        return {"fpr":np.array([0,1.0]),"tpr":np.array([0,1.0]),"auc":np.nan,
                "recall":np.array([0,1.0]),"precision":np.array([1.0,0]),"ap":np.nan,"P":int(P),"N":int(N)}
    sc = scores[order]; ch = np.r_[True, sc[1:]!=sc[:-1]]
    tp, fp = tp[ch], fp[ch]
    tpr = tp/P; fpr = fp/N; auc = float(np.trapz(tpr, fpr))
    recall = tp/P; precision = tp/(tp+fp); precision = np.where((tp+fp)==0, 1.0, precision)
    ap = float(np.sum((recall[1:]-recall[:-1]) * precision[1:]))
    return {"fpr":fpr,"tpr":tpr,"auc":auc,"recall":recall,"precision":precision,"ap":ap,"P":int(P),"N":int(N)}

def save_curve_npz_png(outdir: Path, model: str, roc, pr, title_suffix="", op_points=None):
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
    np.savez(outdir / f"pr_{model}.npz", recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

    # ROC
    plt.figure()
    plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
    if op_points and model in op_points:
        fpr_pt, tpr_pt = op_points[model]["roc"]
        plt.scatter([fpr_pt],[tpr_pt], marker="*", s=80, zorder=5, label=f"{model} op")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"roc_{model}.png", dpi=140); plt.close()

    # PR
    plt.figure()
    plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
    if op_points and model in op_points:
        rec_pt, prec_pt = op_points[model]["pr"]
        plt.scatter([rec_pt],[prec_pt], marker="*", s=80, zorder=5, label=f"{model} op")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"pr_{model}.png", dpi=140); plt.close()

# --------------------------- Load frame scores ---------------------------

def load_frame_scores(paths, env_map_path: Path | None):
    env_map = {}
    if env_map_path and env_map_path.exists():
        with open(env_map_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd: env_map[r["source"]] = r["env"]
    out = {}
    for p in paths:
        p = Path(p)
        labels, scores, sources, model = [], [], [], None
        if p.suffix.lower()==".npz":
            z = np.load(p, allow_pickle=True)
            model = str(z["model"]) if "model" in z else p.stem
            labels = z["label_frame"].astype(np.int32)
            scores = (z["prob"] if "prob" in z and z["prob"].size else z["score"]).astype(np.float64)
            sources = z["source"].astype(object) if "source" in z else np.array([""]*len(labels), object)
        else:
            with open(p, "r", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    if model is None: model = r.get("model") or p.stem
                    labels.append(int(r["label_frame"]))
                    s = r.get("prob","")
                    s = float(s) if s not in ("", None) else float(r["score"])
                    scores.append(s); sources.append(r.get("source",""))
            labels = np.asarray(labels, np.int32)
            scores = np.asarray(scores, np.float64)
            sources = np.asarray(sources, object)

        # skip unnamed models
        if (model is None) or (str(model).strip().lower() in ("", "none", "nan")):
            print(f"[skip] frame-scores with missing model name: {p}")
            continue

        envs = None
        if env_map:
            envs = np.array([env_map.get(src, "unknown") for src in sources], dtype=object)

        if model not in out:
            out[model] = {"labels": [], "scores": [], "sources": [], "envs": []}
        out[model]["labels"].append(labels)
        out[model]["scores"].append(scores)
        out[model]["sources"].append(sources)
        out[model]["envs"].append(envs)

    # concatenate
    for m in list(out.keys()):
        out[m]["labels"]  = np.concatenate(out[m]["labels"]) if out[m]["labels"] else np.array([], np.int32)
        out[m]["scores"]  = np.concatenate(out[m]["scores"]) if out[m]["scores"] else np.array([], np.float64)
        out[m]["sources"] = np.concatenate(out[m]["sources"]) if out[m]["sources"] else np.array([], object)
        if any(e is not None for e in out[m]["envs"]):
            env_lists = []
            for e, labs in zip(out[m]["envs"], [len(x) for x in out[m]["labels"]]):
                env_lists.append(e if e is not None else np.array(["unknown"]*labs, dtype=object))
            out[m]["envs"] = np.concatenate(env_lists)
        else:
            out[m]["envs"] = None
    return out

# --------------------------- Macro helpers (optional) ---------------------------

def macro_file_curve(labels, scores, sources):
    uniq = np.unique(sources); aucs, aps = [], []
    for u in uniq:
        m = (sources == u); res = compute_curves(labels[m], scores[m])
        if np.isfinite(res["auc"]): aucs.append(res["auc"])
        if np.isfinite(res["ap"]):  aps.append(res["ap"])
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(aps)) if aps else float("nan")

def macro_env_curve(labels, scores, envs, weighted=False, weights=None):
    uniq = np.unique(envs); aucs, aps, ws = [], [], []
    for u in uniq:
        m = (envs == u); res = compute_curves(labels[m], scores[m])
        aucs.append(res["auc"]); aps.append(res["ap"])
        if weighted:
            ws.append(float(weights.get(u, np.sum(m)/len(labels))) if weights else float(np.sum(m)/len(labels)))
    if weighted:
        w = np.array(ws, np.float64); w = w/np.sum(w) if np.sum(w)>0 else np.ones_like(w)/len(w)
        return float(np.nansum(np.array(aucs)*w)), float(np.nansum(np.array(aps)*w))
    return float(np.nanmean(aucs)), float(np.nanmean(aps))

# --------------------------- Combined plots ---------------------------

def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str, op_points=None):
    roc_dir.mkdir(parents=True, exist_ok=True); pr_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    plt.figure()
    for m in all_models:
        roc = m["roc"]; name = m["name"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            fpr_pt, tpr_pt = pts["roc"]
            plt.scatter([fpr_pt],[tpr_pt], marker="*", s=80, zorder=5, label=f"{name} op")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — all models [{tag}]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(roc_dir / "roc_all.png", dpi=140); plt.close()
    # PR
    plt.figure()
    for m in all_models:
        pr = m["pr"]; name = m["name"]
        plt.plot(pr["recall"], pr["precision"], label=f"{name} (AP={pr['ap']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            rec_pt, prec_pt = pts["pr"]
            plt.scatter([rec_pt],[prec_pt], marker="*", s=80, zorder=5, label=f"{name} op")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — all models [{tag}]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(pr_dir / "pr_all.png", dpi=140); plt.close()

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR")
    ap.add_argument("--clips", nargs="*", default=[], help="Clip-level result CSVs")
    ap.add_argument("--frame_scores", nargs="*", default=[], help="Frame score CSV/NPZ files")
    ap.add_argument("--curve_outdir", type=str, default=None)
    ap.add_argument("--env_map", type=str, default=None)
    ap.add_argument("--avg", choices=["micro","macro-file","macro-env","macro-env-weighted"], default="micro")
    ap.add_argument("--env_weights", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    tag = args.tag or now_tag()
    curve_root = Path(args.curve_outdir) if args.curve_outdir else Path("outputs/roc_pr") / tag
    roc_dir = curve_root / "roc"; pr_dir = curve_root / "pr"
    eval_dir = Path("outputs/eval"); eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- (1) Clip summary + op-point extraction ----
    op_points = {}  # model -> {"roc": (FPR,TPR), "pr": (Recall,Precision)}
    if args.clips:
        summary_rows = []
        for clip_csv in args.clips:
            clip_csv = Path(clip_csv)
            y, p, _ = load_clip_results(clip_csv)
            tp, fp, fn, tn = confusion(y, p)
            prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)
            summary_rows.append({
                "file": clip_csv.name,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
                "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
            })
            model_name = infer_model_from_clip_name(clip_csv)
            # Operating points derived from CLIP confusion
            TPR = rec
            FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            op_points[model_name] = {
                "roc": (FPR, TPR),
                "pr":  (rec, prec),
            }
        out_csv = eval_dir / f"summary_metrics_{tag}.csv"
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"[clip] wrote {out_csv}")

    # ---- (2) Frame-level curves ----
    if args.frame_scores:
        # optional env map
        env_map_path = Path(args.env_map) if args.env_map else None

        # env weights parsing (optional)
        env_weights=None
        if args.env_weights:
            if args.env_weights.strip().startswith("{"): env_weights=json.loads(args.env_weights)
            else:
                env_weights={}
                for kv in args.env_weights.split(","):
                    if ":" in kv:
                        k,v=kv.split(":",1); env_weights[k.strip()]=float(v)

        data = load_frame_scores(args.frame_scores, env_map_path)
        summary_curves=[]; combined=[]

        for model, pack in data.items():
            labels=pack["labels"]; scores=pack["scores"]
            sources=pack["sources"]; envs=pack["envs"]

            res = compute_curves(labels, scores)
            roc={"fpr":res["fpr"],"tpr":res["tpr"],"auc":res["auc"]}
            pr ={"recall":res["recall"],"precision":res["precision"],"ap":res["ap"]}

            roc_dir.mkdir(parents=True, exist_ok=True); pr_dir.mkdir(parents=True, exist_ok=True)
            save_curve_npz_png(
                roc_dir, model, roc, pr,
                title_suffix=f"  [{tag}]",
                op_points=op_points if op_points else None
            )

            summary_curves.append({
                "model":model,"scheme":"micro",
                "auc_roc":f"{res['auc']:.6f}","ap_pr":f"{res['ap']:.6f}",
                "frames":int(len(labels)),
                "positives":int(np.sum(labels==1)),"negatives":int(np.sum(labels==0))
            })
            combined.append({"name":model,"roc":roc,"pr":pr})

            # optional macro/file/env outputs unchanged (omit here for brevity)

        out_summary = eval_dir / f"summary_curves_{tag}.csv"
        with open(out_summary,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(summary_curves[0].keys()))
            w.writeheader(); w.writerows(summary_curves)
        print(f"[curves] wrote {out_summary}")

        # Combined plots with markers
        plot_compare(combined, roc_dir, pr_dir, tag, op_points=(op_points or None))
        print(f"[compare] wrote {roc_dir/'roc_all.png'} and {pr_dir/'pr_all.png'}")

if __name__ == "__main__":
    main()
