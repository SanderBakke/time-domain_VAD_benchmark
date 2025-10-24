# #!/usr/bin/env python
# # scripts/evaluate_vad.py
# # Unified evaluator:
# # - Auto-discover inputs by --tag (clips/frame-scores)
# # - Clip summary + confusion bar (optional)
# # - Fast ROC/PR (sort + cumsum); always produce combined ROC/PR (markers if no curves)
# # - Merge runtime metrics (sec/hour, mean, std) from runtime_summary.csv
# # - Copy runtime summary into eval folder for reproducibility
# # - Write a manifest of all inputs used
# # - Outputs isolated by tag + timestamp (no overwrites)

# import argparse, csv, datetime, json, shutil
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams["figure.autolayout"] = True

# # ---------- utils ----------
# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)
#     return p

# def read_csv_rows(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         rd = csv.DictReader(f)
#         return list(rd)

# def to_float(x, default=np.nan):
#     try:
#         return float(x)
#     except Exception:
#         return default

# # ---------- clip helpers ----------
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
#     rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
#     acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
#     return prec, rec, f1, acc

# def infer_model_from_clip_name(p: Path) -> str:
#     s = p.stem.lower()
#     if "webrtc" in s:
#         for L in ("l0","l1","l2","l3"):
#             if L in s: return f"webrtc_{L}"
#         return "webrtc"
#     for name in ("energy","zcr","combo","fft","fft_tuned"):
#         if name in s: return name.capitalize() if name!="fft_tuned" else "FFT_tuned"
#     return p.stem

# def plot_confusion_bar(summary_rows, out_png: Path, title: str):
#     labels = [r["file"] for r in summary_rows]
#     tp = np.array([r["tp"] for r in summary_rows])
#     fp = np.array([r["fp"] for r in summary_rows])
#     fn = np.array([r["fn"] for r in summary_rows])
#     tn = np.array([r["tn"] for r in summary_rows])
#     width = 0.6
#     plt.figure(figsize=(max(6, 1.0*len(labels)), 3.5))
#     bottom = np.zeros_like(tp, dtype=float)
#     for arr, name, color in [(tp,"TP","#2ca02c"), (fp,"FP","#d62728"),
#                              (fn,"FN","#ff7f0e"), (tn,"TN","#1f77b4")]:
#         plt.bar(labels, arr, width=width, bottom=bottom, label=name, color=color)
#         bottom += arr
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Count"); plt.title(title)
#     plt.legend(ncol=4, bbox_to_anchor=(0.5,1.15), loc="upper center")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=140)
#     plt.close()

# # ---------- curves (frame-level) ----------
# def compute_curves(labels: np.ndarray, scores: np.ndarray):
#     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
#     if labels.size == 0:
#         return {"fpr":np.array([]),"tpr":np.array([]),"auc":np.nan,
#                 "recall":np.array([]),"precision":np.array([]),"ap":np.nan,"P":0,"N":0}
#     order = np.argsort(scores)[::-1]; y = labels[order]
#     tp = np.cumsum(y); fp = np.cumsum(1-y); P = int(tp[-1]); N = int(fp[-1])
#     if P==0 or N==0:
#         return {"fpr":np.array([]),"tpr":np.array([]),"auc":np.nan,
#                 "recall":np.array([]),"precision":np.array([]),"ap":np.nan,"P":P,"N":N}
#     sc = scores[order]; ch = np.r_[True, sc[1:]!=sc[:-1]]
#     tp, fp = tp[ch], fp[ch]
#     tpr = tp/P; fpr = fp/N; auc = float(np.trapz(tpr, fpr))
#     recall = tp/P; precision = tp/(tp+fp); precision = np.where((tp+fp)==0, 1.0, precision)
#     ap = float(np.sum((recall[1:]-recall[:-1]) * precision[1:]))
#     return {"fpr":fpr,"tpr":tpr,"auc":auc,"recall":recall,"precision":precision,"ap":ap,"P":P,"N":N}

# def save_curve_npz_png(roc_dir: Path, pr_dir: Path, model: str, roc, pr, title_suffix="", op_points=None):
#     ensure_dir(roc_dir); ensure_dir(pr_dir)
#     # NPZ only if we have curves
#     if roc["fpr"].size:
#         np.savez(roc_dir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
#     if pr["recall"].size:
#         np.savez(pr_dir  / f"pr_{model}.npz",  recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

#     # ROC
#     plt.figure(figsize=(7,6))
#     if roc["fpr"].size:
#         plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
#     if op_points and model in op_points:
#         fpr_pt, tpr_pt = op_points[model]["roc"]
#         plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{model} op")
#     plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.title(f"ROC — {model}{title_suffix}")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(roc_dir / f"roc_{model}.png", dpi=150); plt.close()

#     # PR
#     plt.figure(figsize=(7,6))
#     if pr["recall"].size:
#         plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
#     if op_points and model in op_points:
#         rec_pt, prec_pt = op_points[model]["pr"]
#         plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{model} op")
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(f"PR — {model}{title_suffix}")
#     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#     plt.savefig(pr_dir / f"pr_{model}.png", dpi=150); plt.close()

# # ---------- load frame scores ----------
# def load_frame_scores(paths):
#     out = {}
#     for p in paths:
#         p = Path(p)
#         labels, scores, sources, model = [], [], [], None
#         if p.suffix.lower()==".npz":
#             z = np.load(p, allow_pickle=True)
#             model = str(z["model"]) if "model" in z else p.stem
#             labels = z["label_frame"].astype(np.int32)
#             scores = (z["prob"] if "prob" in z and z["prob"].size else z["score"]).astype(np.float64)
#             # sources optional
#         else:
#             with open(p, "r", encoding="utf-8") as f:
#                 rd = csv.DictReader(f)
#                 for r in rd:
#                     if model is None: model = r.get("model") or p.stem
#                     labels.append(int(r["label_frame"]))
#                     s = r.get("prob","")
#                     s = float(s) if s not in ("", None) else float(r["score"])
#                     scores.append(s)
#         if (model is None) or (str(model).strip().lower() in ("", "none", "nan")):
#             print(f"[skip] frame-scores with missing model name: {p}"); continue
#         labels = np.asarray(labels, np.int32)
#         scores = np.asarray(scores, np.float64)
#         if model not in out:
#             out[model] = {"labels": [], "scores": []}
#         out[model]["labels"].append(labels)
#         out[model]["scores"].append(scores)

#     for m in list(out.keys()):
#         out[m]["labels"]  = np.concatenate(out[m]["labels"]) if out[m]["labels"] else np.array([], np.int32)
#         out[m]["scores"]  = np.concatenate(out[m]["scores"]) if out[m]["scores"] else np.array([], np.float64)
#     return out

# # ---------- combined plots ----------
# def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str, op_points=None):
#     ensure_dir(roc_dir); ensure_dir(pr_dir)
#     # ROC
#     plt.figure(figsize=(7.5, 6))
#     drew_any = False
#     for m in all_models:
#         roc = m["roc"]; name = m["name"]
#         if roc["fpr"].size:
#             plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
#             drew_any = True
#     if op_points:
#         for name, pts in op_points.items():
#             fpr_pt, tpr_pt = pts["roc"]
#             plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{name} op")
#             drew_any = True
#     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
#     plt.title(f"ROC — all models [{tag}]")
#     if drew_any:
#         plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#         plt.savefig(roc_dir / f"roc_all_{tag}.png", dpi=150)
#     plt.close()

#     # PR
#     plt.figure(figsize=(7.5, 6))
#     drew_any = False
#     for m in all_models:
#         pr = m["pr"]; name = m["name"]
#         if pr["recall"].size:
#             plt.plot(pr["recall"], pr["precision"], label=f"{name} (AP={pr['ap']:.3f})")
#             drew_any = True
#     if op_points:
#         for name, pts in op_points.items():
#             rec_pt, prec_pt = pts["pr"]
#             plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{name} op")
#             drew_any = True
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(f"PR — all models [{tag}]")
#     if drew_any:
#         plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#         plt.savefig(pr_dir / f"pr_all_{tag}.png", dpi=150)
#     plt.close()

# # ---------- summary table figure ----------
# def draw_summary_table(df_rows, out_png: Path, title: str):
#     # df_rows: list of dicts with final columns already as strings/numbers
#     cols = ["Model","TP","FP","FN","TN","Precision","Recall","F1","Accuracy","AUC","AP","sec/hour","timing mean","± std"]
#     data = []
#     for r in df_rows:
#         data.append([
#             r.get("Model",""),
#             r.get("TP",""), r.get("FP",""), r.get("FN",""), r.get("TN",""),
#             f"{to_float(r.get('Precision'),0):.4f}",
#             f"{to_float(r.get('Recall'),0):.4f}",
#             f"{to_float(r.get('F1'),0):.4f}",
#             f"{to_float(r.get('Accuracy'),0):.4f}",
#             ("" if np.isnan(to_float(r.get("AUC"),np.nan)) else f"{to_float(r.get('AUC')):.3f}"),
#             ("" if np.isnan(to_float(r.get("AP"),np.nan))  else f"{to_float(r.get('AP')):.3f}"),
#             ("" if np.isnan(to_float(r.get('sec/hour'),np.nan)) else f"{to_float(r.get('sec/hour')):.2f}"),
#             ("" if np.isnan(to_float(r.get('timing mean'),np.nan)) else f"{to_float(r.get('timing mean')):.2f}"),
#             ("" if np.isnan(to_float(r.get('± std'),np.nan)) else f"{to_float(r.get('± std')):.2f}"),
#         ])
#     fig, ax = plt.subplots(figsize=(18, 3 + 0.4*len(data)))
#     ax.axis("off")
#     table = ax.table(cellText=data, colLabels=cols, loc="center")
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.4)
#     ax.set_title(title, fontsize=16, pad=18)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=150)
#     plt.close()

# # ---------- auto-discovery ----------
# def discover_by_tag(tag: str):
#     clips = sorted([p for p in (Path("outputs/clips")/tag).glob("clip_results_*.csv")])
#     frames = sorted([p for p in (Path("outputs/frames")/tag).glob("*.csv")])
#     return clips, frames

# # ---------- runtime merge ----------
# def normalize_model_name(s: str) -> str:
#     s = s.strip()
#     aliases = {
#         "energy":"Energy", "Energy":"Energy",
#         "zcr":"ZCR", "ZCR":"ZCR",
#         "combo":"Combo", "Combo":"Combo",
#         "webrtc_l0":"webrtc_l0", "webrtc_l1":"webrtc_l1",
#         "webrtc_l2":"webrtc_l2", "webrtc_l3":"webrtc_l3",
#         "WebRTC-L2":"webrtc_l2", "WebRTC-L3":"webrtc_l3",
#         "webrtc":"webrtc",
#     }
#     return aliases.get(s, s)

# def merge_runtime(summary_rows, runtime_csv: Path):
#     if not runtime_csv.exists():
#         return summary_rows, None
#     rt = read_csv_rows(runtime_csv)
#     # build map
#     rmap = {}
#     for r in rt:
#         m = normalize_model_name(r.get("model",""))
#         rmap[m] = {
#             "sec/hour": to_float(r.get("sec_per_hour_mean", r.get("sec_per_hour"))),
#             "timing mean": to_float(r.get("sec_per_hour_mean")),
#             "± std": to_float(r.get("sec_per_hour_std")),
#         }
#     # apply
#     for r in summary_rows:
#         m = normalize_model_name(r["Model"])
#         if m in rmap:
#             r["sec/hour"]   = rmap[m]["sec/hour"]
#             r["timing mean"]= rmap[m]["timing mean"]
#             r["± std"]      = rmap[m]["± std"]
#         else:
#             r["sec/hour"]   = np.nan
#             r["timing mean"]= np.nan
#             r["± std"]      = np.nan
#     return summary_rows, rt

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR + runtime merge")
#     ap.add_argument("--clips", nargs="*", default=[], help="Clip-level result CSVs")
#     ap.add_argument("--frame_scores", nargs="*", default=[], help="Frame score CSV/NPZ files")
#     ap.add_argument("--curve_outdir", type=str, default=None, help="Override for ROC/PR root")
#     ap.add_argument("--tag", type=str, default=None, help="Dataset/eval tag (light/heavy/clean)")
#     ap.add_argument("--make_confbar", action="store_true", help="Save a confusion bar from clip CSVs")
#     ap.add_argument("--runtime_csv", type=str, default="outputs/runtime/runtime_summary.csv",
#                     help="Path to runtime summary to merge & copy")
#     args = ap.parse_args()

#     # auto-discover by tag if lists empty
#     if args.tag and not args.clips and not args.frame_scores:
#         auto_clips, auto_frames = discover_by_tag(args.tag)
#         args.clips = [str(p) for p in auto_clips]
#         args.frame_scores = [str(p) for p in auto_frames]

#     tag = args.tag or now_tag()
#     stamp = now_tag()

#     # output roots
#     eval_run_dir  = ensure_dir(Path("outputs/eval") / f"{tag}__{stamp}")
#     rocpr_root    = Path(args.curve_outdir) if args.curve_outdir else Path("outputs/roc_pr") / f"{tag}__{stamp}"
#     roc_dir, pr_dir = ensure_dir(rocpr_root / "roc"), ensure_dir(rocpr_root / "pr")

#     manifest = {
#         "tag": tag,
#         "timestamp": stamp,
#         "clips": args.clips,
#         "frame_scores": args.frame_scores,
#         "runtime_csv": args.runtime_csv,
#     }

#     # -------- (1) Clip summary (+ op points) --------
#     op_points = {}
#     clip_summary_rows = []
#     if args.clips:
#         for clip_csv in args.clips:
#             clip_csv = Path(clip_csv)
#             y, p, _ = load_clip_results(clip_csv)
#             tp, fp, fn, tn = confusion(y, p)
#             prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)
#             model_name = infer_model_from_clip_name(clip_csv)
#             clip_summary_rows.append({
#                 "file": clip_csv.name,
#                 "Model": model_name,
#                 "TP": tp, "FP": fp, "FN": fn, "TN": tn,
#                 "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc,
#                 "AUC": np.nan, "AP": np.nan,  # filled later if curves exist
#             })
#             TPR = rec
#             FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
#             op_points[model_name] = {"roc": (FPR, TPR), "pr": (rec, prec)}

#         # write raw clip summary
#         out_csv = eval_run_dir / f"summary_metrics_{tag}.csv"
#         with open(out_csv,"w",newline="",encoding="utf-8") as f:
#             w=csv.DictWriter(f, fieldnames=list(clip_summary_rows[0].keys()))
#             w.writeheader(); w.writerows(clip_summary_rows)
#         print(f"[clip] wrote {out_csv}")

#         if args.make_confbar:
#             conf_png = eval_run_dir / f"confusion_bar_{tag}.png"
#             plot_confusion_bar(
#                 [{"file":r["Model"],"tp":r["TP"],"fp":r["FP"],"fn":r["FN"],"tn":r["TN"]} for r in clip_summary_rows],
#                 conf_png,
#                 f"Confusion counts — {tag}"
#             )
#             print(f"[clip] wrote {conf_png}")
#     else:
#         print("[info] no clip CSVs provided/found.")

#     # -------- (2) Frame-level curves --------
#     curve_summary_rows = []
#     combined = []
#     model_to_auc_ap = {}

#     if args.frame_scores:
#         data = load_frame_scores(args.frame_scores)
#         if not data:
#             print("[curves] no valid frame-score files found.")
#         else:
#             print(f"[curves] models: {', '.join(sorted(data.keys()))}")
#             for model, pack in data.items():
#                 labels=pack["labels"]; scores=pack["scores"]
#                 res = compute_curves(labels, scores)
#                 roc={"fpr":res["fpr"],"tpr":res["tpr"],"auc":res["auc"]}
#                 pr ={"recall":res["recall"],"precision":res["precision"],"ap":res["ap"]}
#                 save_curve_npz_png(roc_dir, pr_dir, model, roc, pr, title_suffix=f"  [{tag}]", op_points=op_points)
#                 curve_summary_rows.append({
#                     "model": model, "scheme":"micro",
#                     "auc_roc": ("" if np.isnan(res["auc"]) else f"{res['auc']:.6f}"),
#                     "ap_pr":  ("" if np.isnan(res["ap"])  else f"{res['ap']:.6f}"),
#                     "frames": int(len(labels)),
#                     "positives": int(np.sum(labels==1)),
#                     "negatives": int(np.sum(labels==0)),
#                 })
#                 model_to_auc_ap[model] = (res["auc"], res["ap"])
#                 combined.append({"name":model,"roc":roc,"pr":pr})

#             # per-model curve summary CSV
#             out_curves = eval_run_dir / f"summary_curves_{tag}.csv"
#             with open(out_curves,"w",newline="",encoding="utf-8") as f:
#                 w=csv.DictWriter(f, fieldnames=list(curve_summary_rows[0].keys()))
#                 w.writeheader(); w.writerows(curve_summary_rows)
#             print(f"[curves] wrote {out_curves}")

#     # always draw combined figures (markers if needed)
#     plot_compare(combined, roc_dir, pr_dir, tag, op_points=(op_points or None))
#     print(f"[compare] wrote combined ROC/PR under: {rocpr_root}")

#     # -------- (3) Merge AUC/AP into clip table --------
#     if clip_summary_rows:
#         for r in clip_summary_rows:
#             m = r["Model"]
#             if m in model_to_auc_ap:
#                 auc, ap = model_to_auc_ap[m]
#                 r["AUC"] = auc
#                 r["AP"]  = ap

#     # -------- (4) Merge runtime + copy runtime CSV --------
#     runtime_csv = Path(args.runtime_csv)
#     clip_summary_rows, rt_rows = merge_runtime(clip_summary_rows, runtime_csv)
#     if runtime_csv.exists():
#         # copy into eval folder
#         copied = eval_run_dir / f"runtime_summary_{tag}.csv"
#         shutil.copy2(runtime_csv, copied)
#         print(f"[runtime] copied {runtime_csv} -> {copied}")

#     # -------- (5) Final table PNG --------
#     if clip_summary_rows:
#         # Write merged CSV (final)
#         out_csv_final = eval_run_dir / f"summary_metrics_merged_{tag}.csv"
#         field_order = ["Model","TP","FP","FN","TN","Precision","Recall","F1","Accuracy","AUC","AP","sec/hour","timing mean","± std","file"]
#         with open(out_csv_final,"w",newline="",encoding="utf-8") as f:
#             w=csv.DictWriter(f, fieldnames=field_order)
#             w.writeheader()
#             for r in clip_summary_rows:
#                 row = {k:r.get(k,"") for k in field_order}
#                 w.writerow(row)
#         print(f"[table] wrote {out_csv_final}")

#         # Draw the paper-ready table
#         tbl_png = eval_run_dir / f"summary_table_{tag}.png"
#         draw_summary_table(clip_summary_rows, tbl_png, f"VAD Summary — {tag}")
#         print(f"[table] wrote {tbl_png}")

#     # -------- (6) Manifest --------
#     manifest["resolved"] = {
#         "clips": args.clips,
#         "frame_scores": args.frame_scores,
#         "eval_dir": str(eval_run_dir),
#         "roc_pr_dir": str(rocpr_root),
#     }
#     with open(eval_run_dir / f"eval_manifest_{tag}.json","w",encoding="utf-8") as f:
#         json.dump(manifest, f, indent=2)
#     print(f"[manifest] {eval_run_dir / f'eval_manifest_{tag}.json'}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# scripts/evaluate_vad.py
# Unified evaluator:
# - Clip summary (+ optional confusion bar)
# - ROC/PR from frame scores (per-model + combined)
# - Pulls runtime_summary_<TAG>__<STAMP>.csv and joins into table
# - All outputs isolated under outputs/eval/<TAG>__<STAMP> and outputs/roc_pr/<TAG>__<STAMP>

import argparse, csv, datetime, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True

def now_tag(): return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# -------- basic helpers --------
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
    if "webrtc" in s:
        for L in ("l0","l1","l2","l3"):
            if L in s: return f"webrtc_{L}"
        return "webrtc"
    for name in ("energy","zcr","combo"):
        if name in s: return name.capitalize()
    return p.stem

def plot_confusion_bar(summary_rows, out_png: Path, title: str):
    labels = [r["model"] for r in summary_rows]
    tp = np.array([r["tp"] for r in summary_rows])
    fp = np.array([r["fp"] for r in summary_rows])
    fn = np.array([r["fn"] for r in summary_rows])
    tn = np.array([r["tn"] for r in summary_rows])
    width = 0.6
    plt.figure(figsize=(max(6, 1.0*len(labels)), 3.8))
    bottom = np.zeros_like(tp, dtype=float)
    for arr, name, color in [(tp,"TP","#2ca02c"), (fp,"FP","#d62728"),
                             (fn,"FN","#ff7f0e"), (tn,"TN","#1f77b4")]:
        plt.bar(labels, arr, width=width, bottom=bottom, label=name, color=color)
        bottom += arr
    plt.xticks(rotation=20, ha="right"); plt.ylabel("Count"); plt.title(title)
    plt.legend(ncol=4, bbox_to_anchor=(0.5,1.18), loc="upper center")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# -------- ROC/PR from frames --------
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

def save_curve_npz_png(roc_dir: Path, pr_dir: Path, model: str, roc, pr, title_suffix="", op_points=None):
    roc_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)
    np.savez(roc_dir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
    np.savez(pr_dir  / f"pr_{model}.npz",  recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

    # ROC
    plt.figure()
    plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
    if op_points and model in op_points:
        fpr_pt, tpr_pt = op_points[model]["roc"]
        plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{model} op")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(roc_dir / f"roc_{model}.png", dpi=150); plt.close()

    # PR
    plt.figure()
    plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
    if op_points and model in op_points:
        rec_pt, prec_pt = op_points[model]["pr"]
        plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{model} op")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(pr_dir / f"pr_{model}.png", dpi=150); plt.close()

def load_frame_scores(paths):
    out = {}
    for p in paths:
        p = Path(p)
        model = None
        labels, scores, sources = [], [], []
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                if model is None:
                    model = r.get("model") or p.stem
                labels.append(int(r.get("label_frame", 0)))   # we store decisions here; OK for OP overlay/consistency
                s = r.get("prob","")
                s = float(s) if s not in ("", None) else float(r.get("score", 0.0))
                scores.append(s)
                sources.append(r.get("source",""))
        if model is None or str(model).strip().lower() in ("", "none", "nan"):
            continue
        if model not in out:
            out[model] = {"labels": [], "scores": []}
        out[model]["labels"].append(np.asarray(labels, np.int32))
        out[model]["scores"].append(np.asarray(scores, np.float64))
    # concat
    for m in list(out.keys()):
        out[m]["labels"] = np.concatenate(out[m]["labels"]) if out[m]["labels"] else np.array([], np.int32)
        out[m]["scores"] = np.concatenate(out[m]["scores"]) if out[m]["scores"] else np.array([], np.float64)
    return out

def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str, op_points=None):
    # ROC
    plt.figure(figsize=(7.5, 6))
    for m in all_models:
        r, name = m["roc"], m["name"]
        plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['auc']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            fpr_pt, tpr_pt = pts["roc"]
            plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{name} op")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — all models [{tag}]"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(roc_dir / f"roc_all_{tag}.png", dpi=150); plt.close()

    # PR
    plt.figure(figsize=(7.5, 6))
    for m in all_models:
        prc, name = m["pr"], m["name"]
        plt.plot(prc["recall"], prc["precision"], label=f"{name} (AP={prc['ap']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            rec_pt, prec_pt = pts["pr"]
            plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{name} op")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — all models [{tag}]"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(pr_dir / f"pr_all_{tag}.png", dpi=150); plt.close()

# -------- runtime helpers --------
def read_runtime_summary(summary_csv: Path) -> dict:
    if not summary_csv or not summary_csv.exists():
        return {}
    out = {}
    with open(summary_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            k = (r.get("model","") or "").strip()
            if k:
                out[k] = r
    return out

def main():
    ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR + runtime")
    ap.add_argument("--clips", nargs="+", required=True, help="Clip-level result CSVs")
    ap.add_argument("--frame_scores", nargs="*", default=[], help="Frame score CSVs (any models)")
    ap.add_argument("--tag", type=str, required=True, help="Tag used during run_vad / run_vad_webrtc")
    ap.add_argument("--runtime_summary_dir", type=str, required=True,
                    help="Point to outputs/runtime/<TAG>__<STAMP> that matches this eval")
    ap.add_argument("--make_confbar", action="store_true")
    args = ap.parse_args()

    stamp = now_tag()
    eval_dir = Path("outputs/eval") / f"{args.tag}__{stamp}"
    rocpr_root = Path("outputs/roc_pr") / f"{args.tag}__{stamp}"
    roc_dir = rocpr_root / "roc"; pr_dir = rocpr_root / "pr"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- (1) Clip summary + OP extraction
    op_points = {}
    summary_rows = []
    for clip_csv in args.clips:
        clip_csv = Path(clip_csv)
        y, p, _ = load_clip_results(clip_csv)
        tp, fp, fn, tn = confusion(y, p)
        prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)
        model_name = infer_model_from_clip_name(clip_csv)
        summary_rows.append({
            "model": model_name,
            "file": clip_csv.name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
            "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
        })
        # Operating point marker from clip confusion
        TPR = rec
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        op_points[model_name] = {"roc": (FPR, TPR), "pr": (rec, prec)}

    # runtime summary (same TAG+STAMP run folder you pass in)
    rt_summary = read_runtime_summary(Path(args.runtime_summary_dir) / f"runtime_summary_{Path(args.runtime_summary_dir).name}.csv")

    # ---- (2) Frame-level curves & plots
    combined = []
    if args.frame_scores:
        data = load_frame_scores(args.frame_scores)
        for model, pack in data.items():
            res = compute_curves(pack["labels"], pack["scores"])
            roc={"fpr":res["fpr"],"tpr":res["tpr"],"auc":res["auc"]}
            pr ={"recall":res["recall"],"precision":res["precision"],"ap":res["ap"]}
            save_curve_npz_png(roc_dir, pr_dir, model, roc, pr, title_suffix=f"  [{args.tag}]", op_points=op_points)
            combined.append({"name":model,"roc":roc,"pr":pr})

    # write clip summary table (joined with runtime)
    out_csv = eval_dir / f"summary_metrics_{args.tag}.csv"
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        cols = ["model","file","tp","fp","fn","tn","precision","recall","f1","acc",
                "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary_rows:
            rt = rt_summary.get(r["model"], {})
            w.writerow({
                **r,
                "sec_per_hour": rt.get("sec_per_hour",""),
                "sec_per_hour_mean": rt.get("sec_per_hour_mean",""),
                "sec_per_hour_std": rt.get("sec_per_hour_std",""),
                "repeats": rt.get("repeats",""),
            })
    print(f"[clip] wrote {out_csv}")

    if args.make_confbar:
        conf_png = eval_dir / f"confusion_bar_{args.tag}.png"
        plot_confusion_bar(summary_rows, conf_png, f"Confusion counts — {args.tag}")
        print(f"[clip] wrote {conf_png}")

    if combined:
        plot_compare(combined, roc_dir, pr_dir, args.tag, op_points=op_points)
        print(f"[roc/pr] wrote {roc_dir/'roc_all_'+args.tag+'.png'} and {pr_dir/'pr_all_'+args.tag+'.png'}")
    else:
        print("[info] no frame scores passed; ROC/PR skipped.")

if __name__ == "__main__":
    main()
