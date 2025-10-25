# # # # #!/usr/bin/env python3
# # # # # scripts/evaluate_vad.py
# # # # # - Clip summary (+ optional confusion bar)
# # # # # - Auto-discover frame scores by tag (or accept explicit list)
# # # # # - ROC/PR per model + combined, with OP markers
# # # # # - Merge runtime summaries into the metrics table
# # # # # - All outputs isolated under outputs/eval/<TAG>__<STAMP>/

# # # # import argparse, csv, datetime, json, shutil
# # # # from pathlib import Path
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # plt.rcParams["figure.autolayout"] = True

# # # # def now_tag():
# # # #     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # # # # ---------------- Clip helpers ----------------
# # # # def load_clip_results(path: Path):
# # # #     lab, pred, src = [], [], []
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         rd = csv.DictReader(f)
# # # #         for r in rd:
# # # #             lab.append(int(r["label"]))
# # # #             pred.append(int(r["pred"]))
# # # #             src.append(r.get("source",""))
# # # #     return np.asarray(lab, np.int32), np.asarray(pred, np.int32), np.asarray(src, object)

# # # # def confusion(y, p):
# # # #     tn = int(((y == 0) & (p == 0)).sum())
# # # #     tp = int(((y == 1) & (p == 1)).sum())
# # # #     fp = int(((y == 0) & (p == 1)).sum())
# # # #     fn = int(((y == 1) & (p == 0)).sum())
# # # #     return tp, fp, fn, tn

# # # # def metrics_from_conf(tp, fp, fn, tn):
# # # #     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# # # #     rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# # # #     f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
# # # #     acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
# # # #     return prec, rec, f1, acc

# # # # def infer_model_from_clip_name(p: Path) -> str:
# # # #     s = p.stem.lower()
# # # #     if "webrtc" in s:
# # # #         for L in ("l0","l1","l2","l3"):
# # # #             if L in s: return f"webrtc_{L}"
# # # #         return "webrtc"
# # # #     for name in ("energy","zcr","combo"):
# # # #         if name in s: return name.capitalize()
# # # #     return p.stem

# # # # def plot_confusion_bar(summary_rows, out_png: Path, title: str):
# # # #     labels = [r["model"] for r in summary_rows]
# # # #     tp = np.array([r["tp"] for r in summary_rows])
# # # #     fp = np.array([r["fp"] for r in summary_rows])
# # # #     fn = np.array([r["fn"] for r in summary_rows])
# # # #     tn = np.array([r["tn"] for r in summary_rows])
# # # #     width = 0.6
# # # #     plt.figure(figsize=(max(6, 1.0*len(labels)), 3.5))
# # # #     bottom = np.zeros_like(tp, dtype=float)
# # # #     for arr, name, color in [(tp,"TP","#2ca02c"), (fp,"FP","#d62728"),
# # # #                              (fn,"FN","#ff7f0e"), (tn,"TN","#1f77b4")]:
# # # #         plt.bar(labels, arr, width=width, bottom=bottom, label=name, color=color)
# # # #         bottom += arr
# # # #     plt.xticks(rotation=0)
# # # #     plt.ylabel("Count"); plt.title(title)
# # # #     plt.legend(ncol=4, bbox_to_anchor=(0.5,1.15), loc="upper center")
# # # #     plt.tight_layout()
# # # #     plt.savefig(out_png, dpi=140)
# # # #     plt.close()

# # # # # ---------------- Curves (frame-level) ----------------
# # # # def compute_curves(labels: np.ndarray, scores: np.ndarray):
# # # #     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
# # # #     if labels.size == 0:
# # # #         return {"fpr":np.array([0,1.0]),"tpr":np.array([0,1.0]),"auc":np.nan,
# # # #                 "recall":np.array([0,1.0]),"precision":np.array([1.0,0]),"ap":np.nan,"P":0,"N":0}
# # # #     order = np.argsort(scores)[::-1]; y = labels[order]
# # # #     tp = np.cumsum(y); fp = np.cumsum(1-y); P = tp[-1]; N = fp[-1]
# # # #     if P==0 or N==0:
# # # #         return {"fpr":np.array([0,1.0]),"tpr":np.array([0,1.0]),"auc":np.nan,
# # # #                 "recall":np.array([0,1.0]),"precision":np.array([1.0,0]),"ap":np.nan,"P":int(P),"N":int(N)}
# # # #     sc = scores[order]; ch = np.r_[True, sc[1:]!=sc[:-1]]
# # # #     tp, fp = tp[ch], fp[ch]
# # # #     tpr = tp/P; fpr = fp/N; auc = float(np.trapz(tpr, fpr))
# # # #     recall = tp/P; precision = tp/(tp+fp); precision = np.where((tp+fp)==0, 1.0, precision)
# # # #     ap = float(np.sum((recall[1:]-recall[:-1]) * precision[1:]))
# # # #     return {"fpr":fpr,"tpr":tpr,"auc":auc,"recall":recall,"precision":precision,"ap":ap,"P":int(P),"N":int(N)}

# # # # def save_curve_npz_png(roc_dir: Path, pr_dir: Path, model: str, roc, pr, title_suffix="", op_points=None):
# # # #     roc_dir.mkdir(parents=True, exist_ok=True)
# # # #     pr_dir.mkdir(parents=True, exist_ok=True)
# # # #     np.savez(roc_dir / f"roc_{model}.npz", fpr=roc["fpr"], tpr=roc["tpr"], auc=roc["auc"])
# # # #     np.savez(pr_dir  / f"pr_{model}.npz",  recall=pr["recall"], precision=pr["precision"], ap=pr["ap"])

# # # #     # ROC
# # # #     plt.figure()
# # # #     plt.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={roc['auc']:.3f})")
# # # #     if op_points and model in op_points:
# # # #         fpr_pt, tpr_pt = op_points[model]["roc"]
# # # #         plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{model} op")
# # # #     plt.xlabel("FPR"); plt.ylabel("TPR")
# # # #     plt.title(f"ROC — {model}{title_suffix}")
# # # #     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
# # # #     plt.savefig(roc_dir / f"roc_{model}.png", dpi=140); plt.close()

# # # #     # PR
# # # #     plt.figure()
# # # #     plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
# # # #     if op_points and model in op_points:
# # # #         rec_pt, prec_pt = op_points[model]["pr"]
# # # #         plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{model} op")
# # # #     plt.xlabel("Recall"); plt.ylabel("Precision")
# # # #     plt.title(f"PR — {model}{title_suffix}")
# # # #     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
# # # #     plt.savefig(pr_dir / f"pr_{model}.png", dpi=140); plt.close()

# # # # def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str, op_points=None):
# # # #     # ROC
# # # #     plt.figure(figsize=(7.5, 6))
# # # #     for m in all_models:
# # # #         roc = m["roc"]; name = m["name"]
# # # #         plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
# # # #     if op_points:
# # # #         for name, pts in op_points.items():
# # # #             fpr_pt, tpr_pt = pts["roc"]
# # # #             plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{name} op")
# # # #     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
# # # #     plt.title(f"ROC — all models [{tag}]")
# # # #     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
# # # #     plt.savefig(roc_dir / f"roc_all_{tag}.png", dpi=150); plt.close()
# # # #     # PR
# # # #     plt.figure(figsize=(7.5, 6))
# # # #     for m in all_models:
# # # #         pr = m["pr"]; name = m["name"]
# # # #         plt.plot(pr["recall"], pr["precision"], label=f"{name} (AP={pr['ap']:.3f})")
# # # #     if op_points:
# # # #         for name, pts in op_points.items():
# # # #             rec_pt, prec_pt = pts["pr"]
# # # #             plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{name} op")
# # # #     plt.xlabel("Recall"); plt.ylabel("Precision")
# # # #     plt.title(f"PR — all models [{tag}]")
# # # #     plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
# # # #     plt.savefig(pr_dir / f"pr_all_{tag}.png", dpi=150); plt.close()

# # # # # --------------- discovery helpers ---------------
# # # # def discover_frames(frame_root: Path, tag: str, models: list[str]):
# # # #     """Return dict model -> path to frame_scores_<Model>_<tag>.csv if present."""
# # # #     found = {}
# # # #     for m in models:
# # # #         f = frame_root / m / f"frame_scores_{m}_{tag}.csv"
# # # #         if f.exists():
# # # #             found[m] = f
# # # #     return found

# # # # # --------------- main ---------------
# # # # def main():
# # # #     ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR")
# # # #     ap.add_argument("--clips", nargs="+", required=True, help="Clip-level result CSVs (per model).")
# # # #     ap.add_argument("--frame_scores", nargs="*", default=[], help="Optional explicit frame-score files.")
# # # #     ap.add_argument("--frame_root", type=str, default=None, help="Root to auto-find frames, e.g., outputs/frames/<TAG>")
# # # #     ap.add_argument("--tag", type=str, required=True, help="Dataset/eval tag, e.g., light/heavy")
# # # #     ap.add_argument("--runtime_summary_dir", type=str, required=True, help="Folder with runtime_summary_<TAG>__*.csv")
# # # #     ap.add_argument("--make_confbar", action="store_true", help="Save a confusion bar from clip CSVs")
# # # #     args = ap.parse_args()

# # # #     tag = args.tag
# # # #     stamp = now_tag()
# # # #     eval_run_dir = Path("outputs/eval") / f"{tag}__{stamp}"
# # # #     eval_run_dir.mkdir(parents=True, exist_ok=True)
# # # #     (eval_run_dir / "inputs").mkdir(parents=True, exist_ok=True)

# # # #     # ---- (1) Clip summary + op-point extraction ----
# # # #     op_points = {}
# # # #     summary_rows = []
# # # #     models = []

# # # #     for clip_csv in args.clips:
# # # #         clip_csv = Path(clip_csv)
# # # #         # copy inputs for traceability
# # # #         try:
# # # #             shutil.copy2(clip_csv, eval_run_dir / "inputs" / clip_csv.name)
# # # #         except Exception:
# # # #             pass

# # # #         y, p, _ = load_clip_results(clip_csv)
# # # #         tp, fp, fn, tn = confusion(y, p)
# # # #         prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

# # # #         model_name = infer_model_from_clip_name(clip_csv)
# # # #         models.append(model_name)

# # # #         summary_rows.append({
# # # #             "model": model_name,
# # # #             "file": clip_csv.name,
# # # #             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
# # # #             "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
# # # #             "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
# # # #         })

# # # #         TPR = rec
# # # #         FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
# # # #         op_points[model_name] = {"roc": (FPR, TPR), "pr": (rec, prec)}

# # # #     # merge runtime summaries
# # # #     rs_dir = Path(args.runtime_summary_dir)
# # # #     rt_cols = {"sec_per_hour":"","sec_per_hour_mean":"","sec_per_hour_std":"","repeats":""}
# # # #     if rs_dir.exists():
# # # #         # most recent summary per model (by reading all summaries in folder)
# # # #         per_model = {}
# # # #         for f in sorted(rs_dir.glob("runtime_summary_*__*.csv")):
# # # #             with open(f, "r", encoding="utf-8") as h:
# # # #                 rd = csv.DictReader(h)
# # # #                 for r in rd:
# # # #                     per_model[r["model"]] = {
# # # #                         "sec_per_hour": r.get("sec_per_hour",""),
# # # #                         "sec_per_hour_mean": r.get("sec_per_hour_mean",""),
# # # #                         "sec_per_hour_std": r.get("sec_per_hour_std",""),
# # # #                         "repeats": r.get("repeats",""),
# # # #                     }
# # # #         for row in summary_rows:
# # # #             row.update(per_model.get(row["model"], rt_cols.copy()))
# # # #     else:
# # # #         for row in summary_rows:
# # # #             row.update(rt_cols.copy())

# # # #     out_csv = eval_run_dir / f"summary_metrics_{tag}.csv"
# # # #     with open(out_csv,"w",newline="",encoding="utf-8") as f:
# # # #         w=csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
# # # #         w.writeheader(); w.writerows(summary_rows)
# # # #     print(f"[clip] wrote {out_csv}")

# # # #     if args.make_confbar:
# # # #         conf_png = eval_run_dir / f"confusion_bar_{tag}.png"
# # # #         plot_confusion_bar(summary_rows, conf_png, f"Confusion counts — {tag}")
# # # #         print(f"[clip] wrote {conf_png}")

# # # #     # ---- (2) Frame-level curves ----
# # # #     # collect frame files: explicit list or auto-discover by tag
# # # #     frame_files = [Path(p) for p in args.frame_scores] if args.frame_scores else []
# # # #     if (not frame_files) and args.frame_root:
# # # #         auto = discover_frames(Path(args.frame_root), tag, models)
# # # #         frame_files = list(auto.values())

# # # #     if not frame_files:
# # # #         print("[info] no frame scores passed; ROC/PR skipped.")
# # # #         return

# # # #     # load, compute curves
# # # #     def load_frame_csv(p: Path):
# # # #         labels, scores = [], []
# # # #         with open(p, "r", encoding="utf-8") as f:
# # # #             rd = csv.DictReader(f)
# # # #             for r in rd:
# # # #                 labels.append(int(r.get("label_frame", "0")))
# # # #                 s = r.get("prob","")
# # # #                 s = float(s) if s not in ("", None) else float(r.get("score","0.0"))
# # # #                 scores.append(s)
# # # #         return np.asarray(labels, np.int32), np.asarray(scores, np.float64)

# # # #     roc_dir = eval_run_dir / "roc"
# # # #     pr_dir  = eval_run_dir / "pr"

# # # #     combined = []
# # # #     for p in frame_files:
# # # #         # infer model from filename
# # # #         name = p.stem.replace("frame_scores_","")
# # # #         # e.g. "Energy_light" -> model is part before "_<tag>"
# # # #         if name.endswith(f"_{tag}"):
# # # #             name = name[:-(len(tag)+1)]
# # # #         model = name
# # # #         y, s = load_frame_csv(p)
# # # #         res = compute_curves(y, s)
# # # #         roc={"fpr":res["fpr"],"tpr":res["tpr"],"auc":res["auc"]}
# # # #         pr ={"recall":res["recall"],"precision":res["precision"],"ap":res["ap"]}
# # # #         save_curve_npz_png(roc_dir, pr_dir, model, roc, pr, title_suffix=f"  [{tag}]", op_points=op_points)
# # # #         combined.append({"name":model,"roc":roc,"pr":pr})

# # # #     plot_compare(combined, roc_dir, pr_dir, tag, op_points=op_points)
# # # #     print(f"[curves] wrote {roc_dir / f'roc_all_{tag}.png'} and {pr_dir / f'pr_all_{tag}.png'}")

# # # #     # summary table image
# # # #     # (simple one-row-per-model table)
# # # #     import matplotlib.pyplot as plt
# # # #     import pandas as pd
# # # #     df = pd.read_csv(out_csv)
# # # #     fig, ax = plt.subplots(figsize=(16,4))
# # # #     ax.axis('off')
# # # #     tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
# # # #     tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.4)
# # # #     plt.title(f"VAD Summary — {tag}")
# # # #     out_png = eval_run_dir / f"summary_table_{tag}.png"
# # # #     plt.savefig(out_png, dpi=160, bbox_inches='tight')
# # # #     plt.close()
# # # #     print(f"[table] wrote {out_png}")

# # # # if __name__ == "__main__":
# # # #     main()

# # # #!/usr/bin/env python3
# # # # scripts/evaluate_vad.py
# # # # - Reads clip CSVs, computes confusion + summary table (+runtime merge)
# # # # - Loads frame score CSVs from outputs/frames/<TAG>/<Model>/*.csv (or single file)
# # # # - ROC/PR for each model + combined plots
# # # # - Marks the clip OP (star) and the FP-minimizing OP at recall>=target (diamond)

# # # # import argparse, csv, datetime
# # # # from pathlib import Path
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # plt.rcParams["figure.autolayout"] = True

# # # # def now_tag(): return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # # # def confusion(y, yhat):
# # # #     tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
# # # #     fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
# # # #     return tp,fp,fn,tn

# # # # def metrics(tp,fp,fn,tn):
# # # #     prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
# # # #     rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
# # # #     f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
# # # #     acc  = (tp+tn)/max(tp+fp+fn+tn,1)
# # # #     return prec,rec,f1,acc

# # # # def infer_model_from_path(p: Path):
# # # #     s = p.stem.lower()
# # # #     if "webrtc_l2" in s: return "webrtc_l2"
# # # #     if "webrtc_l3" in s: return "webrtc_l3"
# # # #     for k in ("energy","zcr","combo"):
# # # #         if k in s: return k
# # # #     return s

# # # # def compute_curves(labels, scores):
# # # #     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
# # # #     if len(labels)==0: 
# # # #         return dict(fpr=np.array([0,1.0]), tpr=np.array([0,1.0]), auc=np.nan,
# # # #                     recall=np.array([0,1.0]), precision=np.array([1.0,0]), ap=np.nan)
# # # #     idx = np.argsort(scores)[::-1]; y = labels[idx]
# # # #     tp = np.cumsum(y); fp = np.cumsum(1-y); P = tp[-1]; N = fp[-1]
# # # #     sc = scores[idx]; ch = np.r_[True, sc[1:]!=sc[:-1]]
# # # #     tp = tp[ch]; fp=fp[ch]
# # # #     tpr = tp/ max(P,1); fpr = fp/ max(N,1)
# # # #     auc = float(np.trapz(tpr,fpr))
# # # #     recall = tpr
# # # #     precision = tp/np.maximum(tp+fp,1); ap = float(np.sum((recall[1:]-recall[:-1])*precision[1:]))
# # # #     return dict(fpr=fpr,tpr=tpr,auc=auc,recall=recall,precision=precision,ap=ap)

# # # # def pick_low_fp_point(roc, target_recall=0.80):
# # # #     m = roc["tpr"] >= target_recall
# # # #     if not np.any(m):  # fallback to max tpr
# # # #         j = int(np.argmax(roc["tpr"]))
# # # #     else:
# # # #         cand = np.where(m)[0]
# # # #         j = cand[int(np.argmin(roc["fpr"][cand]))]
# # # #     return float(roc["fpr"][j]), float(roc["tpr"][j])

# # # # def plot_conf_bar(rows, out_png, title):
# # # #     names = [r["model"] for r in rows]
# # # #     tp = np.array([r["tp"] for r in rows]); fp=np.array([r["fp"] for r in rows])
# # # #     fn = np.array([r["fn"] for r in rows]); tn=np.array([r["tn"] for r in rows])
# # # #     x = np.arange(len(names))
# # # #     plt.figure(figsize=(8,4))
# # # #     plt.bar(x, tp, label="TP")
# # # #     plt.bar(x, fp, bottom=tp, label="FP")
# # # #     plt.bar(x, fn, bottom=tp+fp, label="FN")
# # # #     plt.bar(x, tn, bottom=tp+fp+fn, label="TN")
# # # #     plt.xticks(x, [n.replace("_"," ").title() for n in names])
# # # #     plt.ylabel("Count"); plt.title(title); plt.legend()
# # # #     plt.savefig(out_png, dpi=150); plt.close()

# # # # def main():
# # # #     ap = argparse.ArgumentParser(description="Evaluate VAD (clips + ROC/PR)")
# # # #     ap.add_argument("--clips", nargs="+", required=True, help="clip_results_*.csv (all models)")
# # # #     ap.add_argument("--frame_scores", nargs="*", default=[], help="optional explicit frame CSVs")
# # # #     ap.add_argument("--tag", required=True, help="e.g. light / heavy")
# # # #     ap.add_argument("--runtime_summary_dir", required=True, help="outputs/runtime/<TAG>")
# # # #     ap.add_argument("--recall_target", type=float, default=0.80)
# # # #     ap.add_argument("--make_confbar", action="store_true")
# # # #     args = ap.parse_args()

# # # #     stamp = now_tag()
# # # #     out_root = Path("outputs/eval")/f"{args.tag}__{stamp}"
# # # #     out_root.mkdir(parents=True, exist_ok=True)

# # # #     # (1) Clip summary + OP from clips
# # # #     rows=[]; op_points={}
# # # #     for cfile in args.clips:
# # # #         cpath = Path(cfile)
# # # #         model = infer_model_from_path(cpath)
# # # #         y=[]; yhat=[]
# # # #         with open(cpath,"r",encoding="utf-8") as f:
# # # #             rd=csv.DictReader(f)
# # # #             for r in rd:
# # # #                 y.append(int(r["label"])); yhat.append(int(r["pred"]))
# # # #         y=np.array(y,np.int32); yhat=np.array(yhat,np.int32)
# # # #         tp,fp,fn,tn = confusion(y,yhat); pr,rc,f1,acc = metrics(tp,fp,fn,tn)
# # # #         rows.append(dict(model=model,file=cpath.name,tp=tp,fp=fp,fn=fn,tn=tn,
# # # #                          precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),acc=round(acc,4)))
# # # #         # clip OP marker
# # # #         FPR = fp/max(fp+tn,1); TPR = rc
# # # #         op_points[model] = {"roc":(FPR,TPR), "pr":(rc,pr)}

# # # #     # Merge runtime (if present)
# # # #     rt = Path(args.runtime_summary_dir)/f"runtime_summary_{args.tag}.csv"
# # # #     rt_map={}
# # # #     if rt.exists():
# # # #         with open(rt,"r",encoding="utf-8") as f:
# # # #             rd=csv.DictReader(f)
# # # #             for r in rd:
# # # #                 rt_map[r["model"].lower()] = r

# # # #     # write summary table (csv + png)
# # # #     out_csv = out_root/f"summary_metrics_{args.tag}.csv"
# # # #     headers = ["model","file","tp","fp","fn","tn","precision","recall","f1","acc",
# # # #                "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
# # # #     with open(out_csv,"w",newline="",encoding="utf-8") as f:
# # # #         w=csv.DictWriter(f, fieldnames=headers); w.writeheader()
# # # #         for r in rows:
# # # #             rtrow = rt_map.get(r["model"].lower(), {})
# # # #             r.update({
# # # #                 "sec_per_hour": rtrow.get("sec_per_hour",""),
# # # #                 "sec_per_hour_mean": rtrow.get("sec_per_hour_mean",""),
# # # #                 "sec_per_hour_std": rtrow.get("sec_per_hour_std",""),
# # # #                 "repeats": rtrow.get("repeats",""),
# # # #             })
# # # #             w.writerow(r)

# # # #     # pretty table PNG
# # # #     try:
# # # #         import pandas as pd
# # # #         df = pd.read_csv(out_csv)
# # # #         fig, ax = plt.subplots(figsize=(12,3.1))
# # # #         ax.axis("off")
# # # #         ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        
# # # #         ax.set_title(f"VAD Summary — {args.tag}")
# # # #         plt.savefig(out_root/f"summary_table_{args.tag}.png", dpi=150); plt.close()
# # # #     except Exception:
# # # #         pass

# # # #     if args.make_confbar:
# # # #         plot_conf_bar(rows, out_root/f"confusion_bar_{args.tag}.png", f"Confusion counts — {args.tag}")

# # # #     # (2) Frame scores → ROC/PR
# # # #     # If user did not pass explicit frame files, discover by TAG:
# # # #     frame_files = list(map(Path, args.frame_scores))
# # # #     if not frame_files:
# # # #         root = Path("outputs/frames")/args.tag
# # # #         if root.exists():
# # # #             frame_files = sorted(root.rglob("frame_scores_*.csv"))

# # # #     if not frame_files:
# # # #         print("[info] no frame scores found; ROC/PR skipped.")
# # # #         return

# # # #     models=[]
# # # #     curves=[]
# # # #     # compact loaders: expect columns: model, source, frame_idx, score, prob
# # # #     for fp in frame_files:
# # # #         m = fp.name.split("_")[2].lower() if "frame_scores_" in fp.name else infer_model_from_path(fp)
# # # #         sc=[]; lab=[]
# # # #         # For curves we treat prob as score, and we fake frame-labels from pred_post (since we don’t have GT per frame)
# # # #         with open(fp,"r",encoding="utf-8") as f:
# # # #             rd=csv.DictReader(f)
# # # #             for r in rd:
# # # #                 sc.append(float(r.get("prob", r.get("score","0"))))
# # # #                 lab.append(int(r.get("pred_post","0")))
# # # #         sc=np.array(sc,np.float64); lab=np.array(lab,np.int32)
# # # #         rocpr = compute_curves(lab, sc)
# # # #         models.append(m); curves.append(rocpr)

# # # #     # combined plots + markers
# # # #     def _plot_all(kind):
# # # #         plt.figure(figsize=(9,7))
# # # #         for m,rc in zip(models,curves):
# # # #             if kind=="roc":
# # # #                 plt.plot(rc["fpr"], rc["tpr"], label=f"{m} (AUC={rc['auc']:.3f})")
# # # #             else:
# # # #                 plt.plot(rc["recall"], rc["precision"], label=f"{m} (AP={rc['ap']:.3f})")
# # # #         # clip OP stars
# # # #         for m in models:
# # # #             if m in op_points:
# # # #                 if kind=="roc":
# # # #                     x,y = op_points[m]["roc"]
# # # #                 else:
# # # #                     x,y = op_points[m]["pr"]
# # # #                 plt.scatter([x],[y], marker="*", s=120, label=f"{m} op")
# # # #         # FP-min OP diamonds (recall >= target)
# # # #         for m,rc in zip(models,curves):
# # # #             if kind=="roc":
# # # #                 x,y = pick_low_fp_point(rc, target_recall=args.recall_target)
# # # #                 plt.scatter([x],[y], marker="D", s=70, label=f"{m} low-FP")
# # # #         if kind=="roc":
# # # #             plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC — all models [{args.tag}]")
# # # #         else:
# # # #             plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — all models [{args.tag}]")
# # # #         plt.grid(alpha=0.3); plt.legend(ncol=1); plt.tight_layout()
# # # #         plt.savefig(out_root/(f"{kind}_all_{args.tag}.png"), dpi=150); plt.close()

# # # #     _plot_all("roc")
# # # #     _plot_all("pr")
# # # #     print(f"[plots] ROC/PR saved in {out_root}")

# # # # if __name__ == "__main__":
# # # #     main()


# # # #!/usr/bin/env python3
# # # import argparse, csv, datetime
# # # from pathlib import Path
# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # plt.rcParams["figure.autolayout"] = True

# # # def now_tag(): 
# # #     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # # # ---------- basic metrics ----------
# # # def confusion(y, yhat):
# # #     tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
# # #     fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
# # #     return tp,fp,fn,tn

# # # def metrics(tp,fp,fn,tn):
# # #     prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
# # #     rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
# # #     f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
# # #     acc  = (tp+tn)/max(tp+fp+fn+tn,1)
# # #     return prec,rec,f1,acc

# # # def infer_model_from_path(p: Path):
# # #     s = p.stem.lower()
# # #     if "webrtc_l2" in s: return "webrtc_l2"
# # #     if "webrtc_l3" in s: return "webrtc_l3"
# # #     for k in ("energy","zcr","combo"):
# # #         if k in s: return k
# # #     return s

# # # # ---------- curves ----------
# # # def compute_curves(labels, scores):
# # #     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
# # #     if len(labels)==0: 
# # #         return dict(fpr=np.array([0,1.0]), tpr=np.array([0,1.0]), auc=np.nan,
# # #                     recall=np.array([0,1.0]), precision=np.array([1.0,0]), ap=np.nan)
# # #     order = np.argsort(scores)[::-1]
# # #     y = labels[order]
# # #     tp = np.cumsum(y)
# # #     fp = np.cumsum(1-y)
# # #     P = float(tp[-1]); N = float(fp[-1])
# # #     sc = scores[order]
# # #     ch = np.r_[True, sc[1:]!=sc[:-1]]
# # #     tp = tp[ch]; fp=fp[ch]

# # #     tpr = tp/ max(P,1.0); fpr = fp/ max(N,1.0)
# # #     auc = float(np.trapezoid(tpr,fpr))
# # #     recall = tpr
# # #     precision = tp/np.maximum(tp+fp,1)
# # #     ap = float(np.sum((recall[1:]-recall[:-1])*precision[1:])) if len(recall)>1 else float('nan')
# # #     return dict(fpr=fpr,tpr=tpr,auc=auc,recall=recall,precision=precision,ap=ap)

# # # def pick_low_fp_point(roc, target_recall=0.80):
# # #     m = roc["tpr"] >= target_recall
# # #     if not np.any(m):
# # #         j = int(np.argmax(roc["tpr"]))
# # #     else:
# # #         cand = np.where(m)[0]
# # #         j = cand[int(np.argmin(roc["fpr"][cand]))]
# # #     return float(roc["fpr"][j]), float(roc["tpr"][j])

# # # # ---------- pretty summary table ----------
# # # def plot_summary_table(df, tag, out_png):
# # #     fig, ax = plt.subplots(figsize=(18, 4.5), dpi=200)
# # #     ax.axis('off')
# # #     tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
# # #     tbl.auto_set_font_size(False)
# # #     tbl.set_fontsize(10)
# # #     tbl.scale(1.2, 1.2)
# # #     ax.set_title(f"VAD Summary — {tag}", pad=16, fontsize=16)
# # #     fig.savefig(out_png, bbox_inches='tight')
# # #     plt.close(fig)

# # # # ---------- confusion bar ----------
# # # def plot_conf_bar(rows, out_png, title):
# # #     names = [r["model"] for r in rows]
# # #     tp = np.array([r["tp"] for r in rows]); fp=np.array([r["fp"] for r in rows])
# # #     fn = np.array([r["fn"] for r in rows]); tn=np.array([r["tn"] for r in rows])
# # #     x = np.arange(len(names))
# # #     plt.figure(figsize=(9.5,5))
# # #     b1 = plt.bar(x, tp, label="TP")
# # #     b2 = plt.bar(x, fp, bottom=tp, label="FP")
# # #     b3 = plt.bar(x, fn, bottom=tp+fp, label="FN")
# # #     b4 = plt.bar(x, tn, bottom=tp+fp+fn, label="TN")
# # #     plt.xticks(x, [n.replace("_"," ").title() for n in names])
# # #     plt.ylabel("Count"); plt.title(title); plt.legend()
# # #     plt.savefig(out_png, dpi=180); plt.close()

# # # # ---------- frame-file discovery ----------
# # # def find_frame_files(tag, models_expected):
# # #     """
# # #     Search outputs/frames/<tag>/<Model>/frame_scores_<Model>_<tag>.csv
# # #     Returns dict: model -> list[Path]
# # #     """
# # #     base = Path("outputs/frames") / tag
# # #     found = {m: [] for m in models_expected}
# # #     if not base.exists():
# # #         return found
# # #     prefix = {
# # #         'energy':     'frame_scores_Energy_',
# # #         'zcr':        'frame_scores_ZCR_',
# # #         'combo':      'frame_scores_Combo_',
# # #         'webrtc_l2':  'frame_scores_webrtc_l2_',
# # #         'webrtc_l3':  'frame_scores_webrtc_l3_',
# # #     }
# # #     for m in models_expected:
# # #         sub = base / m
# # #         if sub.is_dir():
# # #             found[m] = sorted(sub.glob(f"{prefix[m]}{tag}.csv"))
# # #         else:
# # #             found[m] = []
# # #     return found

# # # def _add_op_star(ax, xy, label, color=None, zorder=4):
# # #     if not xy: return
# # #     x, y = xy
# # #     ax.scatter([x], [y], marker='*', s=180, linewidths=0.8,
# # #                edgecolors='k', facecolors=color if color else 'gold',
# # #                zorder=zorder, label=label)

# # # def main():
# # #     ap = argparse.ArgumentParser(description="Evaluate VAD (clips + ROC/PR)")
# # #     ap.add_argument("--clips", nargs="+", required=True, help="clip_results_*.csv (all models)")
# # #     ap.add_argument("--frame_scores", nargs="*", default=[], help="optional explicit frame CSVs")
# # #     ap.add_argument("--tag", required=True, help="e.g. light / heavy")
# # #     ap.add_argument("--runtime_summary_dir", required=True, help="outputs/runtime/<TAG>")
# # #     ap.add_argument("--recall_target", type=float, default=0.80)
# # #     ap.add_argument("--make_confbar", action="store_true")
# # #     args = ap.parse_args()

# # #     stamp = now_tag()
# # #     out_root = Path("outputs/eval")/f"{args.tag}__{stamp}"
# # #     out_root.mkdir(parents=True, exist_ok=True)

# # #     # (1) Clip summary + OP from clips
# # #     rows=[]; op_points={}
# # #     models_in_order=[]
# # #     for cfile in args.clips:
# # #         cpath = Path(cfile)
# # #         model = infer_model_from_path(cpath)
# # #         models_in_order.append(model)
# # #         y=[]; yhat=[]
# # #         with open(cpath,"r",encoding="utf-8") as f:
# # #             rd=csv.DictReader(f)
# # #             for r in rd:
# # #                 y.append(int(r["label"])); yhat.append(int(r["pred"]))
# # #         y=np.array(y,np.int32); yhat=np.array(yhat,np.int32)
# # #         tp,fp,fn,tn = confusion(y,yhat); pr,rc,f1,acc = metrics(tp,fp,fn,tn)
# # #         rows.append(dict(model=model,file=cpath.name,tp=tp,fp=fp,fn=fn,tn=tn,
# # #                          precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),acc=round(acc,4)))
# # #         # clip OP star coords
# # #         FPR = fp/max(fp+tn,1); TPR = rc
# # #         op_points[model] = {"roc":(FPR,TPR), "pr":(rc,pr)}

# # #     # Merge runtime (if present)
# # #     rt = Path(args.runtime_summary_dir)/f"runtime_summary_{args.tag}.csv"
# # #     rt_map={}
# # #     if rt.exists():
# # #         with open(rt,"r",encoding="utf-8") as f:
# # #             rd=csv.DictReader(f)
# # #             for r in rd:
# # #                 rt_map[r["model"].lower()] = r

# # #     # write summary table (csv)
# # #     out_csv = out_root/f"summary_metrics_{args.tag}.csv"
# # #     headers = ["model","file","tp","fp","fn","tn","precision","recall","f1","acc",
# # #                "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
# # #     with open(out_csv,"w",newline="",encoding="utf-8") as f:
# # #         w=csv.DictWriter(f, fieldnames=headers); w.writeheader()
# # #         for r in rows:
# # #             rtrow = rt_map.get(r["model"].lower(), {})
# # #             r_out = dict(r)
# # #             r_out.update({
# # #                 "sec_per_hour": rtrow.get("sec_per_hour",""),
# # #                 "sec_per_hour_mean": rtrow.get("sec_per_hour_mean",""),
# # #                 "sec_per_hour_std": rtrow.get("sec_per_hour_std",""),
# # #                 "repeats": rtrow.get("repeats",""),
# # #             })
# # #             w.writerow(r_out)

# # #     # table PNG (readable)
# # #     try:
# # #         import pandas as pd
# # #         df = pd.read_csv(out_csv)
# # #         plot_summary_table(df, args.tag, out_root/f"summary_table_{args.tag}.png")
# # #     except Exception:
# # #         pass

# # #     if args.make_confbar:
# # #         plot_conf_bar(rows, out_root/f"confusion_bar_{args.tag}.png", f"Confusion counts — {args.tag}")

# # #     # (2) Frame scores → ROC/PR
# # #     # Discover frame files if not given explicitly
# # #     frame_files = list(map(Path, args.frame_scores))
# # #     if not frame_files:
# # #         discovered = find_frame_files(args.tag, models_in_order)
# # #         for m in models_in_order:
# # #             frame_files.extend(discovered.get(m, []))

# # #     if not frame_files:
# # #         print("[info] no frame scores found; ROC/PR skipped.")
# # #         return

# # #     # build curves
# # #     models=[]; curves=[]
# # #     for fp in frame_files:
# # #         # infer model from the CSV name
# # #         name = fp.name.lower()
# # #         if "webrtc_l2" in name: m = "webrtc_l2"
# # #         elif "webrtc_l3" in name: m = "webrtc_l3"
# # #         elif "_energy_" in name: m = "energy"
# # #         elif "_zcr_" in name: m = "zcr"
# # #         elif "_combo_" in name: m = "combo"
# # #         else: m = infer_model_from_path(fp)

# # #         sc=[]; lab=[]
# # #         # Expect columns: model, source, frame_idx, score, prob, label_frame OR pred_post
# # #         with open(fp,"r",encoding="utf-8") as f:
# # #             rd=csv.DictReader(f)
# # #             for r in rd:
# # #                 p = r.get("prob", r.get("score","0"))
# # #                 sc.append(float(p))
# # #                 lf = r.get("label_frame", r.get("pred_post","0"))  # fallback
# # #                 lab.append(int(lf))
# # #         sc=np.array(sc,np.float64); lab=np.array(lab,np.int32)
# # #         rocpr = compute_curves(lab, sc)
# # #         models.append(m); curves.append(rocpr)

# # #     # combined plots + markers
# # #     def _plot_all(kind):
# # #         plt.figure(figsize=(9.5,7.5))
# # #         # consistent color order
# # #         for m,rc in zip(models,curves):
# # #             if kind=="roc":
# # #                 plt.plot(rc["fpr"], rc["tpr"], label=f"{m} (AUC={rc['auc']:.3f})")
# # #             else:
# # #                 plt.plot(rc["recall"], rc["precision"], label=f"{m} (AP={rc['ap']:.3f})")

# # #         # OP stars from clip metrics (always shown)
# # #         for m in models_in_order:
# # #             pts = op_points.get(m)
# # #             if not pts: continue
# # #             xy = pts["roc"] if kind=="roc" else pts["pr"]
# # #             _add_op_star(plt.gca(), xy, f"{m} op")

# # #         # low-FP diamonds (min FPR at recall>=target) on ROC
# # #         if kind=="roc":
# # #             for m,rc in zip(models,curves):
# # #                 x,y = pick_low_fp_point(rc, target_recall=args.recall_target)
# # #                 plt.scatter([x],[y], marker="D", s=70, label=f"{m} low-FP")

# # #         if kind=="roc":
# # #             plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
# # #             plt.title(f"ROC — all models [{args.tag}]")
# # #         else:
# # #             plt.xlabel("Recall"); plt.ylabel("Precision")
# # #             plt.title(f"PR — all models [{args.tag}]")

# # #         leg = plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
# # #         for lh in leg.legendHandles:
# # #             try: lh.set_alpha(1.0)
# # #             except: pass
# # #         plt.grid(alpha=0.3); plt.tight_layout()
# # #         plt.savefig(out_root/(f"{kind}_all_{args.tag}.png"), dpi=180); plt.close()

# # #     _plot_all("roc")
# # #     _plot_all("pr")
# # #     print(f"[plots] ROC/PR saved in {out_root}")

# # # if __name__ == "__main__":
# # #     main()


# # #!/usr/bin/env python3
# # import argparse, csv, datetime
# # from pathlib import Path
# # import numpy as np
# # import matplotlib.pyplot as plt

# # plt.rcParams["figure.autolayout"] = True

# # def now_tag():
# #     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # # ---------- basic metrics ----------
# # def confusion(y, yhat):
# #     tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
# #     fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
# #     return tp,fp,fn,tn

# # def metrics(tp,fp,fn,tn):
# #     prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
# #     rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
# #     f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
# #     acc  = (tp+tn)/max(tp+fp+fn+tn,1)
# #     return prec,rec,f1,acc

# # def infer_model_from_path(p: Path):
# #     s = p.stem.lower()
# #     if "webrtc_l2" in s: return "webrtc_l2"
# #     if "webrtc_l3" in s: return "webrtc_l3"
# #     for k in ("energy","zcr","combo"):
# #         if k in s: return k
# #     return s

# # # ---------- curves ----------
# # def compute_curves(labels, scores):
# #     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
# #     if len(labels)==0:
# #         return dict(fpr=np.array([0,1.0]), tpr=np.array([0,1.0]), auc=np.nan,
# #                     recall=np.array([0,1.0]), precision=np.array([1.0,0]), ap=np.nan)
# #     order = np.argsort(scores)[::-1]
# #     y = labels[order]
# #     tp = np.cumsum(y)
# #     fp = np.cumsum(1-y)
# #     P = float(tp[-1]); N = float(fp[-1])
# #     sc = scores[order]
# #     ch = np.r_[True, sc[1:]!=sc[:-1]]
# #     tp = tp[ch]; fp = fp[ch]

# #     tpr = tp/max(P,1.0); fpr = fp/max(N,1.0)
# #     # use np.trapezoid(y, x)
# #     auc = float(np.trapezoid(tpr, fpr))
# #     recall = tpr
# #     precision = tp/np.maximum(tp+fp,1)
# #     ap = float(np.sum((recall[1:]-recall[:-1]) * precision[1:])) if len(recall)>1 else float('nan')
# #     return dict(fpr=fpr,tpr=tpr,auc=auc,recall=recall,precision=precision,ap=ap)

# # def pick_low_fp_point(roc, target_recall=0.80):
# #     m = roc["tpr"] >= target_recall
# #     if not np.any(m):
# #         j = int(np.argmax(roc["tpr"]))
# #     else:
# #         cand = np.where(m)[0]
# #         j = cand[int(np.argmin(roc["fpr"][cand]))]
# #     return float(roc["fpr"][j]), float(roc["tpr"][j])

# # # ---------- pretty summary table ----------
# # def plot_summary_table(df, tag, out_png):
# #     fig, ax = plt.subplots(figsize=(16, 4.8), dpi=160)
# #     ax.axis('off')
# #     tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
# #     tbl.auto_set_font_size(False)
# #     tbl.set_fontsize(10)
# #     tbl.scale(1.0, 1.3)
# #     ax.set_title(f"VAD Summary — {tag}", pad=12, fontsize=14)
# #     fig.savefig(out_png, bbox_inches='tight')
# #     plt.close(fig)

# # # ---------- confusion bar ----------
# # def plot_conf_bar(rows, out_png, title):
# #     names = [r["model"] for r in rows]
# #     tp = np.array([r["tp"] for r in rows]); fp=np.array([r["fp"] for r in rows])
# #     fn = np.array([r["fn"] for r in rows]); tn=np.array([r["tn"] for r in rows])
# #     x = np.arange(len(names))
# #     plt.figure(figsize=(10,5))
# #     plt.bar(x, tp, label="TP")
# #     plt.bar(x, fp, bottom=tp, label="FP")
# #     plt.bar(x, fn, bottom=tp+fp, label="FN")
# #     plt.bar(x, tn, bottom=tp+fp+fn, label="TN")
# #     plt.xticks(x, [n.replace("_"," ").title() for n in names], rotation=0)
# #     plt.ylabel("Count"); plt.title(title); plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(out_png, dpi=180); plt.close()

# # # ---------- frame-file discovery ----------
# # def find_frame_files(tag, models_expected):
# #     """
# #     Search outputs/frames/<tag>/<Model>/frame_scores_<Model>_<tag>.csv
# #     Returns dict: model -> list[Path]
# #     """
# #     base = Path("outputs/frames") / tag
# #     found = {m: [] for m in models_expected}
# #     if not base.exists():
# #         return found
# #     prefix = {
# #         'energy':     'frame_scores_Energy_',
# #         'zcr':        'frame_scores_ZCR_',
# #         'combo':      'frame_scores_Combo_',
# #         'webrtc_l2':  'frame_scores_webrtc_l2_',
# #         'webrtc_l3':  'frame_scores_webrtc_l3_',
# #     }
# #     for m in models_expected:
# #         sub = base / m
# #         if sub.is_dir():
# #             found[m] = sorted(sub.glob(f"{prefix[m]}{tag}.csv"))
# #         else:
# #             found[m] = []
# #     return found

# # def _add_op_star(ax, xy, label, color=None, zorder=4):
# #     if not xy: return
# #     x, y = xy
# #     ax.scatter([x], [y], marker='*', s=160, linewidths=0.8,
# #                edgecolors='k', facecolors=color if color else 'gold',
# #                zorder=zorder, label=label)

# # def main():
# #     ap = argparse.ArgumentParser(description="Evaluate VAD (clips + ROC/PR)")
# #     ap.add_argument("--clips", nargs="+", required=True, help="clip_results_*.csv (all models)")
# #     ap.add_argument("--frame_scores", nargs="*", default=[], help="optional explicit frame CSVs")
# #     ap.add_argument("--tag", required=True, help="e.g. light / heavy")
# #     ap.add_argument("--runtime_summary_dir", required=True, help="outputs/runtime/<TAG>")
# #     ap.add_argument("--recall_target", type=float, default=0.80)
# #     ap.add_argument("--make_confbar", action="store_true")
# #     args = ap.parse_args()

# #     stamp = now_tag()
# #     out_root = Path("outputs/eval")/f"{args.tag}__{stamp}"
# #     out_root.mkdir(parents=True, exist_ok=True)

# #     # (1) Clip summary + OP from clips
# #     rows=[]; op_points={}
# #     models_in_order=[]
# #     for cfile in args.clips:
# #         cpath = Path(cfile)
# #         model = infer_model_from_path(cpath)
# #         models_in_order.append(model)
# #         y=[]; yhat=[]
# #         with open(cpath,"r",encoding="utf-8") as f:
# #             rd=csv.DictReader(f)
# #             for r in rd:
# #                 y.append(int(r["label"])); yhat.append(int(r["pred"]))
# #         y=np.array(y,np.int32); yhat=np.array(yhat,np.int32)
# #         tp,fp,fn,tn = confusion(y,yhat); pr,rc,f1,acc = metrics(tp,fp,fn,tn)
# #         rows.append(dict(model=model,file=cpath.name,tp=tp,fp=fp,fn=fn,tn=tn,
# #                          precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),acc=round(acc,4)))
# #         # clip OP star coords
# #         FPR = fp/max(fp+tn,1); TPR = rc
# #         op_points[model] = {"roc":(FPR,TPR), "pr":(rc,pr)}

# #     # Merge runtime (if present)
# #     rt = Path(args.runtime_summary_dir)/f"runtime_summary_{args.tag}.csv"
# #     rt_map={}
# #     if rt.exists():
# #         with open(rt,"r",encoding="utf-8") as f:
# #             rd=csv.DictReader(f)
# #             for r in rd:
# #                 rt_map[r["model"].lower()] = r

# #     # write summary table (csv)
# #     out_csv = out_root/f"summary_metrics_{args.tag}.csv"
# #     headers = ["model","file","tp","fp","fn","tn","precision","recall","f1","acc",
# #                "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
# #     with open(out_csv,"w",newline="",encoding="utf-8") as f:
# #         w=csv.DictWriter(f, fieldnames=headers); w.writeheader()
# #         for r in rows:
# #             rtrow = rt_map.get(r["model"].lower(), {})
# #             r_out = dict(r)
# #             r_out.update({
# #                 "sec_per_hour": rtrow.get("sec_per_hour",""),
# #                 "sec_per_hour_mean": rtrow.get("sec_per_hour_mean",""),
# #                 "sec_per_hour_std": rtrow.get("sec_per_hour_std",""),
# #                 "repeats": rtrow.get("repeats",""),
# #             })
# #             w.writerow(r_out)

# #     # table PNG (readable)
# #     try:
# #         import pandas as pd
# #         df = pd.read_csv(out_csv)
# #         plot_summary_table(df, args.tag, out_root/f"summary_table_{args.tag}.png")
# #     except Exception:
# #         pass

# #     if args.make_confbar:
# #         plot_conf_bar(rows, out_root/f"confusion_bar_{args.tag}.png", f"Confusion counts — {args.tag}")

# #     # (2) Frame scores → ROC/PR
# #     # Discover frame files if not given explicitly
# #     frame_files = list(map(Path, args.frame_scores))
# #     if not frame_files:
# #         discovered = find_frame_files(args.tag, models_in_order)
# #         for m in models_in_order:
# #             frame_files.extend(discovered.get(m, []))

# #     if not frame_files:
# #         print("[info] no frame scores found; ROC/PR skipped.")
# #         return

# #     # build curves
# #     models=[]; curves=[]
# #     for fp in frame_files:
# #         # infer model from the CSV name
# #         name = fp.name.lower()
# #         if "webrtc_l2" in name: m = "webrtc_l2"
# #         elif "webrtc_l3" in name: m = "webrtc_l3"
# #         elif "_energy_" in name: m = "energy"
# #         elif "_zcr_" in name: m = "zcr"
# #         elif "_combo_" in name: m = "combo"
# #         else: m = infer_model_from_path(fp)

# #         sc=[]; lab=[]
# #         # Expect columns: model, source, frame_idx, score, prob, label_frame OR pred_post
# #         with open(fp,"r",encoding="utf-8") as f:
# #             rd=csv.DictReader(f)
# #             for r in rd:
# #                 p = r.get("prob", r.get("score","0"))
# #                 sc.append(float(p))
# #                 lf = r.get("label_frame", r.get("pred_post","0"))  # fallback
# #                 lab.append(int(lf))
# #         sc=np.array(sc,np.float64); lab=np.array(lab,np.int32)
# #         rocpr = compute_curves(lab, sc)
# #         models.append(m); curves.append(rocpr)

# #     # combined plots + markers
# #     def _plot_all(kind):
# #         plt.figure(figsize=(9.5,7.5))
# #         # model curves
# #         for m,rc in zip(models,curves):
# #             if kind=="roc":
# #                 plt.plot(rc["fpr"], rc["tpr"], label=f"{m} (AUC={rc['auc']:.3f})")
# #             else:
# #                 plt.plot(rc["recall"], rc["precision"], label=f"{m} (AP={rc['ap']:.3f})")

# #         ax = plt.gca()

# #         # OP stars from clip metrics (always shown)
# #         for m in models_in_order:
# #             pts = op_points.get(m)
# #             if not pts: continue
# #             xy = pts["roc"] if kind=="roc" else pts["pr"]
# #             _add_op_star(ax, xy, f"{m} op")

# #         # low-FP diamonds (min FPR at recall>=target) on ROC
# #         if kind=="roc":
# #             for m,rc in zip(models,curves):
# #                 x,y = pick_low_fp_point(rc, target_recall=args.recall_target)
# #                 plt.scatter([x],[y], marker="D", s=70, label=f"{m} low-FP")

# #         if kind=="roc":
# #             plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
# #             plt.title(f"ROC — all models [{args.tag}]")
# #         else:
# #             plt.xlabel("Recall"); plt.ylabel("Precision")
# #             plt.title(f"PR — all models [{args.tag}]")

# #         # de-dupe legend entries and be robust across Matplotlib versions
# #         handles, labels = ax.get_legend_handles_labels()
# #         by_label = dict(zip(labels, handles))
# #         leg = plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10, framealpha=0.9)

# #         # Try to bold line entries if supported
# #         for lh in getattr(leg, "legend_handles", []):
# #             try:
# #                 lh.set_linewidth(2.5)
# #             except Exception:
# #                 pass

# #         plt.grid(alpha=0.3); plt.tight_layout()
# #         plt.savefig(out_root/(f"{kind}_all_{args.tag}.png"), dpi=180); plt.close()

# #     _plot_all("roc")
# #     _plot_all("pr")
# #     print(f"[plots] ROC/PR saved in {out_root}")

# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# import argparse, csv, shutil, datetime
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams["figure.autolayout"] = True

# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # --------- basic metrics ----------
# def confusion(y, yhat):
#     tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
#     fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
#     return tp,fp,fn,tn

# def metrics(tp,fp,fn,tn):
#     prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
#     rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
#     f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
#     acc  = (tp+tn)/max(tp+fp+fn+tn,1)
#     return prec,rec,f1,acc

# def infer_model_from_path(p: Path):
#     s = p.stem.lower()
#     if "webrtc_l2" in s: return "webrtc_l2"
#     if "webrtc_l3" in s: return "webrtc_l3"
#     for k in ("energy","zcr","combo"):
#         if k in s: return k
#     return s

# # --------- curves ----------
# def compute_curves(labels, scores):
#     labels = labels.astype(np.int32); scores = scores.astype(np.float64)
#     if len(labels)==0:
#         return dict(fpr=np.array([0,1.0]), tpr=np.array([0,1.0]), auc=np.nan,
#                     recall=np.array([0,1.0]), precision=np.array([1.0,0]), ap=np.nan)
#     order = np.argsort(scores)[::-1]
#     y = labels[order]
#     tp = np.cumsum(y)
#     fp = np.cumsum(1-y)
#     P = float(tp[-1]); N = float(fp[-1])
#     sc = scores[order]
#     ch = np.r_[True, sc[1:]!=sc[:-1]]
#     tp = tp[ch]; fp=fp[ch]
#     tpr = tp/max(P,1.0); fpr = fp/max(N,1.0)
#     # use trapezoid (np.trapz is deprecated alias)
#     auc = float(np.trapezoid(tpr, fpr))
#     recall = tpr
#     precision = tp/np.maximum(tp+fp,1)
#     ap = float(np.sum((recall[1:]-recall[:-1])*precision[1:])) if len(recall)>1 else float("nan")
#     return dict(fpr=fpr,tpr=tpr,auc=auc,recall=recall,precision=precision,ap=ap)

# def pick_low_fp_point(roc, target_recall=0.80):
#     m = roc["tpr"] >= target_recall
#     if not np.any(m):
#         j = int(np.argmax(roc["tpr"]))
#     else:
#         cand = np.where(m)[0]
#         j = cand[int(np.argmin(roc["fpr"][cand]))]
#     return float(roc["fpr"][j]), float(roc["tpr"][j])

# # --------- summary table ----------
# def plot_summary_table(df, tag, out_png):
#     fig, ax = plt.subplots(figsize=(18, 4.5), dpi=200)
#     ax.axis("off")
#     tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(10)
#     tbl.scale(1.2, 1.2)
#     ax.set_title(f"VAD Summary — {tag}", pad=16, fontsize=16)
#     fig.savefig(out_png, bbox_inches="tight")
#     plt.close(fig)

# # --------- confusion bar ----------
# def plot_conf_bar(rows, out_png, title):
#     names = [r["model"] for r in rows]
#     tp = np.array([r["tp"] for r in rows]); fp=np.array([r["fp"] for r in rows])
#     fn = np.array([r["fn"] for r in rows]); tn=np.array([r["tn"] for r in rows])
#     x = np.arange(len(names))
#     plt.figure(figsize=(9.5,5))
#     plt.bar(x, tp, label="TP")
#     plt.bar(x, fp, bottom=tp, label="FP")
#     plt.bar(x, fn, bottom=tp+fp, label="FN")
#     plt.bar(x, tn, bottom=tp+fp+fn, label="TN")
#     plt.xticks(x, [n.replace("_"," ").title() for n in names])
#     plt.ylabel("Count"); plt.title(title); plt.legend()
#     plt.savefig(out_png, dpi=180); plt.close()

# # --------- frame-file discovery ----------
# def find_frame_files(tag, models_expected):
#     base = Path("outputs/frames") / tag
#     found = {m: [] for m in models_expected}
#     if not base.exists():
#         return found
#     prefix = {
#         "energy":     f"frame_scores_Energy_{tag}.csv",
#         "zcr":        f"frame_scores_ZCR_{tag}.csv",
#         "combo":      f"frame_scores_Combo_{tag}.csv",
#         "webrtc_l2":  f"frame_scores_webrtc_l2_{tag}.csv",
#         "webrtc_l3":  f"frame_scores_webrtc_l3_{tag}.csv",
#     }
#     for m in models_expected:
#         path = base / m / prefix[m]
#         if path.exists():
#             found[m] = [path]
#     return found

# def copy_inputs(dest_inputs_dir: Path, clip_paths):
#     dest_inputs_dir.mkdir(parents=True, exist_ok=True)
#     for p in clip_paths:
#         try:
#             shutil.copy2(p, dest_inputs_dir / Path(p).name)
#         except Exception:
#             pass

# # annotate near markers
# def annotate(ax, x, y, text):
#     ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 6),
#                 fontsize=9, ha="left", va="bottom",
#                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# def main():
#     ap = argparse.ArgumentParser(description="Evaluate VAD (clips + ROC/PR)")
#     ap.add_argument("--clips", nargs="+", required=True, help="clip_results_*.csv (all models)")
#     ap.add_argument("--frame_scores", nargs="*", default=[], help="optional explicit frame CSVs")
#     ap.add_argument("--tag", required=True, help="e.g. light / heavy")
#     ap.add_argument("--runtime_summary_dir", required=True, help="outputs/runtime/<TAG>")
#     ap.add_argument("--recall_target", type=float, default=0.80)
#     ap.add_argument("--make_confbar", action="store_true")
#     args = ap.parse_args()

#     stamp = now_tag()
#     out_root = Path("outputs/eval") / f"{args.tag}__{stamp}"
#     out_root.mkdir(parents=True, exist_ok=True)

#     # Copy inputs for traceability
#     copy_inputs(out_root/"inputs", args.clips)

#     # (1) Clip summary + OP from clips
#     rows=[]; op_points={}
#     models_in_order=[]
#     for cfile in args.clips:
#         cpath = Path(cfile)
#         model = infer_model_from_path(cpath)
#         models_in_order.append(model)
#         y=[]; yhat=[]
#         with open(cpath,"r",encoding="utf-8") as f:
#             rd=csv.DictReader(f)
#             for r in rd:
#                 y.append(int(r["label"])); yhat.append(int(r["pred"]))
#         y=np.array(y,np.int32); yhat=np.array(yhat,np.int32)
#         tp,fp,fn,tn = confusion(y,yhat); pr,rc,f1,acc = metrics(tp,fp,fn,tn)
#         rows.append(dict(model=model,tp=tp,fp=fp,fn=fn,tn=tn,
#                          precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),acc=round(acc,4)))
#         # clip OP star coords
#         FPR = fp/max(fp+tn,1); TPR = rc
#         op_points[model] = {"roc":(FPR,TPR), "pr":(rc,pr)}

#     # Merge runtime (if present)
#     rt = Path(args.runtime_summary_dir)/f"runtime_summary_{args.tag}.csv"
#     rt_map={}
#     if rt.exists():
#         with open(rt,"r",encoding="utf-8") as f:
#             rd=csv.DictReader(f)
#             for r in rd:
#                 rt_map[r["model"].lower()] = r

#     # write summary table (csv)
#     out_csv = out_root/f"summary_metrics_{args.tag}.csv"
#     headers = ["model","tp","fp","fn","tn","precision","recall","f1","acc",
#                "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
#     with open(out_csv,"w",newline="",encoding="utf-8") as f:
#         w=csv.DictWriter(f, fieldnames=headers); w.writeheader()
#         for r in rows:
#             rtrow = rt_map.get(r["model"].lower(), {})
#             r_out = dict(r)
#             r_out.update({
#                 "sec_per_hour": rtrow.get("sec_per_hour",""),
#                 "sec_per_hour_mean": rtrow.get("sec_per_hour_mean",""),
#                 "sec_per_hour_std": rtrow.get("sec_per_hour_std",""),
#                 "repeats": rtrow.get("repeats",""),
#             })
#             w.writerow(r_out)

#     # pretty table
#     try:
#         import pandas as pd
#         df = pd.read_csv(out_csv)
#         plot_summary_table(df, args.tag, out_root/f"summary_table_{args.tag}.png")
#     except Exception:
#         pass

#     if args.make_confbar:
#         plot_conf_bar(rows, out_root/f"confusion_bar_{args.tag}.png", f"Confusion counts — {args.tag}")

#     # (2) Frame scores → ROC/PR
#     # Discover frame files if not given explicitly
#     frame_files = list(map(Path, args.frame_scores))
#     if not frame_files:
#         discovered = find_frame_files(args.tag, models_in_order)
#         for m in models_in_order:
#             frame_files.extend(discovered.get(m, []))

#     if not frame_files:
#         print("[info] no frame scores found; ROC/PR skipped.")
#         return

#     # build curves and also dump per-model CSV of curves
#     models=[]; curves=[]
#     roc_dir = (out_root/"roc"); pr_dir = (out_root/"pr")
#     roc_dir.mkdir(exist_ok=True); pr_dir.mkdir(exist_ok=True)

#     for fp in frame_files:
#         name = fp.name.lower()
#         if "webrtc_l2" in name: m = "webrtc_l2"
#         elif "webrtc_l3" in name: m = "webrtc_l3"
#         elif "_energy_" in name: m = "energy"
#         elif "_zcr_" in name: m = "zcr"
#         elif "_combo_" in name: m = "combo"
#         else: m = infer_model_from_path(fp)

#         sc=[]; lab=[]
#         with open(fp,"r",encoding="utf-8") as f:
#             rd=csv.DictReader(f)
#             for r in rd:
#                 p = r.get("prob", r.get("score","0"))
#                 sc.append(float(p))
#                 lf = r.get("label_frame", r.get("pred_post","0"))
#                 lab.append(int(lf))
#         sc=np.array(sc,np.float64); lab=np.array(lab,np.int32)
#         rc = compute_curves(lab, sc)
#         models.append(m); curves.append(rc)

#         # export curve points for traceability
#         with open(roc_dir/f"roc_{m}.csv","w",newline="",encoding="utf-8") as f:
#             w=csv.writer(f); w.writerow(["fpr","tpr"]); w.writerows(zip(rc["fpr"], rc["tpr"]))
#         with open(pr_dir/f"pr_{m}.csv","w",newline="",encoding="utf-8") as f:
#             w=csv.writer(f); w.writerow(["recall","precision"]); w.writerows(zip(rc["recall"], rc["precision"]))

#     # combined plots + markers (markers match line colors & are annotated)
#     def _plot_all(kind):
#         fig, ax = plt.subplots(figsize=(9.5,7.5))
#         handles_by_model = {}
#         for m,rc in zip(models,curves):
#             if kind=="roc":
#                 h, = ax.plot(rc["fpr"], rc["tpr"], label=f"{m} (AUC={rc['auc']:.3f})")
#             else:
#                 h, = ax.plot(rc["recall"], rc["precision"], label=f"{m} (AP={rc['ap']:.3f})")
#             handles_by_model[m] = h

#         # OP stars & low-FP diamonds (colored like the line) + numeric annotations
#         for m in models_in_order:
#             if m not in handles_by_model: continue
#             color = handles_by_model[m].get_color()
#             pts = op_points.get(m)
#             if not pts: continue
#             if kind=="roc":
#                 x,y = pts["roc"]; ax.scatter([x],[y], marker="*", s=160, c=[color], edgecolors="k", zorder=5)
#                 annotate(ax, x, y, f"FPR={x:.2f}, TPR={y:.2f}")
#             else:
#                 x,y = pts["pr"]; ax.scatter([x],[y], marker="*", s=160, c=[color], edgecolors="k", zorder=5)
#                 annotate(ax, x, y, f"R={x:.2f}, P={y:.2f}")

#         if kind=="roc":
#             for m,rc in zip(models,curves):
#                 color = handles_by_model[m].get_color()
#                 x,y = pick_low_fp_point(rc, target_recall=args.recall_target)
#                 ax.scatter([x],[y], marker="D", s=90, c=[color], edgecolors="k", zorder=5)
#                 annotate(ax, x, y, f"FPR={x:.2f}, TPR={y:.2f}")

#         if kind=="roc":
#             ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
#             ax.set_title(f"ROC — all models [{args.tag}]")
#         else:
#             ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
#             ax.set_title(f"PR — all models [{args.tag}]")

#         ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
#         fig.tight_layout()
#         fig.savefig(out_root/(f"{kind}_all_{args.tag}.png"), dpi=180)
#         plt.close(fig)

#     _plot_all("roc")
#     _plot_all("pr")
#     print(f"[plots] ROC/PR saved in {out_root}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse, csv, datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------- basic metrics ----------
def confusion(y, yhat):
    tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
    fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
    return tp,fp,fn,tn

def metrics(tp,fp,fn,tn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tp+tn)/max(tp+fp+fn+tn,1)
    return prec,rec,f1,acc

def infer_model_from_path(p: Path):
    s = p.stem.lower()
    if "webrtc_l2" in s: return "webrtc_l2"
    if "webrtc_l3" in s: return "webrtc_l3"
    for k in ("energy","zcr","combo"):
        if k in s: return k
    return s

# ---------- curves ----------
def compute_curves(labels, scores):
    labels = labels.astype(np.int32); scores = scores.astype(np.float64)
    if len(labels)==0:
        return dict(fpr=np.array([0,1.0]), tpr=np.array([0,1.0]), auc=np.nan,
                    recall=np.array([0,1.0]), precision=np.array([1.0,0]), ap=np.nan)
    order = np.argsort(scores)[::-1]
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1-y)
    P = float(tp[-1]); N = float(fp[-1])
    sc = scores[order]
    ch = np.r_[True, sc[1:]!=sc[:-1]]
    tp = tp[ch]; fp=fp[ch]

    tpr = tp/ max(P,1.0); fpr = fp/ max(N,1.0)
    auc = float(np.trapezoid(tpr, fpr))
    recall = tpr
    precision = tp/np.maximum(tp+fp,1)
    ap = float(np.sum((recall[1:]-recall[:-1])*precision[1:])) if len(recall)>1 else float('nan')
    return dict(fpr=fpr,tpr=tpr,auc=auc,recall=recall,precision=precision,ap=ap)

def pick_low_fp_point(roc, target_recall=0.80):
    m = roc["tpr"] >= target_recall
    if not np.any(m):
        j = int(np.argmax(roc["tpr"]))
    else:
        cand = np.where(m)[0]
        j = cand[int(np.argmin(roc["fpr"][cand]))]
    return float(roc["fpr"][j]), float(roc["tpr"][j])

# ---------- pretty summary table ----------
def plot_summary_table(df, tag, out_png):
    fig, ax = plt.subplots(figsize=(18, 4.5), dpi=200)
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 1.15)
    ax.set_title(f"VAD Summary — {tag}", pad=16, fontsize=16)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

# ---------- confusion bar ----------
def plot_conf_bar(rows, out_png, title):
    names = [r["model"] for r in rows]
    tp = np.array([r["tp"] for r in rows]); fp=np.array([r["fp"] for r in rows])
    fn = np.array([r["fn"] for r in rows]); tn=np.array([r["tn"] for r in rows])
    x = np.arange(len(names))
    plt.figure(figsize=(9.5,5))
    plt.bar(x, tp, label="TP")
    plt.bar(x, fp, bottom=tp, label="FP")
    plt.bar(x, fn, bottom=tp+fp, label="FN")
    plt.bar(x, tn, bottom=tp+fp+fn, label="TN")
    plt.xticks(x, [n.replace("_"," ").title() for n in names])
    plt.ylabel("Count"); plt.title(title); plt.legend()
    plt.savefig(out_png, dpi=180); plt.close()

# ---------- frame-file discovery ----------
def find_frame_files(tag, models_expected):
    base = Path("outputs/frames") / tag
    found = {m: [] for m in models_expected}
    if not base.exists():
        return found
    prefix = {
        'energy':     'frame_scores_Energy_',
        'zcr':        'frame_scores_ZCR_',
        'combo':      'frame_scores_Combo_',
        'webrtc_l2':  'frame_scores_webrtc_l2_',
        'webrtc_l3':  'frame_scores_webrtc_l3_',
    }
    for m in models_expected:
        sub = base / m
        if sub.is_dir():
            found[m] = sorted(sub.glob(f"{prefix[m]}{tag}.csv"))
        else:
            found[m] = []
    return found

def main():
    ap = argparse.ArgumentParser(description="Evaluate VAD (clips + ROC/PR)")
    ap.add_argument("--clips", nargs="+", required=True, help="clip_results_*.csv (all models)")
    ap.add_argument("--frame_scores", nargs="*", default=[], help="optional explicit frame CSVs")
    ap.add_argument("--tag", required=True, help="e.g. light / heavy")
    ap.add_argument("--runtime_summary_dir", required=True, help="outputs/runtime/<TAG>")
    ap.add_argument("--recall_target", type=float, default=0.80)
    ap.add_argument("--make_confbar", action="store_true")
    args = ap.parse_args()

    stamp = now_tag()
    out_root = Path("outputs/eval")/f"{args.tag}__{stamp}"
    (out_root / "roc").mkdir(parents=True, exist_ok=True)
    (out_root / "pr").mkdir(parents=True, exist_ok=True)
    (out_root / "inputs").mkdir(parents=True, exist_ok=True)

    # (1) Clip summary + OP from clips
    rows=[]; op_points={}
    models_in_order=[]
    for cfile in args.clips:
        cpath = Path(cfile)
        model = infer_model_from_path(cpath)
        models_in_order.append(model)
        # copy inputs for bookkeeping
        try:
            dst = out_root/"inputs"/cpath.name
            if cpath.resolve() != dst.resolve():
                dst.write_bytes(cpath.read_bytes())
        except Exception:
            pass

        y=[]; yhat=[]
        with open(cpath,"r",encoding="utf-8") as f:
            rd=csv.DictReader(f)
            for r in rd:
                y.append(int(r["label"])); yhat.append(int(r["pred"]))
        y=np.array(y,np.int32); yhat=np.array(yhat,np.int32)
        tp,fp,fn,tn = confusion(y,yhat); pr,rc,f1,acc = metrics(tp,fp,fn,tn)
        rows.append(dict(model=model,tp=tp,fp=fp,fn=fn,tn=tn,
                         precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),acc=round(acc,4)))
        # clip OP star coords
        FPR = fp/max(fp+tn,1); TPR = rc
        op_points[model] = {"roc":(FPR,TPR), "pr":(rc,pr)}

    # Merge runtime (if present)
    rt = Path(args.runtime_summary_dir)/f"runtime_summary_{args.tag}.csv"
    rt_map={}
    if rt.exists():
        with open(rt,"r",encoding="utf-8") as f:
            rd=csv.DictReader(f)
            for r in rd:
                rt_map[r["model"].lower()] = r

    # write summary table (csv)
    out_csv = out_root/f"summary_metrics_{args.tag}.csv"
    headers = ["model","tp","fp","fn","tn","precision","recall","f1","acc",
               "sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats"]
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows:
            rtrow = rt_map.get(r["model"].lower(), {})
            r_out = dict(r)
            r_out.update({
                "sec_per_hour": rtrow.get("sec_per_hour",""),
                "sec_per_hour_mean": rtrow.get("sec_per_hour_mean",""),
                "sec_per_hour_std": rtrow.get("sec_per_hour_std",""),
                "repeats": rtrow.get("repeats",""),
            })
            w.writerow(r_out)

    # table PNG (readable)
    try:
        import pandas as pd
        df = pd.read_csv(out_csv)
        plot_summary_table(df, args.tag, out_root/f"summary_table_{args.tag}.png")
    except Exception:
        pass

    if args.make_confbar:
        plot_conf_bar(rows, out_root/f"confusion_bar_{args.tag}.png",
                      f"Confusion counts — {args.tag}")

    # (2) Frame scores → ROC/PR
    # Discover frame files if not given explicitly
    frame_files = list(map(Path, args.frame_scores))
    if not frame_files:
        discovered = find_frame_files(args.tag, models_in_order)
        for m in models_in_order:
            frame_files.extend(discovered.get(m, []))

    if not frame_files:
        print("[info] no frame scores found; ROC/PR skipped.")
        return

    # build curves
    models=[]; curves=[]
    for fp in frame_files:
        name = fp.name.lower()
        if      "webrtc_l2" in name: m = "webrtc_l2"
        elif    "webrtc_l3" in name: m = "webrtc_l3"
        elif    "_energy_"  in name: m = "energy"
        elif    "_zcr_"     in name: m = "zcr"
        elif    "_combo_"   in name: m = "combo"
        else: m = infer_model_from_path(fp)

        sc=[]; lab=[]
        # Expect columns: model, source, frame_idx, score, prob, label_frame OR pred_post
        with open(fp,"r",encoding="utf-8") as f:
            rd=csv.DictReader(f)
            for r in rd:
                p = r.get("prob", r.get("score","0"))
                sc.append(float(p))
                lf = r.get("label_frame", r.get("pred_post","0"))  # fallback for energy/zcr/combo
                lab.append(int(lf))
        sc=np.array(sc,np.float64); lab=np.array(lab,np.int32)
        rocpr = compute_curves(lab, sc)
        models.append(m); curves.append(rocpr)

    # combined plots + markers (legend-only numbers, no on-plot text)
    def _plot_all(kind):
        fig, ax = plt.subplots(figsize=(9.5,7.5))
        handles_by_model = {}

        for m,rc in zip(models,curves):
            if kind=="roc":
                h, = ax.plot(rc["fpr"], rc["tpr"], label=f"{m} (AUC={rc['auc']:.3f})")
            else:
                h, = ax.plot(rc["recall"], rc["precision"], label=f"{m} (AP={rc['ap']:.3f})")
            handles_by_model[m] = h

        # add OP and low-FP with curve-matched colors; numbers go into legend label
        legend_extra = []

        for m in models_in_order:
            if m not in handles_by_model: continue
            color = handles_by_model[m].get_color()
            pts = op_points.get(m);  
            if not pts: continue
            if kind=="roc":
                x,y = pts["roc"]
                h = ax.scatter([x],[y], marker="*", s=160, c=[color], edgecolors="k", zorder=5)
                legend_extra.append((h, f"{m} op (FPR={x:.2f}, TPR={y:.2f})"))
            else:
                x,y = pts["pr"]
                h = ax.scatter([x],[y], marker="*", s=160, c=[color], edgecolors="k", zorder=5)
                legend_extra.append((h, f"{m} op (R={x:.2f}, P={y:.2f})"))

        if kind=="roc":
            for m,rc in zip(models,curves):
                color = handles_by_model[m].get_color()
                x,y = pick_low_fp_point(rc, target_recall=args.recall_target)
                h = ax.scatter([x],[y], marker="D", s=90, c=[color], edgecolors="k", zorder=5)
                legend_extra.append((h, f"{m} low-FP (FPR={x:.2f}, TPR={y:.2f})"))

        if kind=="roc":
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC — all models [{args.tag}]")
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"PR — all models [{args.tag}]")

        h0, l0 = ax.get_legend_handles_labels()
        for h, lbl in legend_extra:
            h0.append(h); l0.append(lbl)
        ax.legend(h0, l0, loc="lower right", fontsize=10, framealpha=0.9)

        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_root/(("roc" if kind=="roc" else "pr") + f"/{kind}_all_{args.tag}.png"), dpi=180)
        plt.close(fig)

    _plot_all("roc")
    _plot_all("pr")
    print(f"[plots] ROC/PR saved in {out_root}")

if __name__ == "__main__":
    main()
