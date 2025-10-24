#!/usr/bin/env python3
# scripts/evaluate_vad.py
# - Clip summary (+ optional confusion bar)
# - Auto-discover frame scores by tag (or accept explicit list)
# - ROC/PR per model + combined, with OP markers
# - Merge runtime summaries into the metrics table
# - All outputs isolated under outputs/eval/<TAG>__<STAMP>/

import argparse, csv, datetime, json, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------------- Clip helpers ----------------
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
    plt.figure(figsize=(max(6, 1.0*len(labels)), 3.5))
    bottom = np.zeros_like(tp, dtype=float)
    for arr, name, color in [(tp,"TP","#2ca02c"), (fp,"FP","#d62728"),
                             (fn,"FN","#ff7f0e"), (tn,"TN","#1f77b4")]:
        plt.bar(labels, arr, width=width, bottom=bottom, label=name, color=color)
        bottom += arr
    plt.xticks(rotation=0)
    plt.ylabel("Count"); plt.title(title)
    plt.legend(ncol=4, bbox_to_anchor=(0.5,1.15), loc="upper center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

# ---------------- Curves (frame-level) ----------------
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
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(roc_dir / f"roc_{model}.png", dpi=140); plt.close()

    # PR
    plt.figure()
    plt.plot(pr["recall"], pr["precision"], label=f"{model} (AP={pr['ap']:.3f})")
    if op_points and model in op_points:
        rec_pt, prec_pt = op_points[model]["pr"]
        plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{model} op")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {model}{title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(pr_dir / f"pr_{model}.png", dpi=140); plt.close()

def plot_compare(all_models, roc_dir: Path, pr_dir: Path, tag: str, op_points=None):
    # ROC
    plt.figure(figsize=(7.5, 6))
    for m in all_models:
        roc = m["roc"]; name = m["name"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            fpr_pt, tpr_pt = pts["roc"]
            plt.scatter([fpr_pt],[tpr_pt], marker="*", s=90, zorder=5, label=f"{name} op")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — all models [{tag}]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(roc_dir / f"roc_all_{tag}.png", dpi=150); plt.close()
    # PR
    plt.figure(figsize=(7.5, 6))
    for m in all_models:
        pr = m["pr"]; name = m["name"]
        plt.plot(pr["recall"], pr["precision"], label=f"{name} (AP={pr['ap']:.3f})")
    if op_points:
        for name, pts in op_points.items():
            rec_pt, prec_pt = pts["pr"]
            plt.scatter([rec_pt],[prec_pt], marker="*", s=90, zorder=5, label=f"{name} op")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — all models [{tag}]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(pr_dir / f"pr_all_{tag}.png", dpi=150); plt.close()

# --------------- discovery helpers ---------------
def discover_frames(frame_root: Path, tag: str, models: list[str]):
    """Return dict model -> path to frame_scores_<Model>_<tag>.csv if present."""
    found = {}
    for m in models:
        f = frame_root / m / f"frame_scores_{m}_{tag}.csv"
        if f.exists():
            found[m] = f
    return found

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate VAD: clip-level + ROC/PR")
    ap.add_argument("--clips", nargs="+", required=True, help="Clip-level result CSVs (per model).")
    ap.add_argument("--frame_scores", nargs="*", default=[], help="Optional explicit frame-score files.")
    ap.add_argument("--frame_root", type=str, default=None, help="Root to auto-find frames, e.g., outputs/frames/<TAG>")
    ap.add_argument("--tag", type=str, required=True, help="Dataset/eval tag, e.g., light/heavy")
    ap.add_argument("--runtime_summary_dir", type=str, required=True, help="Folder with runtime_summary_<TAG>__*.csv")
    ap.add_argument("--make_confbar", action="store_true", help="Save a confusion bar from clip CSVs")
    args = ap.parse_args()

    tag = args.tag
    stamp = now_tag()
    eval_run_dir = Path("outputs/eval") / f"{tag}__{stamp}"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    (eval_run_dir / "inputs").mkdir(parents=True, exist_ok=True)

    # ---- (1) Clip summary + op-point extraction ----
    op_points = {}
    summary_rows = []
    models = []

    for clip_csv in args.clips:
        clip_csv = Path(clip_csv)
        # copy inputs for traceability
        try:
            shutil.copy2(clip_csv, eval_run_dir / "inputs" / clip_csv.name)
        except Exception:
            pass

        y, p, _ = load_clip_results(clip_csv)
        tp, fp, fn, tn = confusion(y, p)
        prec, rec, f1, acc = metrics_from_conf(tp, fp, fn, tn)

        model_name = infer_model_from_clip_name(clip_csv)
        models.append(model_name)

        summary_rows.append({
            "model": model_name,
            "file": clip_csv.name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
            "f1": f"{f1:.4f}", "acc": f"{acc:.4f}",
        })

        TPR = rec
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        op_points[model_name] = {"roc": (FPR, TPR), "pr": (rec, prec)}

    # merge runtime summaries
    rs_dir = Path(args.runtime_summary_dir)
    rt_cols = {"sec_per_hour":"","sec_per_hour_mean":"","sec_per_hour_std":"","repeats":""}
    if rs_dir.exists():
        # most recent summary per model (by reading all summaries in folder)
        per_model = {}
        for f in sorted(rs_dir.glob("runtime_summary_*__*.csv")):
            with open(f, "r", encoding="utf-8") as h:
                rd = csv.DictReader(h)
                for r in rd:
                    per_model[r["model"]] = {
                        "sec_per_hour": r.get("sec_per_hour",""),
                        "sec_per_hour_mean": r.get("sec_per_hour_mean",""),
                        "sec_per_hour_std": r.get("sec_per_hour_std",""),
                        "repeats": r.get("repeats",""),
                    }
        for row in summary_rows:
            row.update(per_model.get(row["model"], rt_cols.copy()))
    else:
        for row in summary_rows:
            row.update(rt_cols.copy())

    out_csv = eval_run_dir / f"summary_metrics_{tag}.csv"
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"[clip] wrote {out_csv}")

    if args.make_confbar:
        conf_png = eval_run_dir / f"confusion_bar_{tag}.png"
        plot_confusion_bar(summary_rows, conf_png, f"Confusion counts — {tag}")
        print(f"[clip] wrote {conf_png}")

    # ---- (2) Frame-level curves ----
    # collect frame files: explicit list or auto-discover by tag
    frame_files = [Path(p) for p in args.frame_scores] if args.frame_scores else []
    if (not frame_files) and args.frame_root:
        auto = discover_frames(Path(args.frame_root), tag, models)
        frame_files = list(auto.values())

    if not frame_files:
        print("[info] no frame scores passed; ROC/PR skipped.")
        return

    # load, compute curves
    def load_frame_csv(p: Path):
        labels, scores = [], []
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                labels.append(int(r.get("label_frame", "0")))
                s = r.get("prob","")
                s = float(s) if s not in ("", None) else float(r.get("score","0.0"))
                scores.append(s)
        return np.asarray(labels, np.int32), np.asarray(scores, np.float64)

    roc_dir = eval_run_dir / "roc"
    pr_dir  = eval_run_dir / "pr"

    combined = []
    for p in frame_files:
        # infer model from filename
        name = p.stem.replace("frame_scores_","")
        # e.g. "Energy_light" -> model is part before "_<tag>"
        if name.endswith(f"_{tag}"):
            name = name[:-(len(tag)+1)]
        model = name
        y, s = load_frame_csv(p)
        res = compute_curves(y, s)
        roc={"fpr":res["fpr"],"tpr":res["tpr"],"auc":res["auc"]}
        pr ={"recall":res["recall"],"precision":res["precision"],"ap":res["ap"]}
        save_curve_npz_png(roc_dir, pr_dir, model, roc, pr, title_suffix=f"  [{tag}]", op_points=op_points)
        combined.append({"name":model,"roc":roc,"pr":pr})

    plot_compare(combined, roc_dir, pr_dir, tag, op_points=op_points)
    print(f"[curves] wrote {roc_dir / f'roc_all_{tag}.png'} and {pr_dir / f'pr_all_{tag}.png'}")

    # summary table image
    # (simple one-row-per-model table)
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(out_csv)
    fig, ax = plt.subplots(figsize=(16,4))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.4)
    plt.title(f"VAD Summary — {tag}")
    out_png = eval_run_dir / f"summary_table_{tag}.png"
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[table] wrote {out_png}")

if __name__ == "__main__":
    main()
