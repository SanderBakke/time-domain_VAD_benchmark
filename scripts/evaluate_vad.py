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
