#!/usr/bin/env python
import argparse, numpy as np, matplotlib.pyplot as plt, csv
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

def load_npz(p): 
    z = np.load(p); return z["y"], z["s"]

def fp_per_hour(y, s, thr, fps):
    yp = (s >= thr).astype(np.int32); noise = (y == 0)
    fp = int(((yp==1) & noise).sum()); nf = int(noise.sum())
    return float("nan") if nf<=0 else (fp/nf)*(fps*3600.0)

ap = argparse.ArgumentParser()
ap.add_argument("--scores_npz", action="append", required=True)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--outdir", default="outputs/pr")
ap.add_argument("--filename_pr", default="pr_all.png")
ap.add_argument("--filename_fp", default="fp_per_hour_vs_recall.png")
ap.add_argument("--frame_hop_ms", type=float, default=10.0)
ap.add_argument("--fp_target", type=float, default=None)
ap.add_argument("--dpi", type=int, default=150)
args = ap.parse_args()

labels = args.labels or [Path(p).stem for p in args.scores_npz]
fps = 1000.0/args.frame_hop_ms
outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6,5)); fig_fp, ax_fp = plt.subplots(figsize=(6,5))
rows=[]
for p,lab in zip(args.scores_npz, labels):
    y,s = load_npz(p)
    prec, rec, thr = precision_recall_curve(y, s)
    ap_score = float(average_precision_score(y, s))
    if thr.size==0:
        rows.append({"label":lab,"average_precision":ap_score}); continue
    thr_ext = np.concatenate([thr, [thr[-1]]])
    fph = np.array([fp_per_hour(y, s, t, fps) for t in thr])
    if fph.size+1 == prec.size: fph = np.concatenate([fph, [fph[-1]]])
    np.savetxt(outdir/f"pr_{lab}.csv", np.c_[prec,rec,thr_ext,fph], delimiter=",", header="precision,recall,threshold,fp_per_hour", comments="")
    plt.plot(rec, prec, label=f"{lab} (AP={ap_score:.3f})")
    ax_fp.plot(rec[:-1] if rec.size>1 else rec, fph, label=lab)
    sel={}
    if args.fp_target is not None and fph.size>0:
        ok = np.where(np.isfinite(fph) & (fph<=args.fp_target))[0]
        idx = int(ok[np.argmax(rec[ok])]) if ok.size>0 else int(np.nanargmin(fph))
        sel = {"threshold": float(thr[idx]), "recall": float(rec[idx]), "precision": float(prec[idx]), "fp_per_hour": float(fph[idx])}
    rows.append({"label":lab,"average_precision":ap_score, **sel})

plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curves")
plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(outdir/args.filename_pr, dpi=args.dpi); plt.close()
ax_fp.set_xlabel("Recall"); ax_fp.set_ylabel("FP per hour (on noise)")
ax_fp.set_title("FP/hour vs Recall"); ax_fp.legend(loc="upper right")
fig_fp.tight_layout(); fig_fp.savefig(outdir/args.filename_fp, dpi=args.dpi); plt.close(fig_fp)
with open(outdir/"summary.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["label","average_precision","threshold","recall","precision","fp_per_hour"]); w.writeheader()
    for r in rows:
        for k in ["threshold","recall","precision","fp_per_hour"]: r.setdefault(k,"")
        w.writerow(r)
print(f"[OK] Wrote PR figure → {outdir/args.filename_pr}")
print(f"[OK] Wrote FP/hour vs Recall → {outdir/args.filename_fp}")
print(f"[OK] Summary → {outdir/'summary.csv'}")
