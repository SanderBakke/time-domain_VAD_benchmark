#!/usr/bin/env python3
# scripts/tune_vad.py
# Policy-only sweep using saved frame CSVs (prefers pred_raw; falls back to label_frame/pred_post).
# Output:
#   outputs/tuning/<TAG>/tuning_results.csv            (best feasible rows per model)
#   outputs/tuning/<TAG>/tuning_results_all.csv        (every evaluated row)
#   outputs/tuning/<TAG>/best_params_<TAG>.json        (min_speech_frames + hangover_ms per model)

import argparse, csv, json
from pathlib import Path
import numpy as np

def apply_hangover(bits, n: int):
    if n <= 0:
        return bits
    y = bits.copy(); last = -10**9
    for i, v in enumerate(y):
        if v == 1:
            last = i
        elif i - last <= n:
            y[i] = 1
    return y

def confusion(y, yhat):
    tn = int(((y==0)&(yhat==0)).sum()); tp = int(((y==1)&(yhat==1)).sum())
    fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
    return tp,fp,fn,tn

def metrics(tp,fp,fn,tn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec,rec,f1

def main():
    ap = argparse.ArgumentParser(description="Tune clip policy from saved frames")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clips_dir", default=None, help="defaults to outputs/clips/<TAG>")
    ap.add_argument("--frames_dir", default=None, help="defaults to outputs/frames/<TAG>")
    ap.add_argument("--hang_grid", default="0,50,100,150,200", help="hangover ms grid")
    ap.add_argument("--min_frames_grid", default="1,2,3,4,5,6,8", help="min speech frames grid")
    ap.add_argument("--hop_ms", type=int, default=16, help="for hangover frames conversion")
    ap.add_argument("--recall_target", type=float, default=0.80)
    args = ap.parse_args()

    tag = args.tag
    clips_dir   = Path(args.clips_dir)  if args.clips_dir  else Path("outputs/clips")/tag
    frames_root = Path(args.frames_dir) if args.frames_dir else Path("outputs/frames")/tag
    outdir = Path("outputs/tuning")/tag
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1) Load GT clip labels per source from any clip_results_*.csv
    gt_map = {}  # source -> label
    clip_files = sorted(clips_dir.glob("clip_results_*.csv"))
    if not clip_files:
        raise FileNotFoundError(f"No clip_results_*.csv found in {clips_dir}")
    with open(clip_files[0], "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt_map[r["source"]] = int(r["label"])

    # --- 2) Group frame bitstreams by (model, source)
    streams = {}  # model -> source -> [bits...]
    fcsvs = sorted(frames_root.rglob("frame_scores_*.csv"))
    if not fcsvs:
        raise FileNotFoundError(f"No frame_scores_*.csv found under {frames_root}")

    for fcsv in fcsvs:
        name = fcsv.name.lower()
        if   "webrtc_l2" in name: model = "webrtc_l2"
        elif "webrtc_l3" in name: model = "webrtc_l3"
        elif "_energy_" in name:  model = "energy"
        elif "_zcr_"    in name:  model = "zcr"
        elif "_combo_"  in name:  model = "combo"
        else:
            parts = fcsv.stem.split("_")
            model = parts[2].lower() if len(parts) > 2 else fcsv.stem.lower()

        with open(fcsv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                src = r["source"]
                # prefer pred_raw; else label_frame; else pred_post; else 0
                if "pred_raw" in r and r["pred_raw"] != "":
                    bit = int(r["pred_raw"])
                elif "label_frame" in r and r["label_frame"] != "":
                    bit = int(r["label_frame"])
                elif "pred_post" in r and r["pred_post"] != "":
                    bit = int(r["pred_post"])
                else:
                    bit = 0
                streams.setdefault(model, {}).setdefault(src, []).append(bit)

    # --- 3) Sweep
    hang_ms_grid = [int(x) for x in args.hang_grid.split(",") if x.strip()!=""]
    min_grid     = [int(x) for x in args.min_frames_grid.split(",") if x.strip()!=""]
    hop          = max(1, args.hop_ms)

    results = []        # best feasible rows per model (meets recall_target)
    all_rows = []       # every evaluated row (diagnostics)
    best     = {}       # json payload

    for model, by_src in streams.items():
        best_row = None
        feasible_found = False

        for hang_ms in hang_ms_grid:
            hang = max(0, int(round(hang_ms / hop)))  # convert ms to frames: hop_ms is the step
            for mf in min_grid:
                ys=[]; yh=[]
                for src, seq in by_src.items():
                    gt = gt_map.get(src, 1)  # default to 'speech' if label missing for this file
                    seq = np.asarray(seq, np.int32)
                    post = apply_hangover(seq, hang)
                    pred = int(np.sum(post) >= mf)
                    ys.append(gt); yh.append(pred)
                ys=np.array(ys,np.int32); yh=np.array(yh,np.int32)
                tp,fp,fn,tn = confusion(ys,yh)
                pr,rc,f1 = metrics(tp,fp,fn,tn)

                # record all attempts
                all_rows.append(dict(model=model, hang_ms=hang_ms, min_frames=mf,
                                     precision=pr, recall=rc, f1=f1, fp=fp))

                # keep only feasible rows (recall >= target) for "best"
                if rc < args.recall_target:
                    continue
                feasible_found = True
                row = dict(model=model, hang_ms=hang_ms, min_frames=mf, precision=pr, recall=rc, f1=f1, fp=fp)
                if (best_row is None or
                    (row["fp"] < best_row["fp"]) or
                    (row["fp"] == best_row["fp"] and row["precision"] > best_row["precision"]) or
                    (row["fp"] == best_row["fp"] and row["precision"] == best_row["precision"] and row["f1"] > best_row["f1"])):
                    best_row = row

        if feasible_found and best_row:
            results.append(best_row)
            best[model] = {"min_speech_frames": best_row["min_frames"], "hangover_ms": best_row["hang_ms"]}
        else:
            print(f"[tune] no policy met recall_target={args.recall_target:.2f} for model={model}; "
                  f"consider lowering --recall_target or widening grids.")

    # --- 4) Write outputs
    if results:
        with open(outdir/"tuning_results.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)

    if all_rows:
        with open(outdir/"tuning_results_all.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    with open(outdir/f"best_params_{tag}.json","w",encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"[tune] wrote {outdir}")

if __name__ == "__main__":
    main()
