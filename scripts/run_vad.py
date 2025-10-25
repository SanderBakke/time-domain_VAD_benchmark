#!/usr/bin/env python3
# scripts/run_vad.py
# Time-domain VAD baselines (Energy / ZCR / Combo) with SoC-aligned framing.
# Writes:
#   outputs/clips/<TAG>/clip_results_<model>_<TAG>.csv
#   outputs/frames/<TAG>/<Model>/frame_scores_<Model>_<TAG>.csv
#   outputs/runtime/<TAG>/timing_<Model>_<STAMP>.csv
#   outputs/runtime/<TAG>/runtime_summary_<TAG>.csv  (append)

import argparse, csv, time, datetime, os, sys
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vad.datasets import iter_dataset
from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD

NOISE_DIR_NAMES = {"_background_noise_", "noise", "silence"}
SR_DEFAULT = 16000

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def coerce_sample_to_tuple4(sample):
    # (source,x,fs,label) OR (source,x,fs) OR (x,fs,source)
    if isinstance(sample, (tuple, list)) and len(sample) == 4:
        source, x, fs, label = sample
        return str(source), np.asarray(x, np.float32), int(fs), int(label)
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        a,b,c = sample
        if isinstance(a,(str,Path)) and not isinstance(b,(str,Path)) and not isinstance(c,(str,Path)):
            source,x,fs = a,b,c
        elif not isinstance(a,(str,Path)) and not isinstance(b,(str,Path)) and isinstance(c,(str,Path)):
            x,fs,source = a,b,c
        else:
            raise ValueError(f"Bad 3-tuple: {type(a)}, {type(b)}, {type(c)}")
        parent = Path(str(source)).parent.name.lower()
        label = 0 if parent in NOISE_DIR_NAMES else 1
        return str(source), np.asarray(x, np.float32), int(fs), int(label)
    raise ValueError("Unsupported dataset sample format")

def median3(bits: np.ndarray) -> np.ndarray:
    if len(bits) < 3: return bits.copy()
    y = bits.copy()
    y[1:-1] = ((bits[:-2] + bits[1:-1] + bits[2:]) >= 2).astype(np.int32)
    return y

def apply_hangover(bits: np.ndarray, n: int) -> np.ndarray:
    if n <= 0: return bits
    y = bits.copy()
    last_on = -10**9
    for i,v in enumerate(y):
        if v == 1: last_on = i
        else:
            if i - last_on <= n: y[i] = 1
    return y

def clip_from_policy(bits: np.ndarray, min_frames: int) -> int:
    return int(np.sum(bits) >= int(min_frames))

def model_label(algo): return {"energy":"Energy","zcr":"ZCR","combo":"Combo"}[algo]

def ensure_dirs(tag: str, model: str):
    clips_dir   = Path("outputs/clips")   / tag
    frames_dir  = Path("outputs/frames")  / tag / model
    runtime_dir = Path("outputs/runtime") / tag
    for d in (clips_dir, frames_dir, runtime_dir): d.mkdir(parents=True, exist_ok=True)
    return clips_dir, frames_dir, runtime_dir

def compute_scores(E, Z, algo, k=2.0, gamma=1.0):
    eps = 1e-12
    nf_e = max(eps, np.percentile(E, 20) * 0.8)
    sE = np.zeros_like(E, dtype=float)
    for i, e in enumerate(E):
        a = 0.05
        nf_e = (1-a)*nf_e + a*min(e, nf_e)
        sE[i] = e / max(nf_e, eps)
    zref = max(eps, np.percentile(Z, 95))
    sZ = 1 - np.clip(Z / zref, 0, 1)
    if algo == "energy": s = sE
    elif algo == "zcr":  s = sZ
    else: s = np.minimum(sE, gamma*sZ)
    p = 1.0/(1.0+np.exp(-k*(s-1.0)))
    return s, p

def main():
    ap = argparse.ArgumentParser(description="Run Energy/ZCR/Combo VAD (SoC-aligned framing).")
    ap.add_argument("--algo", choices=["energy","zcr","combo"], required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--tag", required=True, help="dataset/eval tag, e.g. light / heavy")
    ap.add_argument("--emit_scores", action="store_true")

    # SoC defaults (TinyMLPerf-style)
    ap.add_argument("--frame_ms", type=int, default=32)
    ap.add_argument("--hop_ms",   type=int, default=16)

    # clip policy
    ap.add_argument("--median3", action="store_true")
    ap.add_argument("--hangover_ms", type=int, default=100)     # ≈ low-FP default
    ap.add_argument("--min_speech_frames", type=int, default=3) # ≈ 48 ms with 16ms hop

    # timing
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timing_note", type=str, default="")
    args = ap.parse_args()

    model = model_label(args.algo)
    clips_dir, frames_dir, runtime_dir = ensure_dirs(args.tag, model)
    stamp = now_tag()

    # instantiate model
    fps = 1000.0 / max(1,args.hop_ms)
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0/fps))))
    if args.algo == "energy":
        core = EnergyVAD(hangover_frames=0)   # we apply policy ourselves
    elif args.algo == "zcr":
        core = ZCRVAD(hangover_frames=0)
    else:
        core = ComboVAD(hangover_frames=0)

    # outputs
    clip_csv = clips_dir / f"clip_results_{args.algo}_{args.tag}.csv"
    with open(clip_csv, "w", newline="", encoding="utf-8") as fclip:
        cw = csv.DictWriter(fclip, fieldnames=["source","label","pred"])
        cw.writeheader()

        sw = None
        if args.emit_scores:
            frame_csv = frames_dir / f"frame_scores_{model}_{args.tag}.csv"  # no timestamp (stable per TAG)
            sw = csv.DictWriter(open(frame_csv, "w", newline="", encoding="utf-8"),
                                fieldnames=["model","source","frame_idx","pred_raw","pred_post","score","prob"])
            sw.writeheader()
            print(f"[scores] writing frame scores to: {frame_csv}")

        # timing
        total_audio_sec = 0.0
        wall = []

        for _ in range(max(1,args.repeat)):
            t0 = time.time()
            warned = 0
            for sample in iter_dataset(args.dataset_root):
                source, x, fs, label_clip = coerce_sample_to_tuple4(sample)
                if fs != SR_DEFAULT and warned < 5:
                    print(f"[warn] unexpected sample rate {fs} for {source}; using {SR_DEFAULT}")
                    warned += 1
                fs = SR_DEFAULT

                frames = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))
                E = short_time_energy(frames)
                Z = zero_crossing_rate(frames)

                # raw frame decisions from algorithm
                if isinstance(core, ComboVAD): raw = core.predict_frames(E, Z)
                elif isinstance(core, EnergyVAD): raw = core.predict_frames(E)
                else: raw = core.predict_frames(Z)
                raw = np.asarray(raw, np.int32)

                post = raw
                if args.median3:  post = median3(post)
                if hang_frames>0: post = apply_hangover(post, hang_frames)

                pred_clip = clip_from_policy(post, args.min_speech_frames)
                cw.writerow({"source":source,"label":int(label_clip),"pred":int(pred_clip)})

                if sw is not None:
                    s, p = compute_scores(E, Z, args.algo)
                    for i in range(len(post)):
                        sw.writerow({
                            "model": model, "source": source, "frame_idx": i,
                            "pred_raw": int(raw[i]), "pred_post": int(post[i]),
                            "score": float(s[i]), "prob": float(p[i]),
                        })

                total_audio_sec += len(x)/fs
            wall.append(time.time() - t0)

    # timing CSVs
    audio_hours = total_audio_sec/3600.0 if total_audio_sec>0 else 0.0
    spH_runs = [(w/audio_hours) if audio_hours>0 else 0.0 for w in wall]
    mean_spH = float(np.mean(spH_runs)) if spH_runs else 0.0
    std_spH  = float(np.std(spH_runs, ddof=1)) if len(spH_runs)>1 else 0.0

    timing_csv = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","repeat","wall_time_sec","sec_per_hour_run","audio_hours","date","note"])
        w.writeheader()
        for i,(wt,sph) in enumerate(zip(wall, spH_runs), start=1):
            w.writerow({"model":model,"repeat":i,"wall_time_sec":round(wt,4),
                        "sec_per_hour_run":round(sph,4),"audio_hours":round(audio_hours,6),
                        "date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "note":args.timing_note})
    summary_csv = runtime_dir / f"runtime_summary_{args.tag}.csv"
    new = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats","date","note"])
        if new: w.writeheader()
        w.writerow({"model":model,"sec_per_hour":round(spH_runs[-1],4) if spH_runs else "",
                    "sec_per_hour_mean":round(mean_spH,4),"sec_per_hour_std":round(std_spH,4),
                    "repeats":len(wall),
                    "date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note":args.timing_note})

    print(f"[clips]  {clip_csv}")
    if args.emit_scores: print(f"[frames] {frames_dir}/*.csv")
    print(f"[runtime] {summary_csv}")

if __name__ == "__main__":
    main()
