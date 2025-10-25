#!/usr/bin/env python3
# WebRTC VAD (levels 2/3) with SoC framing + min_speech_frames policy.
import argparse, csv, time, datetime, os, sys
from pathlib import Path
import numpy as np
import webrtcvad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vad.datasets import iter_dataset
from vad.features import frame_signal

SR_DEFAULT = 16000
NOISE_DIR_NAMES = {"_background_noise_", "noise", "silence"}

def now_tag(): return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dirs(tag, model):
    clips = Path("outputs/clips")/tag
    frames = Path("outputs/frames")/tag/model
    runtime= Path("outputs/runtime")/tag
    for d in (clips,frames,runtime): d.mkdir(parents=True, exist_ok=True)
    return clips,frames,runtime

def _pcm16(x):
    x = np.clip(x,-1.0,1.0)
    return (x*32767.0).astype(np.int16).tobytes()

def apply_hangover(bits, n):
    if n<=0: return bits
    y=bits.copy(); last=-10**9
    for i,v in enumerate(y):
        if v==1: last=i
        elif i-last<=n: y[i]=1
    return y

def median3(bits):
    if len(bits)<3: return bits.copy()
    y=bits.copy(); y[1:-1]=((bits[:-2]+bits[1:-1]+bits[2:])>=2).astype(np.int32)
    return y

def coerce_sample_to_tuple4(sample):
    if isinstance(sample, (tuple, list)) and len(sample) == 4:
        source, x, fs, label = sample
        return str(source), np.asarray(x, dtype=np.float32), int(fs), int(label)
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        a, b, c = sample
        if isinstance(a, (str, Path)) and not isinstance(b, (str, Path)) and not isinstance(c, (str, Path)):
            source, x, fs = a, b, c
        elif not isinstance(a, (str, Path)) and not isinstance(b, (str, Path)) and isinstance(c, (str, Path)):
            x, fs, source = a, b, c
        else:
            raise ValueError(f"Unrecognized 3-tuple format from dataset: types={[type(v) for v in sample]}")
        source = str(source)
        x = np.asarray(x, dtype=np.float32)
        fs = int(fs)
        parent = Path(source).parent.name.lower()
        label = 0 if parent in NOISE_DIR_NAMES else 1
        return source, x, fs, label
    raise ValueError("Unsupported dataset sample format.")

def main():
    ap = argparse.ArgumentParser(description="Run WebRTC VAD")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--level", type=int, choices=[2,3], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emit_scores", action="store_true")

    ap.add_argument("--frame_ms", type=int, default=32)    # SoC aligned
    ap.add_argument("--hop_ms",   type=int, default=16)

    ap.add_argument("--median3", action="store_true")
    ap.add_argument("--hangover_ms", type=int, default=100)
    ap.add_argument("--min_speech_frames", type=int, default=3)

    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timing_note", type=str, default="")
    args = ap.parse_args()

    model = f"webrtc_l{args.level}"
    clips_dir, frames_dir, runtime_dir = ensure_dirs(args.tag, model)
    stamp = now_tag()

    vad = webrtcvad.Vad(args.level)
    clip_csv = clips_dir / f"clip_results_{model}_{args.tag}.csv"
    cw = csv.DictWriter(open(clip_csv,"w",newline="",encoding="utf-8"),
                        fieldnames=["source","label","pred"])
    cw.writeheader()

    sw=None
    if args.emit_scores:
        frame_csv = frames_dir / f"frame_scores_{model}_{args.tag}.csv"
        sw = csv.DictWriter(open(frame_csv,"w",newline="",encoding="utf-8"),
                            fieldnames=["model","source","frame_idx","pred_raw","pred_post","score","prob","label_frame"])
        sw.writeheader()
        print(f"[scores] writing frame scores to: {frame_csv}")

    total_sec=0.0; wall=[]
    fps = 1000.0/max(1,args.hop_ms)
    hang = max(0, int(round(args.hangover_ms / (1000.0/fps))))

    # simple MA smoother for energy-derived confidence
    def ma(x, k=5):
        if k<=1: return x
        tmp = np.convolve(x, np.ones(k)/k, mode="same")
        return tmp

    for _ in range(max(1,args.repeat)):
        t0=time.time()
        for sample in iter_dataset(args.dataset_root):
            source, x, fs, label_clip = coerce_sample_to_tuple4(sample)
            fs = SR_DEFAULT  # enforce 16 kHz

            fr = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))

            # raw WebRTC 0/1 decisions
            raw=[]
            for i in range(len(fr)):
                ok = vad.is_speech(_pcm16(fr[i]), sample_rate=fs)
                raw.append(int(ok))
            raw = np.asarray(raw, np.int32)

            # post policy for clip decision (unchanged)
            post = raw
            if args.median3: post = median3(post)
            if hang>0:       post = apply_hangover(post, hang)
            pred = int(np.sum(post) >= int(args.min_speech_frames))

            # label inferred from path
            lab = 0 if Path(source).parent.name.lower() in NOISE_DIR_NAMES else 1
            cw.writerow({"source":source,"label":lab,"pred":pred})

            if sw is not None:
                # --- Hybrid confidence for curves ---
                # energy_conf in [0,1]: RMS scaled by 0.1 (≈ -20 dBFS), clipped
                rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
                energy_conf = np.clip(rms / 0.1, 0.0, 1.0)
                energy_conf = ma(energy_conf, k=5)

                # prob mixes binary decision and energy confidence
                prob = np.clip(0.6*raw + 0.4*energy_conf, 0.0, 1.0)

                for i in range(len(post)):
                    sw.writerow({
                        "model":model,"source":source,"frame_idx":i,
                        "pred_raw":int(raw[i]),"pred_post":int(post[i]),
                        # keep score=prob for ROC/PR
                        "score":float(prob[i]),"prob":float(prob[i]),
                        # use post decision as frame label for overlay (no GT per frame)
                        "label_frame":int(post[i]),
                    })
            total_sec += len(x)/fs
        wall.append(time.time()-t0)

    # timing files
    audio_h = total_sec/3600.0 if total_sec>0 else 0.0
    spH = [(w/audio_h) if audio_h>0 else 0.0 for w in wall]
    mean=np.mean(spH) if spH else 0.0; std=np.std(spH, ddof=1) if len(spH)>1 else 0.0

    timing_csv = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(timing_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["model","repeat","wall_time_sec","sec_per_hour_run","audio_hours","date","note"])
        w.writeheader()
        for i,(wt,sph) in enumerate(zip(wall,spH),start=1):
            w.writerow({"model":model,"repeat":i,"wall_time_sec":round(wt,4),
                        "sec_per_hour_run":round(sph,4),"audio_hours":round(audio_h,6),
                        "date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "note":args.timing_note})
    summary = runtime_dir / f"runtime_summary_{args.tag}.csv"
    new = not summary.exists()
    with open(summary,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["model","sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats","date","note"])
        if new: w.writeheader()
        w.writerow({"model":model,"sec_per_hour":round(spH[-1],4) if spH else "",
                    "sec_per_hour_mean":round(float(mean),4),"sec_per_hour_std":round(float(std),4),
                    "repeats":len(wall),
                    "date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note":args.timing_note})

    print(f"[clips]  {clip_csv}")
    if sw is not None: print(f"[frames] {frames_dir}/*.csv")
    print(f"[runtime] {summary}")

if __name__ == "__main__":
    main()
