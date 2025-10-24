#!/usr/bin/env python3
# scripts/run_vad.py
# Run time-domain VAD baselines (Energy/ZCR/Combo) with tidy outputs & runtime logs.

import argparse, csv, time, datetime, os, sys
from pathlib import Path
import numpy as np

# --- make sure local 'vad' package is importable ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vad.datasets import iter_dataset               # yields (source, x, fs, label_clip) or (source, x, fs)
from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD

NOISE_DIR_NAMES = {"_background_noise_", "noise", "silence"}

def coerce_sample_to_tuple4(sample):
    """
    Accepts any of:
      (source, x, fs, label)
      (source, x, fs)
      (x, fs, source)
    and returns (source:str, x:np.ndarray, fs:int, label:int).
    Label is inferred from parent folder name when missing.
    """
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

    raise ValueError(f"Dataset yielded unsupported sample: type={type(sample)} len={len(sample) if hasattr(sample,'__len__') else 'n/a'}")

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------- clip policy helpers ----------
def median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3: return x.astype(np.int32, copy=True)
    y = x.astype(np.int32, copy=True)
    y[1:-1] = ((x[:-2] + x[1:-1] + x[2:]) >= 2).astype(np.int32)
    return y

def apply_hangover(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(x) == 0: return x.astype(np.int32, copy=True)
    y = x.astype(np.int32, copy=True)
    last_on = -10**9
    for i in range(len(y)):
        if y[i] == 1:
            last_on = i
        elif i - last_on <= n:
            y[i] = 1
    return y

def clip_pred_from_policy(frame_decisions: np.ndarray, min_frames: int) -> int:
    return int(np.sum(frame_decisions) >= int(min_frames))

# ---------- score helper (for plots) ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def compute_scores(energy, zcr, algo, on_mask=None, k=2.0, gamma=1.0):
    """Turn raw features into a monotonic 'score' + [0..1] 'prob' (for ROC/PR files only)."""
    eps = 1e-12
    n = len(energy)
    if on_mask is None: on_mask = np.zeros(n, dtype=bool)

    nf_e = max(eps, np.percentile(energy, 20) * 0.8)
    sE = np.zeros(n, dtype=float)
    for i, e in enumerate(energy):
        a = 0.0125 if on_mask[i] else 0.05
        nf_e = (1.0 - a) * nf_e + a * min(e, nf_e)
        sE[i] = e / max(nf_e, eps)

    zcr_ref = max(eps, np.percentile(zcr, 95))
    sZ = 1.0 - np.clip(zcr / zcr_ref, 0.0, 1.0)

    if algo in ("energy_adaptive", "energy_fixed", "energy"):
        s = sE
    elif algo == "zcr":
        s = sZ
    else:
        s = np.minimum(sE, gamma * sZ)

    p = sigmoid(k * (s - 1.0))
    return {"score": s, "prob": p}

def model_label(algo):
    return {"energy":"Energy", "zcr":"ZCR", "combo":"Combo"}.get(algo, algo)

def ensure_dirs(tag: str, model: str):
    clips_dir  = Path("outputs/clips")  / tag
    frames_dir = Path("outputs/frames") / tag / model
    runtime_dir= Path("outputs/runtime")/ tag
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return clips_dir, frames_dir, runtime_dir

def main():
    p = argparse.ArgumentParser(description="Run time-domain VAD (Energy/ZCR/Combo)")
    p.add_argument("--algo", choices=["energy","zcr","combo"], required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="dataset/eval tag, e.g. light/heavy")
    p.add_argument("--emit_scores", action="store_true", help="write per-frame scores CSV for ROC/PR")
    # framing
    p.add_argument("--frame_ms", type=int, default=25)
    p.add_argument("--hop_ms", type=int, default=10)
    # clip policy
    p.add_argument("--median3", action="store_true")
    p.add_argument("--hangover_ms", type=int, default=0)
    p.add_argument("--min_speech_frames", type=int, default=1)
    # timing
    p.add_argument("--repeat", type=int, default=1, help="Run the full dataset N times for timing stats")
    p.add_argument("--timing_note", type=str, default="", help="Note to include in runtime CSV")

    args = p.parse_args()
    stamp = now_tag()
    model = model_label(args.algo)
    clips_dir, frames_dir, runtime_dir = ensure_dirs(args.tag, model)

    # instantiate VAD
    fps = 1000.0 / args.hop_ms
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))
    if args.algo == "energy":
        vad = EnergyVAD(hangover_frames=hang_frames)
    elif args.algo == "zcr":
        vad = ZCRVAD(hangover_frames=hang_frames)
    else:
        vad = ComboVAD(hangover_frames=hang_frames)

    # files (stable filenames; frames overwrite on re-run)
    clip_path = clips_dir / f"clip_results_{model.lower()}_{args.tag}.csv"
    cw = csv.DictWriter(open(clip_path, "w", newline="", encoding="utf-8"),
                        fieldnames=["source","label","pred"])
    cw.writeheader()

    sw = None
    if args.emit_scores:
        frame_path = frames_dir / f"frame_scores_{model}_{args.tag}.csv"
        sw = csv.DictWriter(open(frame_path, "w", newline="", encoding="utf-8"),
                            fieldnames=["model","source","frame_idx","label_frame","score","prob"])
        sw.writeheader()
        print(f"[scores] writing frame scores to: {frame_path}")

    total_audio_sec = 0.0
    wall_times = []

    warn_fs_shown = 0
    for r in range(max(1, args.repeat)):
        t0 = time.time()
        for sample in iter_dataset(args.dataset_root):
            source, x, fs, label_clip = coerce_sample_to_tuple4(sample)

            # frame; guard against tiny/odd fs values from augmentation artifacts
            try:
                frames = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))
            except Exception:
                if warn_fs_shown < 5:
                    print(f"[warn] unexpected sample rate {fs} for {source}; using 16000")
                    warn_fs_shown += 1
                    if warn_fs_shown == 5:
                        print("[warn] further fs warnings suppressed (using 16000 for remaining files)")
                fs = 16000
                frames = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))

            E = short_time_energy(frames)
            Z = zero_crossing_rate(frames)

            if isinstance(vad, ComboVAD):
                dec = vad.predict_frames(E, Z)
            elif isinstance(vad, EnergyVAD):
                dec = vad.predict_frames(E)
            else:
                dec = vad.predict_frames(Z)

            dec = dec.astype(np.int32)

            if args.median3:
                dec = median3(dec)
            if hang_frames > 0:
                dec = apply_hangover(dec, hang_frames)

            pred_clip = clip_pred_from_policy(dec, args.min_speech_frames)
            cw.writerow({"source": source, "label": int(label_clip), "pred": int(pred_clip)})

            if sw is not None:
                sc = compute_scores(E, Z, algo=args.algo)
                for i in range(len(dec)):
                    sw.writerow({
                        "model": model,
                        "source": source,
                        "frame_idx": i,
                        "label_frame": int(dec[i]),   # OP overlay reference
                        "score": float(sc["score"][i]),
                        "prob": float(sc["prob"][i]),
                    })

            total_audio_sec += len(x) / fs

        wall_times.append(time.time() - t0)

    # runtime logs
    audio_hours = total_audio_sec / 3600.0
    sec_per_hour_list = [(wt / audio_hours) if audio_hours > 0 else 0.0 for wt in wall_times]
    mean_spH = float(np.mean(sec_per_hour_list)) if sec_per_hour_list else 0.0
    std_spH  = float(np.std(sec_per_hour_list, ddof=1)) if len(sec_per_hour_list) > 1 else 0.0

    per_model_timing = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(per_model_timing, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","repeat","wall_time_sec","sec_per_hour_run","audio_hours","date","note"])
        w.writeheader()
        for i, (wt, sph) in enumerate(zip(wall_times, sec_per_hour_list), start=1):
            w.writerow({
                "model": model,
                "repeat": i,
                "wall_time_sec": round(wt, 4),
                "sec_per_hour_run": round(sph, 4),
                "audio_hours": round(audio_hours, 6),
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": args.timing_note,
            })

    summary_path = runtime_dir / f"runtime_summary_{args.tag}__{stamp}.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model","sec_per_hour","sec_per_hour_mean","sec_per_hour_std",
            "repeats","date","note"
        ])
        if write_header:
            w.writeheader()
        w.writerow({
            "model": model,
            "sec_per_hour": round(sec_per_hour_list[-1], 4) if sec_per_hour_list else "",
            "sec_per_hour_mean": round(mean_spH, 4),
            "sec_per_hour_std": round(std_spH, 4),
            "repeats": len(wall_times),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": args.timing_note,
        })

    print(f"[clips]  {clip_path}")
    if sw is not None:
        print(f"[frames] {frames_dir}/*.csv")
    print(f"[runtime] {summary_path}")

if __name__ == "__main__":
    main()
