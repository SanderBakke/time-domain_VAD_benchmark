# #!/usr/bin/env python
# # scripts/run_vad.py
# from __future__ import annotations
# import os, csv, argparse, sys, datetime
# from pathlib import Path
# import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from vad.datasets import iter_dataset           # (x, label, source)
# from vad.features import frame_signal, short_time_energy, zero_crossing_rate
# from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD

# # --- PATH HELPER ---
# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# SR = 16000

# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# def safe_scores_path(scores_csv: str | Path, model: str) -> Path:
#     scores_csv = Path(scores_csv)
#     tag = now_tag()
#     if scores_csv.is_dir() or str(scores_csv).endswith(os.sep):
#         scores_csv.mkdir(parents=True, exist_ok=True)
#         return scores_csv / f"frame_scores_{model}_{tag}.csv"
#     scores_csv.parent.mkdir(parents=True, exist_ok=True)
#     if scores_csv.exists():
#         stem, suf = scores_csv.stem, scores_csv.suffix
#         return scores_csv.with_name(f"{stem}_{model}_{tag}{suf or '.csv'}")
#     if "frame_scores" in scores_csv.name and model not in scores_csv.name:
#         stem, suf = scores_csv.stem, scores_csv.suffix
#         return scores_csv.with_name(f"{stem}_{model}_{tag}{suf or '.csv'}")
#     return scores_csv

# def median3(x: np.ndarray) -> np.ndarray:
#     if len(x) < 3: return x.copy()
#     y = x.copy()
#     y[1:-1] = ((x[:-2] + x[1:-1] + x[2:]) >= 2).astype(np.int32)
#     return y

# def apply_hangover(x: np.ndarray, n: int) -> np.ndarray:
#     if n <= 0 or len(x) == 0: return x
#     y = x.copy()
#     last_on = -10**9
#     for i in range(len(y)):
#         if y[i] == 1:
#             last_on = i
#         else:
#             if i - last_on <= n:
#                 y[i] = 1
#     return y

# def clip_pred_from_policy(frame_decisions: np.ndarray, min_frames: int) -> int:
#     return int(np.sum(frame_decisions) >= min_frames)

# def main():
#     ap = argparse.ArgumentParser(description="Run time-domain VAD (Energy/ZCR/Combo)")
#     ap.add_argument("--algo", choices=["energy","zcr","combo"], required=True)
#     ap.add_argument("--dataset_root", type=str, required=True)
#     ap.add_argument("--out_csv", type=str, default="outputs/clips/clip_results_time.csv")
#     ap.add_argument("--emit_scores", action="store_true")
#     ap.add_argument("--scores_csv", type=str, default="outputs/frames/")
#     ap.add_argument("--frame_ms", type=int, default=25)
#     ap.add_argument("--hop_ms", type=int, default=10)
#     ap.add_argument("--model_name", type=str, default=None)
#     ap.add_argument("--aug", choices=["light", "heavy"], required=True,
#                     help="Which augmentation bucket these scores belong to.")

#     # NEW: clip decision policy (defaults keep old behavior)
#     ap.add_argument("--median3", action="store_true", help="Apply 3-frame median before hangover")
#     ap.add_argument("--hangover_frames", type=int, default=0, help="Keep speech on for N frames after off")
#     ap.add_argument("--min_speech_frames", type=int, default=1, help="Require at least K speech frames per clip")

#     args = ap.parse_args()

#     if args.algo == "energy":
#         vad = EnergyVAD(); model = args.model_name or "Energy"
#     elif args.algo == "zcr":
#         vad = ZCRVAD();    model = args.model_name or "ZCR"
#     else:
#         vad = ComboVAD();  model = args.model_name or "Combo"

#     score_writer = None; score_file = None
#     if args.emit_scores:
#         score_path = safe_scores_path(args.scores_csv, model)
#         score_file = open(score_path, "w", newline="", encoding="utf-8")
#         score_writer = csv.DictWriter(
#             score_file,
#             fieldnames=["model","source","frame_idx","label_frame","score","prob"]
#         )
#         score_writer.writeheader()
#         print(f"[scores] writing frame scores to: {score_path}")

#     rows = []
#     for x, label_clip, source in iter_dataset(args.dataset_root):
#         frames = frame_signal(x, SR, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))
#         E = short_time_energy(frames)
#         Z = zero_crossing_rate(frames)

#         if args.algo == "energy":
#             frame_bin = vad.predict_frames(E)     # 0/1 decisions
#             scores = E.astype(np.float64)         # for curves (uncalibrated)
#         elif args.algo == "zcr":
#             frame_bin = vad.predict_frames(Z)
#             scores = (1.0 - Z).astype(np.float64)
#         else:
#             frame_bin = vad.predict_frames(E, Z)
#             e_min, e_max = np.percentile(E, 5), np.percentile(E, 95)
#             denom = max(1e-12, (e_max - e_min))
#             e_norm = np.clip((E - e_min) / denom, 0.0, 1.0)
#             scores = (e_norm * (1.0 - Z)).astype(np.float64)

#         # Apply clip policy (only affects clip CSV, not scores)
#         dec = frame_bin.astype(np.int32)
#         if args.median3: dec = median3(dec)
#         if args.hangover_frames > 0: dec = apply_hangover(dec, args.hangover_frames)
#         clip_pred = clip_pred_from_policy(dec, args.min_speech_frames)

#         rows.append({"source": source, "label": int(label_clip), "pred": int(clip_pred)})

#         if score_writer is not None:
#             for i, s in enumerate(scores):
#                 score_writer.writerow({
#                     "model": model, "source": source, "frame_idx": i,
#                     "label_frame": int(label_clip), "score": float(s), "prob": ""
#                 })

#     Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
#     with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=["source","label","pred"])
#         w.writeheader(); w.writerows(rows)
#     print(f"\nSaved per-clip results to: {args.out_csv}")

#     if score_writer is not None:
#         score_file.close()
#         print(f"Saved per-frame scores with model='{model}'. (scores unchanged by clip policy)")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# scripts/run_vad.py
# Run time-domain VAD baselines (Energy/ZCR/Combo) with tidy outputs & runtime logs.

import argparse, csv, time, datetime
from pathlib import Path
import numpy as np

# --- your local imports ---
from vad.datasets import iter_dataset               # yields (source, x, fs, label_clip) or (source, x, fs)
from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Normalize dataset sample to (source:str, x:np.ndarray, fs:int, label:int)
from pathlib import Path
import numpy as np

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
    # 4-tuple: trust the order (as in original Speech Commands)
    if isinstance(sample, (tuple, list)) and len(sample) == 4:
        source, x, fs, label = sample
        return str(source), np.asarray(x, dtype=np.float32), int(fs), int(label)

    # 3-tuple: detect where the path is
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        a, b, c = sample
        # Case A: (source, x, fs)
        if isinstance(a, (str, Path)) and not isinstance(b, (str, Path)) and not isinstance(c, (str, Path)):
            source, x, fs = a, b, c
        # Case B: (x, fs, source)
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


FS_MIN = 2000
FS_MAX = 96000
SR_DEFAULT = 16000  # keep one definition per file

_WARN = {"bad_fs": 0}

def _to_scalar_int(v, default=SR_DEFAULT):
    try:
        # handle numpy scalars / arrays / lists cleanly
        return int(np.array(v).reshape(()).item())
    except Exception:
        return int(default)

def _sanitize_fs(fs, source: str, force_fs: int | None):
    if force_fs is not None:
        return int(force_fs)
    fs_val = _to_scalar_int(fs, default=SR_DEFAULT)
    if fs_val < FS_MIN or fs_val > FS_MAX:
        if _WARN["bad_fs"] < 5:
            print(f"[warn] unexpected sample rate {fs_val} for {source}; using {SR_DEFAULT}")
        elif _WARN["bad_fs"] == 5:
            print("[warn] further fs warnings suppressed (using 16000 for remaining files)")
        _WARN["bad_fs"] += 1
        fs_val = SR_DEFAULT
    return fs_val

def _safe_frame_signal(x, fs, frame_ms, hop_ms, source):
    fms = int(round(frame_ms))
    hms = int(round(hop_ms))
    try:
        return frame_signal(x, fs, frame_ms=fms, hop_ms=hms)
    except ValueError as e:
        msg = str(e).lower()
        if "too small" in msg or "frame_ms/hop_ms" in msg:
            # last-resort fallback that always works for 16/32/48 kHz
            if _WARN.get("fallback_framing", 0) < 5:
                print(f"[warn] frame_ms/hop_ms rejected for {source} at fs={fs} "
                      f"({frame_ms}/{hop_ms} ms). Falling back to 30/10 ms.")
            _WARN["fallback_framing"] = _WARN.get("fallback_framing", 0) + 1
            return frame_signal(x, fs, frame_ms=30, hop_ms=10)
        raise


def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------- clip policy helpers ----------
def median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3: return x.copy()
    y = x.copy()
    y[1:-1] = ((x[:-2] + x[1:-1] + x[2:]) >= 2).astype(np.int32)
    return y

def apply_hangover(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(x) == 0: return x
    y = x.copy()
    last_on = -10**9
    for i in range(len(y)):
        if y[i] == 1:
            last_on = i
        else:
            if i - last_on <= n:
                y[i] = 1
    return y

def clip_pred_from_policy(frame_decisions: np.ndarray, min_frames: int) -> int:
    return int(np.sum(frame_decisions) >= min_frames)

# ---------- score helper (for plots) ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def compute_scores(energy, zcr, algo, on_mask=None, k=2.0, gamma=1.0):
    """Turn raw features into a monotonic 'score' + [0..1] 'prob' used for ROC/PR.
       Does not change the model decisions; only for plotting."""
    eps = 1e-12
    n = len(energy)
    if on_mask is None: on_mask = np.zeros(n, dtype=bool)

    # adaptive energy normalisation (dual-rate EMA as crude 'noise floor')
    nf_e = max(eps, np.percentile(energy, 20) * 0.8)
    sE = np.zeros(n, dtype=float)
    for i, e in enumerate(energy):
        a = 0.0125 if on_mask[i] else 0.05
        nf_e = (1.0 - a) * nf_e + a * min(e, nf_e)
        sE[i] = e / max(nf_e, eps)

    zcr_ref = max(eps, np.percentile(zcr, 95))
    sZ = 1.0 - np.clip(zcr / zcr_ref, 0.0, 1.0)

    if algo in ("energy_adaptive", "energy_fixed", "energy"): s = sE
    elif algo == "zcr": s = sZ
    else: s = np.minimum(sE, gamma * sZ)

    p = sigmoid(k * (s - 1.0))
    return {"score": s, "prob": p}

# ---------- model utils ----------
def model_label(algo):
    return {
        "energy":"Energy",
        "zcr":"ZCR",
        "combo":"Combo",
    }.get(algo, algo)

# def ensure_dirs(tag, stamp, model):
#     base = f"{tag}__{stamp}"
#     clips_dir = Path("outputs/clips") / base
#     frames_dir = Path("outputs/frames") / base / model
#     runtime_dir = Path("outputs/runtime") / base
#     clips_dir.mkdir(parents=True, exist_ok=True)
#     frames_dir.mkdir(parents=True, exist_ok=True)
#     runtime_dir.mkdir(parents=True, exist_ok=True)
#     return clips_dir, frames_dir, runtime_dir, base


# # -- AFTER --
# def ensure_dirs(tag, stamp, model):
#     """
#     Folders layout:
#       outputs/
#         clips/<tag>/                                 <- stable across runs (no timestamp)
#         frames/<tag>__<stamp>/<Model>/               <- unique per run
#         runtime/<tag>__<stamp>/                      <- unique per run
#     """
#     base = f"{tag}__{stamp}"
#     clips_dir   = Path("outputs/clips") / tag
#     frames_dir  = Path("outputs/frames") / base / model
#     runtime_dir = Path("outputs/runtime") / base
#     clips_dir.mkdir(parents=True, exist_ok=True)
#     frames_dir.mkdir(parents=True, exist_ok=True)
#     runtime_dir.mkdir(parents=True, exist_ok=True)

#     # optional: write a small pointer file to the latest run (helps evaluation tooling)
#     latest_ptr = clips_dir / "_LATEST.txt"
#     try:
#         latest_ptr.write_text(f"{base}\n", encoding="utf-8")
#     except Exception:
#         pass

#     return clips_dir, frames_dir, runtime_dir, base
def ensure_dirs(tag, stamp, model):
    # Stable folders by TAG; filenames still carry the timestamp
    clips_dir   = Path("outputs/clips")   / tag
    frames_dir  = Path("outputs/frames")  / tag / model
    runtime_dir = Path("outputs/runtime") / tag
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return clips_dir, frames_dir, runtime_dir, tag  # base now equals tag



def main():
    p = argparse.ArgumentParser(description="Run time-domain VAD (Energy/ZCR/Combo)")
    p.add_argument("--algo", choices=["energy","zcr","combo"], required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="dataset/eval tag, e.g. light/heavy")
    p.add_argument("--emit_scores", action="store_true", help="write per-frame scores CSV for ROC/PR")

    # framing
    p.add_argument("--frame_ms", type=int, default=25)
    p.add_argument("--hop_ms", type=int, default=10)

    # clip decision policy
    p.add_argument("--median3", action="store_true")
    p.add_argument("--hangover_ms", type=int, default=0)
    p.add_argument("--min_speech_frames", type=int, default=1)

    # runtime
    p.add_argument("--repeat", type=int, default=1, help="Run the full dataset N times for timing stats")
    p.add_argument("--timing_note", type=str, default="", help="Note to include in runtime CSV")
    # argparse
    p.add_argument("--debug_fs", action="store_true", help="Print sample rate for first 10 files")
    p.add_argument("--force_fs", type=int, default=None, help="Override dataset sample rate (e.g., 16000)")

    

    args = p.parse_args()
    stamp = now_tag()
    model = model_label(args.algo)
    clips_dir, frames_dir, runtime_dir, base = ensure_dirs(args.tag, stamp, model)

    # instantiate correct VAD
    fps = 1000.0 / args.hop_ms
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))
    if args.algo == "energy":
        vad = EnergyVAD(hangover_frames=hang_frames)
    elif args.algo == "zcr":
        vad = ZCRVAD(hangover_frames=hang_frames)
    else:
        vad = ComboVAD(hangover_frames=hang_frames)

    # files
    clip_path = clips_dir / f"clip_results_{model.lower()}_{args.tag}.csv"
    clip_writer = open(clip_path, "w", newline="", encoding="utf-8")
    cw = csv.DictWriter(clip_writer, fieldnames=["source","label","pred"])
    cw.writeheader()

    score_writer = None
    if args.emit_scores:
        frame_path = frames_dir / f"frame_scores_{model}_{args.tag}_{stamp}.csv"
        score_writer = open(frame_path, "w", newline="", encoding="utf-8")
        sw = csv.DictWriter(score_writer, fieldnames=["model","source","frame_idx","label_frame","score","prob"])
        sw.writeheader()
        print(f"[scores] writing frame scores to: {frame_path}")

    # timing accumulators
    total_audio_sec = 0.0
    wall_times = []

    for r in range(max(1, args.repeat)):
        t0 = time.time()
        # iterate the dataset once
        for sample in iter_dataset(args.dataset_root):
            source, x, fs, label_clip = coerce_sample_to_tuple4(sample)          #  fs = int(fs) if fs else SR_DEFAULT
            # frames = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))


            fs = _sanitize_fs(fs, source, args.force_fs)
            frames = _safe_frame_signal(x, fs, args.frame_ms, args.hop_ms, source)

            if args.debug_fs and r == 0:
                if "_dbg_fs_counter" not in globals():
                    global _dbg_fs_counter; _dbg_fs_counter = 0
                if _dbg_fs_counter < 10:
                    print(f"[dbg] {source} fs={fs}")
                    _dbg_fs_counter += 1 
            E = short_time_energy(frames)
            Z = zero_crossing_rate(frames)

            # per-frame decisions (fix: pass the right inputs to each VAD)
            if isinstance(vad, ComboVAD):
                dec = vad.predict_frames(E, Z)
            elif isinstance(vad, EnergyVAD):
                dec = vad.predict_frames(E)
            elif isinstance(vad, ZCRVAD):
                dec = vad.predict_frames(Z)
            else:
                raise RuntimeError("Unknown VAD type")

            dec = dec.astype(np.int32)

            # optional median & hangover for clip decision (does NOT touch scores)
            if args.median3:
                dec = median3(dec)
            if hang_frames > 0:
                dec = apply_hangover(dec, hang_frames)

            pred_clip = clip_pred_from_policy(dec, args.min_speech_frames)
            cw.writerow({"source": source, "label": int(label_clip), "pred": int(pred_clip)})

            # optional frame scores (for ROC/PR)
            if score_writer is not None:
                sc = compute_scores(E, Z, algo=args.algo)
                for i in range(len(dec)):
                    sw.writerow({
                        "model": model,
                        "source": source,
                        "frame_idx": i,
                        "label_frame": int(dec[i]),   # using decisions as frame labels for OP overlay
                        "score": float(sc["score"][i]),
                        "prob": float(sc["prob"][i]),
                    })

            total_audio_sec += len(x) / fs

        wall_times.append(time.time() - t0)

    # close files
    clip_writer.close()
    if score_writer is not None:
        score_writer.close()

    # write per-run timing and summary
    # sec_per_hour = seconds of wall time per 1 hour of audio processed (lower is faster)
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
    
    tag = args.tag
    summary_path = runtime_dir / f"runtime_summary_{tag}__{stamp}.csv"
    # append or create summary for this run folder
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
    if args.emit_scores:
        print(f"[frames] {frames_dir}/*.csv")
    print(f"[runtime] {summary_path}")

if __name__ == "__main__":
    main()
