# #!/usr/bin/env python
# # scripts/run_vad_webrtc.py
# import argparse, csv, datetime, os, sys
# from pathlib import Path
# import numpy as np
# import webrtcvad

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from vad.datasets import iter_dataset
# from vad.features import frame_signal

# SR = 16000

# def now_tag():
#     import datetime as _dt
#     return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

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

# def wav_to_int16_bytes(x: np.ndarray) -> bytes:
#     x = np.clip(x, -1.0, 1.0)
#     x = (x * 32767.0).astype(np.int16)
#     return x.tobytes()

# def median3(x):
#     if len(x) < 3: return x.copy()
#     y = x.copy(); y[1:-1] = ((x[:-2]+x[1:-1]+x[2:])>=2).astype(np.int32); return y
# def apply_hangover(x, n):
#     if n <= 0 or len(x)==0: return x
#     y=x.copy(); last_on=-10**9
#     for i in range(len(y)):
#         if y[i]==1: last_on=i
#         elif i-last_on<=n: y[i]=1
#     return y

# def main():
#     ap = argparse.ArgumentParser(description="Run WebRTC VAD (levels 0–3)")
#     ap.add_argument("--dataset_root", type=str, required=True)
#     ap.add_argument("--level", type=int, default=2, choices=[0,1,2,3])
#     ap.add_argument("--frame_ms", type=int, default=20, choices=[10,20,30])
#     ap.add_argument("--hop_ms", type=int, default=10)
#     ap.add_argument("--out_csv", type=str, default="outputs/clips/clip_results_webrtc_l2.csv")
#     ap.add_argument("--emit_scores", action="store_true")
#     ap.add_argument("--scores_csv", type=str, default="outputs/frames/")

#     # NEW clip policy
#     ap.add_argument("--median3", action="store_true")
#     ap.add_argument("--hangover_frames", type=int, default=0)
#     ap.add_argument("--min_speech_frames", type=int, default=1)
#     args = ap.parse_args()

#     vad = webrtcvad.Vad(args.level)
#     model = f"webrtc_l{args.level}"

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
#         dec = []
#         for i, fr in enumerate(frames):
#             is_speech = 1 if vad.is_speech(wav_to_int16_bytes(fr), SR) else 0
#             dec.append(is_speech)
#             if score_writer is not None:
#                 score_writer.writerow({
#                     "model": model, "source": source, "frame_idx": i,
#                     "label_frame": int(label_clip), "score": float(is_speech), "prob": float(is_speech)
#                 })
#         dec = np.asarray(dec, dtype=np.int32)
#         if args.median3: dec = median3(dec)
#         if args.hangover_frames > 0: dec = apply_hangover(dec, args.hangover_frames)
#         clip_pred = int(np.sum(dec) >= args.min_speech_frames)
#         rows.append({"source": source, "label": int(label_clip), "pred": clip_pred})

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
# scripts/run_vad_webrtc.py
# Run WebRTC VAD (levels 2/3) with unified outputs + frame CSVs for ROC/PR.

import argparse, csv, time, datetime
from pathlib import Path
import numpy as np
import webrtcvad

from vad.datasets import iter_dataset
from vad.features import frame_signal

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

# def ensure_dirs(tag, stamp, model):
#     base = f"{tag}__{stamp}"
#     clips_dir = Path("outputs/clips") / base
#     frames_dir = Path("outputs/frames") / base / model
#     runtime_dir = Path("outputs/runtime") / base
#     clips_dir.mkdir(parents=True, exist_ok=True)
#     frames_dir.mkdir(parents=True, exist_ok=True)
#     runtime_dir.mkdir(parents=True, exist_ok=True)
#     return clips_dir, frames_dir, runtime_dir, base

# -- AFTER --
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



def _pcm16(x):
    # float [-1,1] -> int16 bytes (little-endian)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def main():
    p = argparse.ArgumentParser(description="Run WebRTC VAD")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--level", type=int, choices=[2,3], required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--frame_ms", type=int, default=30, help="WebRTC supported: 10/20/30 ms")
    p.add_argument("--hop_ms", type=int, default=10, help="step")
    p.add_argument("--emit_scores", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timing_note", type=str, default="")
    p.add_argument("--debug_fs", action="store_true", help="Print sample rate for first 10 files")
    p.add_argument("--force_fs", type=int, default=None, help="Override dataset sample rate (e.g., 16000)")

    args = p.parse_args()

    stamp = now_tag()
    model = f"webrtc_l{args.level}"
    clips_dir, frames_dir, runtime_dir, base = ensure_dirs(args.tag, stamp, model)

    vad = webrtcvad.Vad(args.level)
    clip_path = clips_dir / f"clip_results_{model}_{args.tag}.csv"
    cw = csv.DictWriter(open(clip_path, "w", newline="", encoding="utf-8"),
                        fieldnames=["source","label","pred"])
    cw.writeheader()

    sw = None
    if args.emit_scores:
        frame_path = frames_dir / f"frame_scores_{model}_{args.tag}_{stamp}.csv"
        sw = csv.DictWriter(open(frame_path, "w", newline="", encoding="utf-8"),
                            fieldnames=["model","source","frame_idx","label_frame","score","prob"])
        sw.writeheader()
        print(f"[scores] writing frame scores to: {frame_path}")

    total_audio_sec = 0.0
    wall = []

    for r in range(max(1, args.repeat)):
        t0 = time.time()
        for sample in iter_dataset(args.dataset_root):
            source, x, fs, label_clip = coerce_sample_to_tuple4(sample)      #      fs = int(fs) if fs else SR_DEFAULT
            # fr = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))

            fs = _sanitize_fs(fs, source, args.force_fs)
            fr = _safe_frame_signal(x, fs, args.frame_ms, args.hop_ms, source)

            if args.debug_fs and r == 0:
                if "_dbg_fs_counter" not in globals():
                    global _dbg_fs_counter; _dbg_fs_counter = 0
                if _dbg_fs_counter < 10:
                    print(f"[dbg] {source} fs={fs}")
                    _dbg_fs_counter += 1
            pred_frames = []
            for i in range(len(fr)):
                ok = vad.is_speech(_pcm16(fr[i]), sample_rate=fs)
                pred_frames.append(int(ok))
                if sw is not None:
                    # score/prob are discrete (0/1) but fine for comparison overlay
                    sw.writerow({
                        "model": model,
                        "source": source,
                        "frame_idx": i,
                        "label_frame": int(ok),
                        "score": float(ok),
                        "prob": float(ok)
                    })
            pred_clip = int(np.sum(pred_frames) > 0)
            cw.writerow({"source": source, "label": int(label_clip), "pred": pred_clip})
            total_audio_sec += len(x) / fs
        wall.append(time.time() - t0)

    # runtime logs
    audio_hours = total_audio_sec / 3600.0
    spH = [(w / audio_hours) if audio_hours > 0 else 0.0 for w in wall]
    mean_spH = float(np.mean(spH)) if spH else 0.0
    std_spH  = float(np.std(spH, ddof=1)) if len(spH) > 1 else 0.0

    per_model_timing = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(per_model_timing, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","repeat","wall_time_sec","sec_per_hour_run","audio_hours","date","note"])
        w.writeheader()
        for i, (wt, sph) in enumerate(zip(wall, spH), start=1):
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
            "sec_per_hour": round(spH[-1], 4) if spH else "",
            "sec_per_hour_mean": round(mean_spH, 4),
            "sec_per_hour_std": round(std_spH, 4),
            "repeats": len(wall),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": args.timing_note,
        })

    print(f"[clips]  {clip_path}")
    print(f"[runtime] {summary_path}")

if __name__ == "__main__":
    main()
