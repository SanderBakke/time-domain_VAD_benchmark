# #!/usr/bin/env python
# # scripts/run_vad_webrtc.py
# # WebRTC VAD benchmark (levels 0–3) on Speech Commands dataset.
# # Supports --timing_only for pure processing benchmarks.

# import argparse, csv, time, datetime
# from pathlib import Path
# import numpy as np
# import webrtcvad

# from vad.datasets import iter_dataset


# def wav_to_int16_bytes(x: np.ndarray) -> bytes:
#     x = np.clip(x, -1.0, 1.0)
#     return (x * 32768.0).astype(np.int16).tobytes()


# def make_frames(x: np.ndarray, fs: int, frame_ms: int, hop_ms: int) -> np.ndarray:
#     """Split signal into overlapping frames (frame_ms, hop_ms)."""
#     frame_len = int(round(fs * frame_ms / 1000))
#     hop = int(round(fs * hop_ms / 1000))
#     if frame_len <= 0 or hop <= 0 or len(x) < frame_len:
#         return np.zeros((0, frame_len), dtype=np.float32)
#     n = 1 + (len(x) - frame_len) // hop
#     out = np.empty((n, frame_len), dtype=np.float32)
#     for i, start in enumerate(range(0, len(x) - frame_len + 1, hop)):
#         out[i] = x[start:start + frame_len]
#     return out


# def smooth_hangover(decisions: np.ndarray, hop_ms: int, hangover_ms: int):
#     """Apply hangover smoothing to binary decisions."""
#     if hangover_ms <= 0 or decisions.size == 0:
#         return decisions
#     hang = int(np.ceil(hangover_ms / hop_ms))
#     out = decisions.copy().astype(np.int32)
#     last_on = -10**9
#     for i, d in enumerate(decisions):
#         if d == 1:
#             last_on = i
#             out[i] = 1
#         elif i - last_on <= hang:
#             out[i] = 1
#     return out.astype(np.int32)


# def main():
#     ap = argparse.ArgumentParser(description="Run WebRTC VAD on Speech Commands.")
#     ap.add_argument("--data_dir", required=True)
#     ap.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
#     ap.add_argument("--frame_ms", type=int, default=20, choices=[10, 20, 30])
#     ap.add_argument("--hop_ms", type=int, default=10)
#     ap.add_argument("--hangover_ms", type=int, default=200)
#     ap.add_argument("--max_files", type=int, default=None)
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--out_csv", type=str, default=None)
#     ap.add_argument("--emit_scores", action="store_true")
#     ap.add_argument("--scores_csv", type=str, default=None)

#     # Timing-only controls
#     ap.add_argument("--timing_only", action="store_true", help="Disable all I/O; measure pure processing time.")
#     ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar.")
#     ap.add_argument("--timing_note", type=str, default="", help="Optional note string for timing logs.")

#     args = ap.parse_args()

#     fs = 16000
#     level = args.aggressiveness
#     model_name = f"webrtc_l{level}"

#     out_clips = Path(args.out_csv) if args.out_csv else Path(f"outputs/clips/clip_results_{model_name}.csv")
#     out_frames = Path(args.scores_csv) if args.scores_csv else Path(f"outputs/frames/frame_scores_{model_name}.csv")

#     if not args.timing_only:
#         out_clips.parent.mkdir(parents=True, exist_ok=True)
#         if args.emit_scores:
#             out_frames.parent.mkdir(parents=True, exist_ok=True)

#     runtime_csv = Path("outputs/runtime/runtime_summary.csv")
#     runtime_timing_csv = Path("outputs/runtime_timing/runtime_summary.csv")
#     dset_csv = Path("outputs/dataset_stats/dataset_summary.csv")

#     for p in [runtime_csv, runtime_timing_csv, dset_csv]:
#         p.parent.mkdir(parents=True, exist_ok=True)

#     vad = webrtcvad.Vad(level)

#     total_sec_audio = 0.0
#     n_clips_speech = n_clips_noise = 0
#     total_frames_speech = total_frames_noise = 0

#     # Optional output setup
#     if not args.timing_only:
#         fc = open(out_clips, "w", newline="", encoding="utf-8")
#         cw = csv.writer(fc)
#         cw.writerow(["source", "label", "pred"])
#         if args.emit_scores:
#             ff = open(out_frames, "w", newline="", encoding="utf-8")
#             fw = csv.writer(ff)
#             fw.writerow(["source", "frame_idx", "label_frame", "score", "prob"])
#     else:
#         cw = fw = None

#     from vad.metrics import Timer
#     from tqdm import tqdm
#     iterator = iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed)
#     if not args.no_progress and not args.timing_only:
#         iterator = tqdm(iterator, desc=f"WebRTC L{level} clips")

#     with Timer() as t_total:
#         for x, label_clip, src in iterator:
#             frames = make_frames(x, fs, args.frame_ms, args.hop_ms)
#             T = frames.shape[0]
#             decisions = np.zeros(T, dtype=np.int32)

#             for i in range(T):
#                 f = frames[i]
#                 decisions[i] = 1 if vad.is_speech(wav_to_int16_bytes(f), fs) else 0

#             decisions = smooth_hangover(decisions, args.hop_ms, args.hangover_ms)
#             pred_clip = int(decisions.any())

#             if label_clip == 1:
#                 n_clips_speech += 1
#                 total_frames_speech += T
#             else:
#                 n_clips_noise += 1
#                 total_frames_noise += T

#             if not args.timing_only:
#                 cw.writerow([src, int(label_clip), pred_clip])
#                 if args.emit_scores:
#                     for i, d in enumerate(decisions):
#                         fw.writerow([src, i, int(label_clip), float(d), float(d)])

#             total_sec_audio += len(x) / fs

#     wall = t_total.dt
#     hours_audio = total_sec_audio / 3600.0
#     sec_per_hour = (wall / hours_audio) if hours_audio > 0 else float("nan")

#     stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     note = args.timing_note or f"frame={args.frame_ms} hop={args.hop_ms} hang={args.hangover_ms}"

#     header = ["model", "sec_per_hour", "date", "note"]
#     line = {"model": model_name, "sec_per_hour": f"{sec_per_hour:.2f}", "date": stamp, "note": note}

#     # Log appropriately
#     target_log = runtime_timing_csv if args.timing_only else runtime_csv
#     exists = target_log.exists()
#     with open(target_log, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=header)
#         if not exists:
#             w.writeheader()
#         w.writerow(line)

#     # Dataset stats (only once per model)
#     if not args.timing_only:
#         dset_exists = dset_csv.exists()
#         ds_header = ["model","num_speech_clips","num_noise_clips","total_frames_speech","total_frames_noise"]
#         ds_line = {
#             "model": model_name,
#             "num_speech_clips": n_clips_speech,
#             "num_noise_clips": n_clips_noise,
#             "total_frames_speech": total_frames_speech,
#             "total_frames_noise": total_frames_noise,
#         }
#         with open(dset_csv, "a", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=ds_header)
#             if not dset_exists:
#                 w.writeheader()
#             w.writerow(ds_line)

#     if not args.timing_only:
#         if args.emit_scores:
#             ff.close()
#         fc.close()

#     print(f"\n[{model_name}] {'TIMING' if args.timing_only else 'FULL'} sec/hour: {sec_per_hour:.2f}")
#     print(f"[{model_name}] Logged to: {target_log}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# scripts/run_vad_webrtc.py
# WebRTC VAD benchmark (levels 0–3). Supports --timing_only and --repeat N.

import argparse, csv, datetime
from pathlib import Path
import numpy as np
import webrtcvad

from vad.datasets import iter_dataset
from vad.metrics import Timer


def wav_to_int16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32768.0).astype(np.int16).tobytes()


def make_frames(x: np.ndarray, fs: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    frame_len = int(round(fs * frame_ms / 1000))
    hop = int(round(fs * hop_ms / 1000))
    if frame_len <= 0 or hop <= 0 or len(x) < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32)
    n = 1 + (len(x) - frame_len) // hop
    out = np.empty((n, frame_len), dtype=np.float32)
    for i, start in enumerate(range(0, len(x) - frame_len + 1, hop)):
        out[i] = x[start:start+frame_len]
    return out


def smooth_hangover(decisions: np.ndarray, hop_ms: int, hangover_ms: int):
    if hangover_ms <= 0 or decisions.size == 0:
        return decisions
    hang = int(np.ceil(hangover_ms / hop_ms))
    out = decisions.copy().astype(np.int32)
    last_on = -10**9
    for i, d in enumerate(decisions):
        if d == 1:
            last_on = i
            out[i] = 1
        elif i - last_on <= hang:
            out[i] = 1
    return out.astype(np.int32)


def model_name(level: int) -> str:
    return f"webrtc_l{level}"


def run_once(args, collect_outputs=True):
    fs = 16000
    vad = webrtcvad.Vad(args.aggressiveness)

    n_sp = n_ns = 0
    tot_sec_audio = 0.0
    rows = []

    iterator = iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed)

    with Timer() as t_total:
        for x, label_clip, src in iterator:
            frames = make_frames(x, fs, args.frame_ms, args.hop_ms)
            T = frames.shape[0]
            decisions = np.zeros(T, dtype=np.int32)
            for i in range(T):
                decisions[i] = 1 if vad.is_speech(wav_to_int16_bytes(frames[i]), fs) else 0
            decisions = smooth_hangover(decisions, args.hop_ms, args.hangover_ms)
            pred_clip = int(decisions.any())

            if label_clip == 1: n_sp += 1
            else:               n_ns += 1

            tot_sec_audio += len(x) / fs
            if collect_outputs:
                rows.append({"source": src, "label": int(label_clip), "pred": pred_clip})

    wall = t_total.dt
    hours_audio = tot_sec_audio / 3600.0
    sec_per_hour = (wall / hours_audio) if hours_audio > 0 else float("nan")
    return sec_per_hour, rows, (n_sp, n_ns)


def main():
    ap = argparse.ArgumentParser(description="WebRTC VAD benchmark.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--aggressiveness", type=int, default=2, choices=[0,1,2,3])
    ap.add_argument("--frame_ms", type=int, default=20, choices=[10,20,30])
    ap.add_argument("--hop_ms", type=int, default=10)
    ap.add_argument("--hangover_ms", type=int, default=200)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default=None)

    # timing-only
    ap.add_argument("--timing_only", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--timing_note", type=str, default="")

    args = ap.parse_args()

    name = model_name(args.aggressiveness)

    if args.timing_only:
        secs = []
        for i in range(max(1, args.repeat)):
            s, _, _ = run_once(args, collect_outputs=False)
            secs.append(s)
        secs = np.asarray(secs, dtype=float)
        mean = float(np.mean(secs)); std = float(np.std(secs, ddof=1)) if len(secs)>1 else 0.0

        timing_csv = Path("outputs/runtime_timing/runtime_summary.csv")
        timing_csv.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note = args.timing_note or f"timing_only; repeat={args.repeat}; frame={args.frame_ms} hop={args.hop_ms} hang={args.hangover_ms}"
        hdr = ["model","sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats","date","note"]
        row = {"model": name, "sec_per_hour": f"{mean:.2f}",
               "sec_per_hour_mean": f"{mean:.2f}", "sec_per_hour_std": f"{std:.2f}",
               "repeats": int(args.repeat), "date": stamp, "note": note}
        exists = timing_csv.exists()
        with open(timing_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if not exists: w.writeheader()
            w.writerow(row)
        print(f"\n[TIMING-ONLY] {name}: mean={mean:.2f} s/h, std={std:.2f} (N={args.repeat})")
        print(f"[TIMING-ONLY] Logged to {timing_csv}")
        return

    # Normal end-to-end
    out_clips = Path(args.out_csv) if args.out_csv else Path(f"outputs/clips/clip_results_{name}.csv")
    out_clips.parent.mkdir(parents=True, exist_ok=True)

    s, rows, (n_sp, n_ns) = run_once(args, collect_outputs=True)

    # runtime log
    runtime_csv = Path("outputs/runtime/runtime_summary.csv")
    runtime_csv.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr = ["model","sec_per_hour","date","note"]
    row = {"model": name, "sec_per_hour": f"{s:.2f}", "date": stamp,
           "note": f"frame={args.frame_ms} hop={args.hop_ms} hang={args.hangover_ms}"}
    exists = runtime_csv.exists()
    with open(runtime_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists: w.writeheader()
        w.writerow(row)

    # dataset stats
    dset_csv = Path("outputs/dataset_stats/dataset_summary.csv")
    dset_csv.parent.mkdir(parents=True, exist_ok=True)
    ds_hdr = ["model","num_speech_clips","num_noise_clips","total_frames_speech","total_frames_noise"]
    ds_row = {"model": name, "num_speech_clips": n_sp, "num_noise_clips": n_ns,
              "total_frames_speech": 0, "total_frames_noise": 0}
    exists2 = dset_csv.exists()
    with open(dset_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ds_hdr)
        if not exists2: w.writeheader()
        w.writerow(ds_row)

    # clip csv
    with open(out_clips, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","label","pred"])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"\n[{name}] FULL sec/hour: {s:.2f}")
    print(f"[{name}] Wrote clip results → {out_clips}")
    print(f"[{name}] Logged to: {runtime_csv}")
    return


if __name__ == "__main__":
    main()
