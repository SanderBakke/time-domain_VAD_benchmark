# #!/usr/bin/env python
# # scripts/run_vad.py
# # Time-domain VAD benchmark (Energy/ZCR/Combo) on Speech Commands v0.02.
# # - Normal mode: saves clip CSV (+ optional frame scores), logs end-to-end sec/hour to outputs/runtime/runtime_summary.csv
# # - Timing-only mode: no outputs written, logs pure processing sec/hour to outputs/runtime_timing/runtime_summary.csv

# from __future__ import annotations
# import os, csv, argparse, sys, datetime
# from pathlib import Path

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import numpy as np

# try:
#     from tqdm import tqdm
#     have_tqdm = True
# except Exception:
#     have_tqdm = False

# from vad.features import frame_signal, short_time_energy, zero_crossing_rate
# from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD
# from vad.datasets import iter_dataset
# from vad.metrics import evaluate_clip_level, Timer


# # ---------------------------
# # CLI
# # ---------------------------
# def parse_args():
#     p = argparse.ArgumentParser(
#         description="Time-domain VAD benchmark (clip-level + optional frame scores)."
#     )
#     p.add_argument("--data_dir", required=True, help="Root of Speech Commands v0.02.")
#     p.add_argument(
#         "--algo",
#         default="energy_adaptive",
#         choices=["energy_fixed", "energy_adaptive", "zcr", "combo"],
#         help="Which VAD to run.",
#     )
#     p.add_argument("--frame_ms", type=float, default=20.0)
#     p.add_argument("--hop_ms", type=float, default=10.0)

#     # Energy/ZCR/Combo knobs
#     p.add_argument("--fixed_threshold", type=float, default=1e-3)
#     p.add_argument("--on_ratio", type=float, default=3.0)
#     p.add_argument("--off_ratio", type=float, default=1.5)
#     p.add_argument("--ema_alpha", type=float, default=0.05)
#     p.add_argument("--zcr_max", type=float, default=0.12)
#     p.add_argument("--hangover_ms", type=float, default=200.0)

#     # Score / probability export (for PR/ROC; disabled in timing_only mode)
#     p.add_argument("--emit_scores", action="store_true", help="Write per-frame score/prob CSV.")
#     p.add_argument("--scores_csv", default="outputs/frames/frame_scores.csv")

#     # Fusion shaping (only used when --emit_scores)
#     p.add_argument("--score_gain", type=float, default=2.0)
#     p.add_argument("--combo_gamma", type=float, default=1.0)

#     # Misc
#     p.add_argument("--max_files", type=int, default=None)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--out_csv", default="outputs/clips/clip_results.csv")

#     # New: timing-only mode (no outputs, just log pure processing speed)
#     p.add_argument("--timing_only", action="store_true",
#                    help="Disable all CSV writes and score computation; log to outputs/runtime_timing.")
#     p.add_argument("--no_progress", action="store_true", help="Disable progress bar (tqdm).")
#     p.add_argument("--timing_note", default="", help="Optional note string added to timing log.")
#     return p.parse_args()


# # ---------------------------
# # Helper for PR/ROC score mapping (only when emit_scores)
# # ---------------------------
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))


# def dual_rate_ema(prev, x, alpha_off=0.05, alpha_on=0.0125, is_on=False):
#     a = float(alpha_on if is_on else alpha_off)
#     return (1.0 - a) * prev + a * x


# def compute_scores(energy, zcr, algo, on_mask=None, k=2.0, gamma=1.0):
#     """
#     Build per-frame 'speech-likeliness' scores and mapped probabilities in [0,1].
#     Only called when --emit_scores is enabled.
#     """
#     eps = 1e-12
#     n = len(energy)
#     if on_mask is None:
#         on_mask = np.zeros(n, dtype=bool)

#     # Energy score: ratio to dynamic noise floor
#     nf_e = max(eps, np.percentile(energy, 20) * 0.8)
#     sE = np.zeros(n, dtype=float)
#     for i, e in enumerate(energy):
#         nf_e = dual_rate_ema(nf_e, e, is_on=on_mask[i])
#         sE[i] = e / max(nf_e, eps)

#     # ZCR score: invert normalized ZCR (low ZCR -> speechy)
#     zcr_ref = max(eps, np.percentile(zcr, 95))
#     sZ = np.zeros(n, dtype=float)
#     for i, z in enumerate(zcr):
#         zn = min(1.0, z / zcr_ref)
#         sZ[i] = 1.0 - zn

#     if algo in ("energy_adaptive", "energy_fixed"):
#         s = sE
#     elif algo == "zcr":
#         s = sZ
#     else:  # combo
#         s = np.minimum(sE, gamma * sZ)

#     p = sigmoid(k * (s - 1.0))
#     return {"score": s, "prob": p}


# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     args = parse_args()

#     # Map to nice names for logs/eval
#     model_name = {
#         "energy_fixed": "Energy-Fixed",
#         "energy_adaptive": "Energy",
#         "zcr": "ZCR",
#         "combo": "Combo",
#     }.get(args.algo, args.algo)

#     # Configure algorithm
#     fps = 1000.0 / args.hop_ms
#     hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))

#     if args.algo == "energy_fixed":
#         vad = EnergyVAD(mode="fixed", fixed_threshold=args.fixed_threshold, hangover_frames=hang_frames)
#     elif args.algo == "energy_adaptive":
#         vad = EnergyVAD(
#             mode="adaptive",
#             on_ratio=args.on_ratio,
#             off_ratio=args.off_ratio,
#             ema_alpha=args.ema_alpha,
#             hangover_frames=hang_frames,
#         )
#     elif args.algo == "zcr":
#         vad = ZCRVAD(zcr_max=args.zcr_max, hangover_frames=hang_frames)
#     elif args.algo == "combo":
#         vad = ComboVAD(
#             on_ratio=args.on_ratio,
#             off_ratio=args.off_ratio,
#             zcr_max=args.zcr_max,
#             ema_alpha=args.ema_alpha,
#             hangover_frames=hang_frames,
#         )
#     else:
#         raise ValueError("Unknown algo")

#     # Optional outputs (disabled in timing-only mode)
#     score_writer = None
#     score_f = None
#     rows = []
#     if not args.timing_only:
#         Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
#         if args.emit_scores:
#             Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
#             score_f = open(args.scores_csv, "w", newline="", encoding="utf-8")
#             score_writer = csv.writer(score_f)
#             score_writer.writerow(["source", "frame_idx", "label_frame", "score", "prob"])

#     # Counters for clip metrics (not used in timing-only mode)
#     y_true, y_pred = [], []

#     # Dataset counters (for logs – optional)
#     n_clips_speech = 0
#     n_clips_noise = 0
#     total_frames_speech = 0
#     total_frames_noise = 0

#     iterator = iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed)
#     if have_tqdm and not args.no_progress:
#         iterator = tqdm(iterator, desc="Processing clips")

#     # Process
#     with Timer() as t_total:
#         for x, label_clip, src in iterator:
#             # Frame into overlapping windows
#             frames = frame_signal(x, sr=16000, frame_ms=args.frame_ms, hop_ms=args.hop_ms)  # (T, L)
#             T = frames.shape[0]

#             # Features
#             energy = short_time_energy(frames)
#             zcr = zero_crossing_rate(frames)

#             # Decisions per frame
#             if isinstance(vad, ComboVAD):
#                 decisions = vad.predict_frames(energy, zcr)
#             elif isinstance(vad, ZCRVAD):
#                 decisions = vad.predict_frames(zcr)
#             else:
#                 decisions = vad.predict_frames(energy)

#             # Clip-level
#             pred_clip = int(decisions.max() > 0)

#             # Dataset counters
#             if label_clip == 1:
#                 n_clips_speech += 1
#                 total_frames_speech += T
#             else:
#                 n_clips_noise += 1
#                 total_frames_noise += T

#             # If not timing-only, collect metrics + outputs
#             if not args.timing_only:
#                 y_true.append(int(label_clip))
#                 y_pred.append(pred_clip)
#                 rows.append({"source": src, "label": int(label_clip), "pred": pred_clip})

#                 # Per-frame scores only when explicitly requested
#                 if score_writer is not None and args.emit_scores:
#                     scores = compute_scores(
#                         energy, zcr, algo=args.algo,
#                         on_mask=decisions.astype(bool),
#                         k=args.score_gain, gamma=args.combo_gamma
#                     )
#                     lbl_frame = int(label_clip)
#                     for fi, (sc, pr) in enumerate(zip(scores["score"], scores["prob"])):
#                         score_writer.writerow([src, fi, lbl_frame, float(sc), float(pr)])

#     # Audio duration proxy (clips ≈ seconds)
#     audio_hours = (n_clips_speech + n_clips_noise) / 3600.0
#     wall_seconds = t_total.dt
#     sec_per_hour = wall_seconds / max(audio_hours, 1e-9)

#     stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if args.timing_only:
#         # -------- TIMING-ONLY LOG --------
#         runtime_csv = Path("outputs/runtime_timing/runtime_summary.csv")
#         runtime_csv.parent.mkdir(parents=True, exist_ok=True)
#         rt_header = ["model", "sec_per_hour", "date", "note"]
#         note = args.timing_note or f"timing_only; frame={args.frame_ms}ms hop={args.hop_ms}ms hang={args.hangover_ms}ms"
#         rt_line = {"model": model_name, "sec_per_hour": f"{sec_per_hour:.2f}", "date": stamp, "note": note}
#         exists = runtime_csv.exists()
#         with open(runtime_csv, "a", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=rt_header)
#             if not exists:
#                 w.writeheader()
#             w.writerow(rt_line)

#         print(f"\n[TIMING-ONLY] {model_name} sec_per_hour: {sec_per_hour:.2f} (processed ~{audio_hours:.2f} h)")
#         print(f"[TIMING-ONLY] Logged to: {runtime_csv}")
#         return

#     # -------- NORMAL MODE: metrics + logs + outputs --------
#     metrics = evaluate_clip_level(y_true, y_pred)
#     print("\n=== Clip-level metrics ===")
#     for k, v in metrics.items():
#         print(f"{k:>10s}: {v:.4f}")
#     print(f"{'sec_per_hour':>10s}: {sec_per_hour:.2f} (wall-clock sec to process 1 hour of audio)")

#     # End-to-end runtime log
#     runtime_csv = Path("outputs/runtime/runtime_summary.csv")
#     runtime_csv.parent.mkdir(parents=True, exist_ok=True)
#     rt_header = ["model", "sec_per_hour", "date", "note"]
#     rt_line = {
#         "model": model_name,
#         "sec_per_hour": f"{sec_per_hour:.2f}",
#         "date": stamp,
#         "note": f"frame={args.frame_ms}ms hop={args.hop_ms}ms hang={args.hangover_ms}ms",
#     }
#     exists = runtime_csv.exists()
#     with open(runtime_csv, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=rt_header)
#         if not exists:
#             w.writeheader()
#         w.writerow(rt_line)

#     # Dataset stats log (optional)
#     dataset_csv = Path("outputs/dataset_stats/dataset_summary.csv")
#     dataset_csv.parent.mkdir(parents=True, exist_ok=True)
#     ds_header = ["model", "num_speech_clips", "num_noise_clips", "total_frames_speech", "total_frames_noise"]
#     ds_line = {
#         "model": model_name,
#         "num_speech_clips": int(n_clips_speech),
#         "num_noise_clips": int(n_clips_noise),
#         "total_frames_speech": int(total_frames_speech),
#         "total_frames_noise": int(total_frames_noise),
#     }
#     ds_exists = dataset_csv.exists()
#     with open(dataset_csv, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=ds_header)
#         if not ds_exists:
#             w.writeheader()
#         w.writerow(ds_line)

#     # Save clip-level CSV
#     with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=["source", "label", "pred"])
#         w.writeheader()
#         for r in rows:
#             w.writerow(r)
#     print(f"\nSaved per-clip results to: {args.out_csv}")

#     if score_writer is not None:
#         score_f.close()
#         print(f"Saved per-frame scores to: {args.scores_csv}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# scripts/run_vad.py
# Time-domain VAD benchmark (Energy/ZCR/Combo) on Speech Commands v0.02.
# Modes:
# - Normal: saves clip CSV (+ optional frame scores), logs end-to-end sec/hour to outputs/runtime/runtime_summary.csv
# - Timing-only: no outputs; logs mean±std sec/hour to outputs/runtime_timing/runtime_summary.csv (supports --repeat N)

from __future__ import annotations
import os, csv, argparse, sys, datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD
from vad.datasets import iter_dataset
from vad.metrics import evaluate_clip_level, Timer


def parse_args():
    p = argparse.ArgumentParser(description="Time-domain VAD benchmark.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--algo", default="energy_adaptive",
                   choices=["energy_fixed","energy_adaptive","zcr","combo"])
    p.add_argument("--frame_ms", type=float, default=20.0)
    p.add_argument("--hop_ms",   type=float, default=10.0)

    # algorithm knobs
    p.add_argument("--fixed_threshold", type=float, default=1e-3)
    p.add_argument("--on_ratio",  type=float, default=3.0)
    p.add_argument("--off_ratio", type=float, default=1.5)
    p.add_argument("--ema_alpha", type=float, default=0.05)
    p.add_argument("--zcr_max",   type=float, default=0.12)
    p.add_argument("--hangover_ms", type=float, default=200.0)

    # optional frame scores (disabled in timing_only)
    p.add_argument("--emit_scores", action="store_true")
    p.add_argument("--scores_csv",  default="outputs/frames/frame_scores.csv")
    p.add_argument("--score_gain",  type=float, default=2.0)
    p.add_argument("--combo_gamma", type=float, default=1.0)

    # misc
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="outputs/clips/clip_results.csv")

    # timing-only
    p.add_argument("--timing_only", action="store_true",
                   help="Disable CSV writes; measure pure processing time.")
    p.add_argument("--repeat", type=int, default=1,
                   help="Timing-only: repeat N runs and report mean±std.")
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--timing_note", default="")
    return p.parse_args()


def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def dual_rate_ema(prev, x, alpha_off=0.05, alpha_on=0.0125, is_on=False):
    a = float(alpha_on if is_on else alpha_off); return (1.0-a)*prev + a*x

def compute_scores(energy, zcr, algo, on_mask=None, k=2.0, gamma=1.0):
    eps = 1e-12
    n = len(energy)
    if on_mask is None: on_mask = np.zeros(n, dtype=bool)

    nf_e = max(eps, np.percentile(energy, 20) * 0.8)
    sE = np.zeros(n, dtype=float)
    for i, e in enumerate(energy):
        nf_e = dual_rate_ema(nf_e, e, is_on=on_mask[i])
        sE[i] = e / max(nf_e, eps)

    zcr_ref = max(eps, np.percentile(zcr, 95))
    sZ = 1.0 - np.clip(zcr / zcr_ref, 0.0, 1.0)

    if algo in ("energy_adaptive", "energy_fixed"): s = sE
    elif algo == "zcr": s = sZ
    else: s = np.minimum(sE, gamma * sZ)

    p = sigmoid(k * (s - 1.0))
    return {"score": s, "prob": p}


def make_vad(args):
    fps = 1000.0 / args.hop_ms
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))
    if args.algo == "energy_fixed":
        return EnergyVAD(mode="fixed", fixed_threshold=args.fixed_threshold, hangover_frames=hang_frames)
    if args.algo == "energy_adaptive":
        return EnergyVAD(mode="adaptive", on_ratio=args.on_ratio, off_ratio=args.off_ratio,
                         ema_alpha=args.ema_alpha, hangover_frames=hang_frames)
    if args.algo == "zcr":
        return ZCRVAD(zcr_max=args.zcr_max, hangover_frames=hang_frames)
    if args.algo == "combo":
        return ComboVAD(on_ratio=args.on_ratio, off_ratio=args.off_ratio, zcr_max=args.zcr_max,
                        ema_alpha=args.ema_alpha, hangover_frames=hang_frames)
    raise ValueError("Unknown algo")


def model_label(algo):
    return {
        "energy_fixed":"Energy-Fixed",
        "energy_adaptive":"Energy",
        "zcr":"ZCR",
        "combo":"Combo",
    }.get(algo, algo)


def run_once(args, timing_only=False, collect_outputs=True):
    """Return (sec_per_hour, metrics_or_None)."""
    vad = make_vad(args)
    iterator = iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed)
    if HAVE_TQDM and not args.no_progress and not timing_only:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Processing clips")

    y_true, y_pred, rows = [], [], []
    n_sp, n_ns = 0, 0
    tot_f_sp, tot_f_ns = 0, 0

    from vad.metrics import Timer
    with Timer() as t_total:
        for x, label_clip, src in iterator:
            frames = frame_signal(x, sr=16000, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
            T = frames.shape[0]
            energy = short_time_energy(frames)
            zcr    = zero_crossing_rate(frames)

            if isinstance(vad, ComboVAD): decisions = vad.predict_frames(energy, zcr)
            elif isinstance(vad, ZCRVAD): decisions = vad.predict_frames(zcr)
            else:                         decisions = vad.predict_frames(energy)

            pred_clip = int(decisions.max() > 0)

            # dataset counters
            if label_clip == 1: n_sp += 1; tot_f_sp += T
            else:               n_ns += 1; tot_f_ns += T

            if collect_outputs:
                y_true.append(int(label_clip)); y_pred.append(pred_clip)
                rows.append({"source": src, "label": int(label_clip), "pred": pred_clip})

                if args.emit_scores and not timing_only:
                    scores = compute_scores(energy, zcr, args.algo, decisions.astype(bool),
                                            k=args.score_gain, gamma=args.combo_gamma)
                    # Write scores outside: caller does it

    audio_hours = (n_sp + n_ns) / 3600.0  # 1 clip ≈ 1 s
    wall_seconds = t_total.dt
    sec_per_hour = wall_seconds / max(audio_hours, 1e-9)

    if not collect_outputs:
        return sec_per_hour, None

    metrics = evaluate_clip_level(y_true, y_pred)
    return sec_per_hour, (metrics, rows, (n_sp, n_ns, tot_f_sp, tot_f_ns))


def main():
    args = parse_args()
    lab = model_label(args.algo)

    # TIMING-ONLY with repeats
    if args.timing_only:
        secs = []
        for i in range(max(1, args.repeat)):
            s, _ = run_once(args, timing_only=True, collect_outputs=False)
            secs.append(s)
        secs = np.asarray(secs, dtype=float)
        mean = float(np.mean(secs)); std = float(np.std(secs, ddof=1)) if len(secs) > 1 else 0.0

        timing_csv = Path("outputs/runtime_timing/runtime_summary.csv")
        timing_csv.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note = args.timing_note or f"timing_only; repeat={args.repeat}; frame={args.frame_ms}ms hop={args.hop_ms}ms hang={args.hangover_ms}ms"
        hdr = ["model","sec_per_hour","sec_per_hour_mean","sec_per_hour_std","repeats","date","note"]
        row = {"model": lab, "sec_per_hour": f"{mean:.2f}",
               "sec_per_hour_mean": f"{mean:.2f}", "sec_per_hour_std": f"{std:.2f}",
               "repeats": int(args.repeat), "date": stamp, "note": note}
        exists = timing_csv.exists()
        with open(timing_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if not exists: w.writeheader()
            w.writerow(row)
        print(f"\n[TIMING-ONLY] {lab}: mean={mean:.2f} s/h, std={std:.2f} (N={args.repeat})")
        print(f"[TIMING-ONLY] Logged to {timing_csv}")
        return

    # NORMAL end-to-end run
    # Optional frame score writer
    score_writer = None
    score_file = None
    if args.emit_scores:
        Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
        score_file = open(args.scores_csv, "w", newline="", encoding="utf-8")
        score_writer = csv.writer(score_file)
        score_writer.writerow(["source","frame_idx","label_frame","score","prob"])

    sec_per_hour, payload = run_once(args, timing_only=False, collect_outputs=True)
    metrics, rows, (n_sp, n_ns, tot_f_sp, tot_f_ns) = payload

    print("\n=== Clip-level metrics ===")
    for k,v in metrics.items(): print(f"{k:>10s}: {v:.4f}")
    print(f"{'sec_per_hour':>10s}: {sec_per_hour:.2f} (wall-clock sec to process 1 hour of audio)")

    # Logs
    runtime_csv = Path("outputs/runtime/runtime_summary.csv")
    runtime_csv.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rt_hdr = ["model","sec_per_hour","date","note"]
    rt_row = {"model": lab, "sec_per_hour": f"{sec_per_hour:.2f}", "date": stamp,
              "note": f"frame={args.frame_ms}ms hop={args.hop_ms}ms hang={args.hangover_ms}ms"}
    exists = runtime_csv.exists()
    with open(runtime_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rt_hdr)
        if not exists: w.writeheader()
        w.writerow(rt_row)

    dataset_csv = Path("outputs/dataset_stats/dataset_summary.csv")
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    ds_hdr = ["model","num_speech_clips","num_noise_clips","total_frames_speech","total_frames_noise"]
    ds_row = {"model": lab, "num_speech_clips": n_sp, "num_noise_clips": n_ns,
              "total_frames_speech": tot_f_sp, "total_frames_noise": tot_f_ns}
    exists2 = dataset_csv.exists()
    with open(dataset_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ds_hdr)
        if not exists2: w.writeheader()
        w.writerow(ds_row)

    # Save clip CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","label","pred"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved per-clip results to: {args.out_csv}")

    if score_writer is not None:
        score_file.close()
        print(f"Saved per-frame scores to: {args.scores_csv}")


if __name__ == "__main__":
    main()
