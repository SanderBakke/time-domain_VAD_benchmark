#!/usr/bin/env python

"""
Grid/Random search for time-domain VAD parameters.
(Place this file in: scripts/grid_search_vad.py)
"""
import argparse
import csv
import itertools
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from vad.features import frame_signal, short_time_energy, zero_crossing_rate
from vad.algorithms import EnergyVAD, ZCRVAD, ComboVAD
from vad.datasets import iter_dataset

def parse_args():
    p = argparse.ArgumentParser(description="Grid/Random search for VAD parameters.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--algo", type=str, required=True, choices=["energy_adaptive", "zcr", "combo"])
    p.add_argument("--frame_ms", type=float, default=20.0)
    p.add_argument("--hop_ms", type=float, default=10.0)

    # Grid sets
    p.add_argument("--on_ratio", type=float, nargs="*", default=None)
    p.add_argument("--off_ratio", type=float, nargs="*", default=None)
    p.add_argument("--zcr_max", type=float, nargs="*", default=None)
    p.add_argument("--ema_alpha", type=float, nargs="*", default=None)
    p.add_argument("--hangover_ms", type=float, nargs="*", default=None)

    # Random search
    p.add_argument("--rand_trials", type=int, default=0)
    p.add_argument("--on_ratio_range", type=float, nargs=2, default=None)
    p.add_argument("--off_ratio_range", type=float, nargs=2, default=None)
    p.add_argument("--zcr_max_range", type=float, nargs=2, default=None)
    p.add_argument("--ema_alpha_range", type=float, nargs=2, default=None)
    p.add_argument("--hangover_ms_range", type=float, nargs=2, default=None)

    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=str, default="outputs/tuning/tuning_results.csv")
    return p.parse_args()

def sec_per_hour(num_clips, wall_seconds):
    audio_hours = num_clips / 3600.0  # ~1s per clip
    return wall_seconds / max(audio_hours, 1e-9)

def grid_trials(args):
    def ensure(values, default):
        return values if values else default
    if args.algo == "energy_adaptive":
        on = ensure(args.on_ratio, [2.5, 3.0, 3.5])
        off = ensure(args.off_ratio, [1.3, 1.5, 1.7])
        ema = ensure(args.ema_alpha, [0.03, 0.05, 0.08])
        hang = ensure(args.hangover_ms, [150, 200, 250])
        for a, b, c, d in itertools.product(on, off, ema, hang):
            yield dict(on_ratio=a, off_ratio=b, ema_alpha=c, zcr_max=None, hangover_ms=d)
    elif args.algo == "zcr":
        z = ensure(args.zcr_max, [0.10, 0.12, 0.14, 0.16])
        hang = ensure(args.hangover_ms, [150, 200, 250])
        for zc, d in itertools.product(z, hang):
            yield dict(on_ratio=None, off_ratio=None, ema_alpha=None, zcr_max=zc, hangover_ms=d)
    else:
        on = ensure(args.on_ratio, [2.5, 3.0, 3.5])
        off = ensure(args.off_ratio, [1.3, 1.5, 1.7])
        z = ensure(args.zcr_max, [0.10, 0.12, 0.14])
        ema = ensure(args.ema_alpha, [0.03, 0.05, 0.08])
        hang = ensure(args.hangover_ms, [150, 200, 250])
        for a, b, c, d, e in itertools.product(on, off, z, ema, hang):
            yield dict(on_ratio=a, off_ratio=b, zcr_max=c, ema_alpha=d, hangover_ms=e)

def random_trials(args):
    rng = random.Random(args.seed)
    for _ in range(args.rand_trials):
        if args.algo == "energy_adaptive":
            yield dict(
                on_ratio=rng.uniform(*(args.on_ratio_range or (2.2, 3.8))),
                off_ratio=rng.uniform(*(args.off_ratio_range or (1.2, 1.8))),
                ema_alpha=rng.uniform(*(args.ema_alpha_range or (0.02, 0.10))),
                zcr_max=None,
                hangover_ms=rng.uniform(*(args.hangover_ms_range or (120, 280))),
            )
        elif args.algo == "zcr":
            yield dict(
                on_ratio=None, off_ratio=None, ema_alpha=None,
                zcr_max=rng.uniform(*(args.zcr_max_range or (0.08, 0.18))),
                hangover_ms=rng.uniform(*(args.hangover_ms_range or (120, 280))),
            )
        else:
            yield dict(
                on_ratio=rng.uniform(*(args.on_ratio_range or (2.2, 3.8))),
                off_ratio=rng.uniform(*(args.off_ratio_range or (1.2, 1.8))),
                zcr_max=rng.uniform(*(args.zcr_max_range or (0.08, 0.18))),
                ema_alpha=rng.uniform(*(args.ema_alpha_range or (0.02, 0.10))),
                hangover_ms=rng.uniform(*(args.hangover_ms_range or (120, 280))),
            )

def run_trial(args, params):
    # convert hangover to frames
    fps = 1000.0 / args.hop_ms
    hang_frames = max(0, int(round(params["hangover_ms"] / (1000.0 / fps))))

    if args.algo == "energy_adaptive":
        vad = EnergyVAD(mode="adaptive",
                        on_ratio=float(params["on_ratio"]),
                        off_ratio=float(params["off_ratio"]),
                        ema_alpha=float(params["ema_alpha"]),
                        hangover_frames=hang_frames)
    elif args.algo == "zcr":
        vad = ZCRVAD(zcr_max=float(params["zcr_max"]),
                     hangover_frames=hang_frames)
    else:
        vad = ComboVAD(on_ratio=float(params["on_ratio"]),
                       off_ratio=float(params["off_ratio"]),
                       zcr_max=float(params["zcr_max"]),
                       ema_alpha=float(params["ema_alpha"]),
                       hangover_frames=hang_frames)

    y_true, y_pred = [], []
    n_clips = 0
    t0 = time.perf_counter()
    for x, label, src in iter_dataset(args.data_dir, max_files=args.max_files, seed=args.seed):
        frames = frame_signal(x, sr=16000, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
        e = short_time_energy(frames)
        z = zero_crossing_rate(frames)
        if isinstance(vad, ComboVAD):
            decisions = vad.predict_frames(e, z)
        elif isinstance(vad, ZCRVAD):
            decisions = vad.predict_frames(z)
        else:
            decisions = vad.predict_frames(e)
        pred = int(decisions.max() > 0)
        y_true.append(label); y_pred.append(pred)
        n_clips += 1
    wall = time.perf_counter() - t0

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                sec_per_hour=(wall / max(n_clips / 3600.0, 1e-9)),
                n_clips=n_clips)

def main():
    args = parse_args()
    trials = list(random_trials(args)) if args.rand_trials > 0 else list(grid_trials(args))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Total trials: {len(trials)}")
    fieldnames = ["algo","frame_ms","hop_ms","on_ratio","off_ratio","zcr_max","ema_alpha","hangover_ms",
                  "accuracy","precision","recall","f1","sec_per_hour","n_clips"]

    results = []
    for i, t in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {t}")
        res = run_trial(args, t)
        row = dict(algo=args.algo, frame_ms=args.frame_ms, hop_ms=args.hop_ms, **t, **res)
        results.append(row)

    results.sort(key=lambda r: (-r["f1"], r["sec_per_hour"]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)

    if results:
        best = results[0]
        print("\n=== Best by F1 ===")
        for k in fieldnames:
            print(f"{k}: {best.get(k, '')}")
        print(f"\nSaved all trials to: {out_csv}")
    else:
        print("No trials executed.")

if __name__ == "__main__":
    main()
