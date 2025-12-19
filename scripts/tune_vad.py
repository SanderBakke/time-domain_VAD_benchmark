#!/usr/bin/env python3
# scripts/tune_vad.py
#
# Policy-only sweep using saved frame CSVs.
# Tunes two clip-level policy parameters:
#   - min_speech_frames
#   - hangover_ms
#
# Outputs (per TAG):
#   outputs/tuning/<TAG>/tuning_results_low_fp.csv
#   outputs/tuning/<TAG>/tuning_results_low_fn.csv
#   outputs/tuning/<TAG>/tuning_results_balanced.csv
#   outputs/tuning/<TAG>/tuning_results_all.csv
#   outputs/tuning/<TAG>/best_params_<TAG>.json
#
# The JSON stores per-model best params for three regimes:
#   low_fp:    minimize FP (tie-break: maximize recall, then maximize F1, then maximize precision)
#   low_fn:    minimize FN (tie-break: minimize FP, then maximize precision, then maximize F1)
#   balanced:  maximize F1 (tie-break: minimize FP, then maximize recall, then maximize precision)

import argparse, csv, json
from pathlib import Path
import numpy as np


def apply_hangover(bits: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return bits
    y = bits.copy()
    last = -10**9
    for i, v in enumerate(y):
        if v == 1:
            last = i
        elif i - last <= n:
            y[i] = 1
    return y


def confusion(y: np.ndarray, yhat: np.ndarray):
    tn = int(((y == 0) & (yhat == 0)).sum())
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    return prec, rec, f1, acc


def pick_better(profile: str, cand: dict, best: dict | None) -> bool:
    """
    Returns True if cand is better than best under the profile objective.
    cand/best are dicts containing fp, fn, precision, recall, f1.
    """
    if best is None:
        return True

    if profile == "low_fp":
        # primary: minimize FP
        # ties: maximize recall, then maximize f1, then maximize precision
        if cand["fp"] != best["fp"]:
            return cand["fp"] < best["fp"]
        if cand["recall"] != best["recall"]:
            return cand["recall"] > best["recall"]
        if cand["f1"] != best["f1"]:
            return cand["f1"] > best["f1"]
        return cand["precision"] > best["precision"]

    if profile == "low_fn":
        # primary: minimize FN
        # ties: minimize FP, then maximize precision, then maximize f1
        if cand["fn"] != best["fn"]:
            return cand["fn"] < best["fn"]
        if cand["fp"] != best["fp"]:
            return cand["fp"] < best["fp"]
        if cand["precision"] != best["precision"]:
            return cand["precision"] > best["precision"]
        return cand["f1"] > best["f1"]

    # balanced
    # primary: maximize F1
    # ties: minimize FP, then maximize recall, then maximize precision
    if cand["f1"] != best["f1"]:
        return cand["f1"] > best["f1"]
    if cand["fp"] != best["fp"]:
        return cand["fp"] < best["fp"]
    if cand["recall"] != best["recall"]:
        return cand["recall"] > best["recall"]
    return cand["precision"] > best["precision"]


def infer_model_from_frame_csv_name(name_lower: str) -> str:
    # Robust mapping for your naming scheme (including spectral + webrtc).
    if "webrtc_l2" in name_lower: return "webrtc_l2"
    if "webrtc_l3" in name_lower: return "webrtc_l3"
    if "_energy_zcr_" in name_lower: return "energy_zcr"
    if "_energy_logvar_" in name_lower: return "energy_logvar"
    if "_energy_zer_" in name_lower: return "energy_zer"
    if "_energy_" in name_lower: return "energy"
    if "_zcr_" in name_lower: return "zcr"
    if "_zrmse_" in name_lower: return "zrmse"
    if "_nrmse_" in name_lower: return "nrmse"
    if "_logvar_" in name_lower: return "logvar"
    if "_zer_" in name_lower: return "zer"
    if "_par_" in name_lower: return "par"
    if "band_snr" in name_lower: return "band_snr"
    if "lsfm" in name_lower: return "lsfm"
    if "ltsv" in name_lower: return "ltsv"
    if "ltsd" in name_lower: return "ltsd"

    # fallback: parse frame_scores_<MODEL>_<TAG>.csv
    stem = Path(name_lower).stem
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "frame" and parts[1] == "scores":
        model_tokens = parts[2:-1]
        if model_tokens:
            return "_".join(model_tokens).lower()
    return stem.lower()


def hop_ms_for_model(model: str, default_hop_ms: int, webrtc_hop_ms: int) -> int:
    # Explicitly enforce WebRTC's canonical step, while leaving everything else on the default hop.
    if model.startswith("webrtc_"):
        return int(webrtc_hop_ms)
    return int(default_hop_ms)


def main():
    ap = argparse.ArgumentParser(description="Tune clip policy from saved frames (low_fp/low_fn/balanced)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clips_dir", default=None, help="defaults to outputs/clips/<TAG>")
    ap.add_argument("--frames_dir", default=None, help="defaults to outputs/frames/<TAG>")
    ap.add_argument("--hang_grid", default="0,50,100,150,200", help="hangover ms grid")
    ap.add_argument("--min_frames_grid", default="1,2,3,4,5,6,8", help="min speech frames grid")
    ap.add_argument("--hop_ms_default", type=int, default=16, help="hop (ms) for all non-WebRTC models")
    ap.add_argument("--hop_ms_webrtc", type=int, default=10, help="hop (ms) for WebRTC models (webrtc_l2/l3)")
    args = ap.parse_args()

    tag = args.tag
    clips_dir   = Path(args.clips_dir)  if args.clips_dir  else Path("outputs/clips")/tag
    frames_root = Path(args.frames_dir) if args.frames_dir else Path("outputs/frames")/tag
    out_dir = Path("outputs") / "tuning" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

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
        model = infer_model_from_frame_csv_name(fcsv.name.lower())

        with open(fcsv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                src = r["source"]

                # Tuning is policy-only: prefer pred_raw (pre-hangover, pre-median).
                if "pred_raw" in r and r["pred_raw"] != "":
                    bit = int(r["pred_raw"])
                elif "pred_post" in r and r["pred_post"] != "":
                    bit = int(r["pred_post"])
                else:
                    bit = 0

                streams.setdefault(model, {}).setdefault(src, []).append(bit)

    # --- 3) Sweep policies for each model and pick best for each regime
    hang_ms_grid = [int(x) for x in args.hang_grid.split(",") if x.strip() != ""]
    min_grid     = [int(x) for x in args.min_frames_grid.split(",") if x.strip() != ""]

    profiles = ["low_fp", "low_fn", "balanced"]
    best_rows = {p: [] for p in profiles}
    best_json = {}
    all_rows  = []

    for model, by_src in streams.items():
        best_for_model = {p: None for p in profiles}

        hop_ms = hop_ms_for_model(model, args.hop_ms_default, args.hop_ms_webrtc)
        hop_ms = max(1, int(hop_ms))

        for hang_ms in hang_ms_grid:
            # Convert ms -> frames using per-model hop
            hang_frames = max(0, int(round(hang_ms / hop_ms)))

            for mf in min_grid:
                ys = []
                yh = []

                for src, seq in by_src.items():
                    gt = gt_map.get(src, 1)  # fallback speech if missing label (shouldn't happen)
                    seq = np.asarray(seq, np.int32)
                    post = apply_hangover(seq, hang_frames)
                    pred = int(np.sum(post) >= mf)
                    ys.append(gt)
                    yh.append(pred)

                ys = np.array(ys, np.int32)
                yh = np.array(yh, np.int32)

                tp, fp, fn, tn = confusion(ys, yh)
                pr, rc, f1, acc = metrics(tp, fp, fn, tn)

                row = dict(
                    model=model,
                    hop_ms=hop_ms,
                    hang_ms=hang_ms,
                    hang_frames=hang_frames,
                    min_frames=mf,
                    tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=pr, recall=rc, f1=f1, acc=acc,
                )
                all_rows.append(row)

                for prof in profiles:
                    if pick_better(prof, row, best_for_model[prof]):
                        best_for_model[prof] = row

        # finalize for this model
        best_json[model] = {}
        for prof in profiles:
            r = best_for_model[prof]
            if r is None:
                continue

            best_rows[prof].append(
                dict(
                    model=model,
                    hop_ms=r["hop_ms"],
                    hang_ms=r["hang_ms"],
                    hang_frames=r["hang_frames"],
                    min_frames=r["min_frames"],
                    precision=r["precision"],
                    recall=r["recall"],
                    f1=r["f1"],
                    acc=r["acc"],
                    tp=r["tp"], fp=r["fp"], fn=r["fn"], tn=r["tn"],
                )
            )

            best_json[model][prof] = {
                "min_speech_frames": int(r["min_frames"]),
                "hangover_ms": int(r["hang_ms"]),
                "hop_ms": int(r["hop_ms"]),
                "hangover_frames": int(r["hang_frames"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
                "acc": float(r["acc"]),
                "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]), "tn": int(r["tn"]),
            }

    # --- 4) Write outputs
    if all_rows:
        out_all = out_dir / "tuning_results_all.csv"
        with open(out_all, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    for prof in profiles:
        rows = best_rows[prof]
        if not rows:
            continue
        out_csv = out_dir / f"tuning_results_{prof}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    out_json = out_dir / f"best_params_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2)

    print(f"[tune] wrote {out_dir}")
    print(f"[tune] best params: {out_json}")


if __name__ == "__main__":
    main()
