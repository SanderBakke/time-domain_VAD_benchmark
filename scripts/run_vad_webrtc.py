#!/usr/bin/env python3
# WebRTC VAD (levels 2/3) with SoC framing + min_speech_frames policy.
# Updated: support --soc_params_json + --prefer_soc_params

import argparse, csv, time, datetime, os, sys, json
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

def estimate_webrtc_op_profile(level: int, frame_ms: int) -> dict:
    """
    Coarse per-frame op profile proxy for WebRTC VAD.

    The WebRTC VAD runs on 10/20/30 ms frames. We provide a baseline estimate
    at 10 ms and scale approximately linearly with frame length.
    """
    # Baseline proxy at 10 ms (matches what estimate_power.py previously assumed)
    if level == 2:
        base = {"mul": 4500, "add": 3000, "cmp": 800, "logic": 200}
    elif level == 3:
        base = {"mul": 5000, "add": 3300, "cmp": 1000, "logic": 200}
    else:
        raise ValueError("WebRTC VAD level must be 2 or 3")

    scale = max(1.0, float(frame_ms) / 10.0)
    return {k: int(round(v * scale)) for k, v in base.items()}

def emit_webrtc_op_profile_json(model: str, level: int, frame_ms: int, hop_ms: int,
                               out_base_dir: Path, subdir: str):
    out_dir = out_base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = {
        model: {
            "frame_size": int(round((frame_ms / 1000.0) * SR_DEFAULT)),
            "hop_ms": float(hop_ms),
            "types": estimate_webrtc_op_profile(level, frame_ms),
        }
    }
    out_path = out_dir / f"{model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)
    print(f"[op_profile] saved to: {out_path}")


def now_tag(): 
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dirs(tag, model):
    clips = Path("outputs/clips") / tag
    frames = Path("outputs/frames") / tag / model
    runtime = Path("outputs/runtime") / tag
    for d in (clips, frames, runtime):
        d.mkdir(parents=True, exist_ok=True)
    return clips, frames, runtime

def _pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def apply_hangover(bits, n):
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

def median3(bits):
    if len(bits) < 3:
        return bits.copy()
    y = bits.copy()
    y[1:-1] = ((bits[:-2] + bits[1:-1] + bits[2:]) >= 2).astype(np.int32)
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

def _first_present(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None

def load_soc_params(json_path: str, model_key: str):
    """
    Accepts SoC JSON layouts like:
      { "energy": {"hangover_ms":..., "min_speech_frames":...}, "webrtc_l2": {...}, ... }
    or:
      { "models": { "energy": {...}, "webrtc_l2": {...}, ... } }
    Returns dict or None if not found.
    """
    if not json_path:
        return None
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"--soc_params_json not found: {json_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", data) if isinstance(data, dict) else {}
    if not isinstance(models, dict):
        return None

    # Preferred exact match
    cfg = models.get(model_key, None)
    if isinstance(cfg, dict):
        return cfg

    # Fallbacks (in case you stored webrtc params under a generic key)
    # Try "webrtc" key, then "webrtc_l{level}" derived keys already covered by model_key
    if model_key.startswith("webrtc_l"):
        cfg = models.get("webrtc", None)
        if isinstance(cfg, dict):
            return cfg

    return None

def main():
    ap = argparse.ArgumentParser(description="Run WebRTC VAD")
    ap.add_argument("--dataset_root", default=None, help="Dataset root (required unless --dump_op_profile_only)")
    ap.add_argument("--level", type=int, choices=[2, 3], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emit_scores", action="store_true")

    ap.add_argument("--frame_ms", type=int, default=30)  # WebRTC supports 10/20/30 ms frames
    ap.add_argument("--hop_ms", type=int, default=10)    # standard streaming step

    ap.add_argument("--median3", action="store_true")
    ap.add_argument("--hangover_ms", type=int, default=100)
    ap.add_argument("--min_speech_frames", type=int, default=3)

    ap.add_argument("--soc_params_json", type=str, default="", help="Path to SoC parameter JSON")
    ap.add_argument("--prefer_soc_params", action="store_true",
                    help="If set, override hangover/min_speech_frames using SoC JSON for this model")

    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timing_note", type=str, default="")

    ap.add_argument("--dump_op_profile_only", action="store_true",
                    help="Only emit op profile JSON and exit (no dataset inference)")
    ap.add_argument("--op_profile_out_dir", type=str, default="outputs/op_profiles",
                    help="Base directory for writing op profiles")
    ap.add_argument("--op_profile_tag", type=str, default="",
                    help="Subdirectory name for op profile output. If empty, uses --tag.")

    args = ap.parse_args()

    if (not args.dump_op_profile_only) and (not args.dataset_root):
        raise SystemExit("--dataset_root is required unless --dump_op_profile_only is set")

    model = f"webrtc_l{args.level}"

    # Emit op profile JSON (coarse proxy) before running inference
    base_dir = Path(args.op_profile_out_dir)
    sub = (args.op_profile_tag.strip() or args.tag)
    emit_webrtc_op_profile_json(model=model, level=args.level, frame_ms=args.frame_ms, hop_ms=args.hop_ms,
                               out_base_dir=base_dir, subdir=sub)

    if args.dump_op_profile_only:
        return

    # Optionally override tunable params from SoC JSON
    if args.soc_params_json and args.prefer_soc_params:
        cfg = load_soc_params(args.soc_params_json, model)
        if cfg is None:
            raise KeyError(
                f"Could not find model '{model}' in SoC JSON '{args.soc_params_json}'. "
                f"Expected a top-level key '{model}' or data['models']['{model}']."
            )
        # Accept a few possible names; prefer explicit ones
        hang = _first_present(cfg, ["hangover_ms", "hang_ms", "hangover"])
        minf = _first_present(cfg, ["min_speech_frames", "min_frames", "minf"])

        if hang is not None:
            args.hangover_ms = int(hang)
        if minf is not None:
            args.min_speech_frames = int(minf)

        print(f"[soc] {model}: hangover_ms={args.hangover_ms}, min_speech_frames={args.min_speech_frames}")

    clips_dir, frames_dir, runtime_dir = ensure_dirs(args.tag, model)
    stamp = now_tag()

    vad = webrtcvad.Vad(args.level)
    clip_csv = clips_dir / f"clip_results_{model}_{args.tag}.csv"
    with open(clip_csv, "w", newline="", encoding="utf-8") as fclip:
        cw = csv.DictWriter(fclip, fieldnames=["source", "label", "pred"])
        cw.writeheader()

        sw_fh = None
        sw = None
        if args.emit_scores:
            frame_csv = frames_dir / f"frame_scores_{model}_{args.tag}.csv"
            sw_fh = open(frame_csv, "w", newline="", encoding="utf-8")
            sw = csv.DictWriter(
                sw_fh,
                fieldnames=["model", "source", "frame_idx", "pred_raw", "pred_post", "score", "prob", "label_frame"],
            )
            sw.writeheader()
            print(f"[scores] writing frame scores to: {frame_csv}")

        total_sec = 0.0
        wall = []

        # hop-derived "frames per second" and hangover in frames
        fps = 1000.0 / max(1, args.hop_ms)
        hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))

        def ma(x, k=5):
            if k <= 1:
                return x
            return np.convolve(x, np.ones(k) / k, mode="same")

        try:
            for _ in range(max(1, args.repeat)):
                t0 = time.time()

                for sample in iter_dataset(args.dataset_root):
                    source, x, fs, label_clip = coerce_sample_to_tuple4(sample)
                    fs = SR_DEFAULT  # enforce 16 kHz

                    fr = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))

                    # raw WebRTC 0/1 decisions
                    raw = []
                    for i in range(len(fr)):
                        ok = vad.is_speech(_pcm16(fr[i]), sample_rate=fs)
                        raw.append(int(ok))
                    raw = np.asarray(raw, np.int32)

                    # post-processing
                    post = raw
                    if args.median3:
                        post = median3(post)
                    if hang_frames > 0:
                        post = apply_hangover(post, hang_frames)

                    # clip decision
                    pred = int(np.sum(post) >= int(args.min_speech_frames))

                    # label inferred from path
                    lab = 0 if Path(source).parent.name.lower() in NOISE_DIR_NAMES else 1
                    cw.writerow({"source": source, "label": lab, "pred": pred})

                    # optional frame scores for ROC/PR (proxy confidence)
                    if sw is not None:
                        rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
                        energy_conf = np.clip(rms / 0.1, 0.0, 1.0)
                        energy_conf = ma(energy_conf, k=5)

                        prob = np.clip(0.6 * raw + 0.4 * energy_conf, 0.0, 1.0)

                        for i in range(len(post)):
                            sw.writerow(
                                {
                                    "model": model,
                                    "source": source,
                                    "frame_idx": i,
                                    "pred_raw": int(raw[i]),
                                    "pred_post": int(post[i]),
                                    "score": float(prob[i]),
                                    "prob": float(prob[i]),
                                    "label_frame": int(post[i]),  # no true frame GT; keep consistent with prior script
                                }
                            )

                    total_sec += len(x) / fs

                wall.append(time.time() - t0)

        finally:
            if sw_fh is not None:
                sw_fh.close()

    # timing files
    audio_h = total_sec / 3600.0 if total_sec > 0 else 0.0
    spH = [(w / audio_h) if audio_h > 0 else 0.0 for w in wall]
    mean = np.mean(spH) if spH else 0.0
    std = np.std(spH, ddof=1) if len(spH) > 1 else 0.0

    timing_csv = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "repeat", "wall_time_sec", "sec_per_hour_run", "audio_hours", "date", "note"],
        )
        w.writeheader()
        for i, (wt, sph) in enumerate(zip(wall, spH), start=1):
            w.writerow(
                {
                    "model": model,
                    "repeat": i,
                    "wall_time_sec": round(wt, 4),
                    "sec_per_hour_run": round(sph, 4),
                    "audio_hours": round(audio_h, 6),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": args.timing_note,
                }
            )

    summary = runtime_dir / f"runtime_summary_{args.tag}.csv"
    new = not summary.exists()
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "sec_per_hour", "sec_per_hour_mean", "sec_per_hour_std", "repeats", "date", "note"],
        )
        if new:
            w.writeheader()
        w.writerow(
            {
                "model": model,
                "sec_per_hour": round(spH[-1], 4) if spH else "",
                "sec_per_hour_mean": round(float(mean), 4),
                "sec_per_hour_std": round(float(std), 4),
                "repeats": len(wall),
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": args.timing_note,
            }
        )

    print(f"[clips]  {clip_csv}")
    if args.emit_scores:
        print(f"[frames] {frames_dir}/*.csv")
    print(f"[runtime] {summary}")

if __name__ == "__main__":
    main()
