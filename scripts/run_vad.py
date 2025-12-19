#!/usr/bin/env python3
# scripts/run_vad.py

import argparse, csv, time, datetime, os, sys, json, math
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vad.datasets import iter_dataset
from vad.features import (
    frame_signal,
    short_time_energy,
    zero_crossing_rate,
    rms_energy,
    zero_energy_ratio,
    log_energy_variance,
    stft_power,
    band_energies,
    spectral_flatness,
)
from vad.algorithms import (
    EnergyVAD, ZCRVAD, EnergyZCRVAD, EnergyLogVarVAD, EnergyZERVAD,
    ZRMSEVAD, NRMSEVAD, ZERVAD, PARVAD, LogVarVAD,
    BandSNRVAD, LSFMVAD, LTSVVAD, LTSDVAD,
)

SR_DEFAULT = 16000

ALL_ALGOS = [
    "energy", "zcr", "energy_zcr", "energy_logvar", "energy_zer",
    "zrmse", "nrmse", "zer", "par", "logvar",
    "band_snr", "lsfm", "ltsv", "ltsd",
]
SPECTRAL_ALGOS = {"band_snr", "lsfm", "ltsv", "ltsd"}

SOC_PROFILES = ["low_fp", "low_fn", "balanced", "low_latency"]


def estimate_op_profile(algo: str, frame_size: int, n_fft: int = 256, n_bands: int = 16):
    """
    Coarse per-frame op profile for power proxying.

    - Uses 'mac' for FFT-equivalent cost (SoC-friendly proxy).
    - Uses 'mul/add/cmp/logic/sqrt/sigmoid' for scalar computations.
    - Excludes Python overhead, file I/O, and clip-level aggregation/postprocessing.
    """
    N = int(frame_size)
    NFFT = int(n_fft)
    B = int(n_bands)

    W_LOGVAR = 5
    W_LSFM = 8
    W_LTSV = 8

    def rfft_mac(nfft: int) -> int:
        return int(round(2.5 * nfft * math.log2(max(nfft, 2))))

    if algo == "energy":
        return {"mul": N, "add": N - 1, "cmp": 1}

    if algo == "zcr":
        return {"cmp": N - 1, "logic": N - 1, "add": max(N - 2, 0)}

    if algo == "energy_zcr":
        return {"mul": N, "add": (N - 1) + max(N - 2, 0), "cmp": (N - 1) + 3, "logic": N - 1}

    if algo == "zrmse":
        return {"mul": N, "add": (N - 1) + max(N - 2, 0), "cmp": (N - 1) + 1, "sqrt": 1, "sigmoid": 1}

    if algo == "zer":
        return {"cmp": N + 1, "add": max(N - 1, 0)}

    if algo == "par":
        return {"mul": N, "add": max(N - 1, 0), "cmp": (N - 1) + 1, "sqrt": 1}

    if algo == "logvar":
        return {"mul": N + W_LOGVAR, "add": (N - 1) + (W_LOGVAR - 1) + W_LOGVAR, "cmp": 1}

    if algo == "nrmse":
        return {"mul": N + 6, "add": (N - 1) + 6, "cmp": 4, "sqrt": 2, "logic": 3}

    if algo == "energy_zer":
        return {"mul": N, "add": (N - 1) + (N - 1), "cmp": N + 3, "logic": 3}

    if algo == "energy_logvar":
        return {"mul": N + W_LOGVAR, "add": (N - 1) + (W_LOGVAR - 1) + W_LOGVAR, "cmp": 2, "logic": 2}

    if algo in ("band_snr", "lsfm", "ltsv", "ltsd"):
        ops = {"mac": rfft_mac(NFFT)}
        K = NFFT // 2 + 1
        ops["mul"] = 2 * K
        ops["add"] = 1 * K
        ops["add"] += K  # band accumulation roughness

        if algo == "band_snr":
            ops["mul"] += 2 * B
            ops["add"] += 2 * B
            ops["cmp"] = B + 1
            return ops

        if algo == "lsfm":
            ops["mul"] += 2 * B
            ops["add"] += 3 * B + B * (W_LSFM - 1)
            ops["cmp"] = 1
            return ops

        if algo == "ltsv":
            ops["mul"] += B * W_LTSV
            ops["add"] += B * (2 * W_LTSV)
            ops["cmp"] = 1
            return ops

        if algo == "ltsd":
            ops["mul"] += 2 * K
            ops["add"] += 1 * K
            ops["cmp"] = 1
            return ops

    raise ValueError(f"Unsupported algo '{algo}' for op profile.")


def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def median3(bits: np.ndarray) -> np.ndarray:
    if len(bits) < 3:
        return bits.copy()
    y = bits.copy()
    y[1:-1] = ((bits[:-2] + bits[1:-1] + bits[2:]) >= 2).astype(np.int32)
    return y


def apply_hangover(bits: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return bits
    y = bits.copy()
    last_on = -10**9
    for i, v in enumerate(y):
        if v == 1:
            last_on = i
        else:
            if i - last_on <= n:
                y[i] = 1
    return y


def clip_from_policy(bits: np.ndarray, min_frames: int) -> int:
    return int(np.sum(bits) >= int(min_frames))


def model_label(algo: str) -> str:
    return algo.upper()


def ensure_dirs(tag: str, model: str):
    clips_dir = Path("outputs/clips") / tag
    frames_dir = Path("outputs/frames") / tag / model
    runtime_dir = Path("outputs/runtime") / tag
    for d in (clips_dir, frames_dir, runtime_dir):
        d.mkdir(parents=True, exist_ok=True)
    return clips_dir, frames_dir, runtime_dir


def score_to_prob_generic(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if score.size == 0:
        return score
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)))
    scale = 1.4826 * mad + 1e-6
    z = (score - med) / scale
    z = np.clip(z, -20.0, 20.0)  # avoid exp overflow
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def compute_scores_time_domain(E, Z, ZER, LOGVAR, algo, k=2.0):
    """
    Produces a real-valued 'score' (higher => more speech-like) plus a logistic 'prob' proxy.
    This is for tuning/plots; it does not affect hard decisions.
    """
    eps = 1e-12

    # Energy ratio vs adaptive lower-envelope estimate
    nf_e = max(eps, float(np.percentile(E, 20)) * 0.8)
    sE = np.zeros_like(E, dtype=np.float32)
    a = 0.05
    for i, e in enumerate(E):
        nf_e = (1 - a) * nf_e + a * min(float(e), nf_e)
        sE[i] = float(e) / max(nf_e, eps)

    # ZCR score: lower ZCR => more speech-like (voiced); normalize by high percentile
    zref = max(eps, float(np.percentile(Z, 95)))
    sZ = (1.0 - np.clip(Z / zref, 0.0, 1.0)).astype(np.float32)

    # ZER score: lower ZER => more speech-like
    if ZER is not None:
        zer_ref = max(eps, float(np.percentile(ZER, 95)))
        sZER = (1.0 - np.clip(ZER / zer_ref, 0.0, 1.0)).astype(np.float32)
    else:
        sZER = None

    # LOGVAR score: higher => more speech-like; normalize by percentile
    if LOGVAR is not None:
        lv_ref = max(eps, float(np.percentile(LOGVAR, 95)))
        sLV = np.clip(LOGVAR / lv_ref, 0.0, 2.0).astype(np.float32)
    else:
        sLV = None

    if algo == "energy":
        s = sE
    elif algo == "zcr":
        s = sZ
    elif algo == "energy_zcr":
        s = np.minimum(sE, sZ)
    elif algo == "energy_zer":
        s = np.minimum(sE, sZER if sZER is not None else sE)
    elif algo == "energy_logvar":
        s = np.minimum(sE, sLV if sLV is not None else sE)
    else:
        # generic fallback for other time-domain models
        s = sE

    p = 1.0 / (1.0 + np.exp(-k * (s - 1.0)))
    return s.astype(np.float32), p.astype(np.float32)


def load_soc_policy(soc_params_json: str | None, algo: str):
    if not soc_params_json:
        return None
    p = Path(soc_params_json)
    if not p.exists():
        raise FileNotFoundError(f"SoC params JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    key = str(algo).lower()
    if key not in j:
        return None
    d = j[key]
    if not isinstance(d, dict):
        return None
    mf = int(d.get("min_speech_frames", 0))
    ho = int(d.get("hangover_ms", 0))
    return {"min_speech_frames": mf, "hangover_ms": ho}


def main():
    ap = argparse.ArgumentParser(description="Run VAD Models (time + spectral).")
    ap.add_argument("--algo", choices=ALL_ALGOS, required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emit_scores", action="store_true")

    ap.add_argument("--frame_ms", type=int, default=32)
    ap.add_argument("--hop_ms", type=int, default=16)

    ap.add_argument("--median3", action="store_true")

    ap.add_argument("--hangover_ms", type=int, default=100)
    ap.add_argument("--min_speech_frames", type=int, default=3)

    ap.add_argument("--soc_params_json", type=str, default=None)
    ap.add_argument("--profile", choices=SOC_PROFILES, default="balanced")
    ap.add_argument("--prefer_soc_params", action="store_true")

    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timing_note", type=str, default="")
    ap.add_argument("--dump_op_profile_only", action="store_true")

    ap.add_argument("--n_fft", type=int, default=256)
    ap.add_argument("--n_bands", type=int, default=16)

    args = ap.parse_args()

    soc_pol = load_soc_policy(args.soc_params_json, args.algo)
    if soc_pol is not None:
        if args.prefer_soc_params:
            args.min_speech_frames = int(soc_pol["min_speech_frames"])
            args.hangover_ms = int(soc_pol["hangover_ms"])
        else:
            if int(args.min_speech_frames) <= 0:
                args.min_speech_frames = int(soc_pol["min_speech_frames"])
            if int(args.hangover_ms) < 0:
                args.hangover_ms = int(soc_pol["hangover_ms"])
        print(f"[policy] profile={args.profile} model={args.algo} "
              f"min_speech_frames={args.min_speech_frames} hangover_ms={args.hangover_ms} "
              f"(soc_params_json={args.soc_params_json}, prefer_soc_params={args.prefer_soc_params})")

    model = model_label(args.algo)
    stamp = now_tag()

    frame_size = int((args.frame_ms / 1000.0) * SR_DEFAULT)
    op_profile = {
        model.lower(): {
            "frame_size": frame_size,
            "hop_ms": args.hop_ms,
            "types": estimate_op_profile(args.algo, frame_size, n_fft=args.n_fft, n_bands=args.n_bands),
        }
    }
    out_dir = Path("outputs/op_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(op_profile, f, indent=2)
    print(f"[op_profile] saved to: {out_path}")

    if args.dump_op_profile_only:
        return

    clips_dir, frames_dir, runtime_dir = ensure_dirs(args.tag, model)

    fps = 1000.0 / max(1, args.hop_ms)
    hang_frames = max(0, int(round(args.hangover_ms / (1000.0 / fps))))

    # Instantiate core with hangover disabled (post-policy handles hangover)
    if args.algo == "energy":
        core = EnergyVAD(hangover_frames=0)
    elif args.algo == "zcr":
        core = ZCRVAD(hangover_frames=0)
    elif args.algo == "energy_zcr":
        core = EnergyZCRVAD(hangover_frames=0)
    elif args.algo == "energy_zer":
        core = EnergyZERVAD(hangover_frames=0)
    elif args.algo == "energy_logvar":
        core = EnergyLogVarVAD(hangover_frames=0)
    elif args.algo == "zer":
        core = ZERVAD(hangover_frames=0)
    elif args.algo == "par":
        core = PARVAD(hangover_frames=0)
    elif args.algo == "logvar":
        core = LogVarVAD(hangover_frames=0)
    elif args.algo == "zrmse":
        core = ZRMSEVAD(hangover_frames=0)
    elif args.algo == "nrmse":
        core = NRMSEVAD(hangover_frames=0)
    elif args.algo == "band_snr":
        core = BandSNRVAD(hangover_frames=0)
    elif args.algo == "lsfm":
        core = LSFMVAD(hangover_frames=0)
    elif args.algo == "ltsv":
        core = LTSVVAD(hangover_frames=0)
    elif args.algo == "ltsd":
        core = LTSDVAD(hangover_frames=0)
    else:
        raise ValueError(f"Unsupported VAD model '{args.algo}'")

    clip_csv = clips_dir / f"clip_results_{args.algo}_{args.tag}.csv"
    frame_csv = frames_dir / f"frame_scores_{model}_{args.tag}.csv"

    with open(clip_csv, "w", newline="", encoding="utf-8") as fclip:
        cw = csv.DictWriter(fclip, fieldnames=["source", "label", "pred"])
        cw.writeheader()

        sw_fh = None
        sw = None
        if args.emit_scores:
            sw_fh = open(frame_csv, "w", newline="", encoding="utf-8")
            sw = csv.DictWriter(
                sw_fh,
                fieldnames=["model", "source", "frame_idx", "pred_raw", "pred_post", "score", "prob"],
            )
            sw.writeheader()
            print(f"[scores] writing frame scores to: {frame_csv}")

        total_audio_sec = 0.0
        wall = []

        try:
            for _ in range(max(1, args.repeat)):
                t0 = time.time()

                for sample in iter_dataset(args.dataset_root):
                    source, x, fs, label_clip = sample
                    x = np.asarray(x, dtype=np.float32)
                    fs = int(fs)
                    label_clip = int(label_clip)

                    if fs != SR_DEFAULT:
                        raise ValueError(f"Expected {SR_DEFAULT} Hz, got {fs} for {source}")

                    frames = frame_signal(x, fs, frame_ms=float(args.frame_ms), hop_ms=float(args.hop_ms))

                    E = short_time_energy(frames)
                    Z = zero_crossing_rate(frames)
                    R = rms_energy(frames)

                    ZER = None
                    LOGVAR = None

                    score = None

                    # --- run model ---
                    if args.algo in SPECTRAL_ALGOS:
                        P = stft_power(frames, n_fft=args.n_fft)
                        Eb = band_energies(P, n_bands=args.n_bands)

                        if args.algo == "band_snr":
                            raw = core.predict_frames(Eb)
                            score = getattr(core, "last_statistic", None)

                        elif args.algo == "lsfm":
                            sfm = spectral_flatness(P)
                            raw = core.predict_frames(sfm)
                            score = getattr(core, "last_statistic", None)

                        elif args.algo == "ltsv":
                            raw = core.predict_frames(Eb)
                            score = getattr(core, "last_statistic", None)

                        elif args.algo == "ltsd":
                            raw = core.predict_frames(P)
                            score = getattr(core, "last_statistic", None)

                        else:
                            raise ValueError("Internal error: spectral algo set but not matched.")

                    else:
                        if args.algo == "energy":
                            raw = core.predict_frames(E)

                        elif args.algo == "zcr":
                            raw = core.predict_frames(Z)

                        elif args.algo == "energy_zcr":
                            raw = core.predict_frames(E, Z)

                        elif args.algo == "zrmse":
                            raw = core.predict_frames(E, Z)

                        elif args.algo == "nrmse":
                            raw = core.predict_frames(R)

                        elif args.algo in ("zer", "par"):
                            raw = core.predict_frames(frames)

                        elif args.algo == "logvar":
                            raw = core.predict_frames(E)
                            LOGVAR = log_energy_variance(E, window=getattr(core, "window", 5))

                        elif args.algo == "energy_zer":
                            ZER = zero_energy_ratio(frames, eps=0.02)
                            raw = core.predict_frames(E, ZER)

                        elif args.algo == "energy_logvar":
                            LOGVAR = log_energy_variance(E, window=5)
                            raw = core.predict_frames(E, LOGVAR)

                        else:
                            raise ValueError(f"Unknown time-domain VAD core model: {args.algo}")

                    raw = np.asarray(raw, np.int32)

                    # --- post processing policy ---
                    post = raw
                    if args.median3:
                        post = median3(post)
                    if hang_frames > 0:
                        post = apply_hangover(post, hang_frames)

                    pred_clip = clip_from_policy(post, args.min_speech_frames)
                    cw.writerow({"source": source, "label": int(label_clip), "pred": int(pred_clip)})

                    # --- frame score export ---
                    if sw is not None:
                        if args.algo in SPECTRAL_ALGOS:
                            if score is None:
                                score = raw.astype(np.float32)
                            else:
                                score = np.asarray(score, dtype=np.float32)
                                if score.shape[0] != post.shape[0]:
                                    score = np.resize(score, post.shape[0])
                            prob = score_to_prob_generic(score)

                        else:
                            if args.algo in ("energy_zer", "energy_logvar", "energy_zcr", "energy", "zcr"):
                                if ZER is None and args.algo == "energy_zer":
                                    ZER = zero_energy_ratio(frames, eps=0.02)
                                if LOGVAR is None and args.algo in ("energy_logvar",):
                                    LOGVAR = log_energy_variance(E, window=5)
                                score, prob = compute_scores_time_domain(E, Z, ZER, LOGVAR, args.algo)
                            else:
                                # fallback: energy-only score for other time-domain models
                                score, prob = compute_scores_time_domain(E, Z, None, None, "energy")

                        for i in range(len(post)):
                            sw.writerow(
                                {
                                    "model": model,
                                    "source": source,
                                    "frame_idx": i,
                                    "pred_raw": int(raw[i]),
                                    "pred_post": int(post[i]),
                                    "score": float(score[i]) if i < len(score) else float(raw[i]),
                                    "prob": float(prob[i]) if i < len(prob) else float(raw[i]),
                                }
                            )

                    total_audio_sec += len(x) / fs

                wall.append(time.time() - t0)

        finally:
            if sw_fh is not None:
                sw_fh.close()

    # Runtime summary (kept, but you can ignore for SoC conclusions)
    audio_hours = total_audio_sec / 3600.0 if total_audio_sec > 0 else 0.0
    spH_runs = [(w / audio_hours) if audio_hours > 0 else 0.0 for w in wall]
    mean_spH = float(np.mean(spH_runs)) if spH_runs else 0.0
    std_spH = float(np.std(spH_runs, ddof=1)) if len(spH_runs) > 1 else 0.0

    timing_csv = runtime_dir / f"timing_{model}_{stamp}.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "repeat", "wall_time_sec", "sec_per_hour_run", "audio_hours", "date", "note"],
        )
        w.writeheader()
        for i, (wt, sph) in enumerate(zip(wall, spH_runs), start=1):
            w.writerow(
                {
                    "model": model,
                    "repeat": i,
                    "wall_time_sec": round(wt, 4),
                    "sec_per_hour_run": round(sph, 4),
                    "audio_hours": round(audio_hours, 6),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": args.timing_note,
                }
            )

    summary_csv = runtime_dir / f"runtime_summary_{args.tag}.csv"
    new = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "sec_per_hour", "sec_per_hour_mean", "sec_per_hour_std", "repeats", "date", "note"],
        )
        if new:
            w.writeheader()
        w.writerow(
            {
                "model": model,
                "sec_per_hour": round(spH_runs[-1], 4) if spH_runs else "",
                "sec_per_hour_mean": round(mean_spH, 4),
                "sec_per_hour_std": round(std_spH, 4),
                "repeats": len(wall),
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": args.timing_note,
            }
        )

    print(f"[clips]  {clip_csv}")
    if args.emit_scores:
        print(f"[frames] {frame_csv}")
    print(f"[runtime] {summary_csv}")


if __name__ == "__main__":
    main()
