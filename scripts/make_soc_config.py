#!/usr/bin/env python3
# scripts/make_soc_config.py
#
# Turn tuning outputs into SoC-deployable defaults.
# - Inputs: one or two best_params_*.json from scripts/tune_vad.py
# - Profiles:
#     low_latency: cap min_frames small, small hangover
#     balanced:   moderate min_frames, moderate hangover (default)
#     low_fp:     larger min_frames, larger hangover
# - Output: outputs/deploy/<PROFILE>/soc_params_<PROFILE>.json (+csv)

import argparse, json, csv
from pathlib import Path

MODELS = ["energy", "zcr", "combo", "webrtc_l2", "webrtc_l3"]

def load_params(p):
    if not p: return {}
    pth = Path(p)
    if not pth.exists(): return {}
    with open(pth, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_value(light, heavy, how="balanced"):
    # Return (min_frames, hang_ms) after merge rule
    lf, lhang = light
    if heavy is None:
        return lf, lhang
    hf, hhang = heavy
    if how == "low_latency":
        return min(lf, hf), min(lhang, hhang)
    if how == "low_fp":
        return max(lf, hf), max(lhang, hhang)
    # balanced: rounded average
    return round((lf + hf)/2), round((lhang + hhang)/2)

def clamp_profile(minf, hang, profile):
    # Hardware-friendly caps per profile (adjust if you need)
    if profile == "low_latency":
        minf = max(1, min(minf, 4))     # <= ~64 ms if hop=16 ms
        hang = max(30, min(hang, 80))   # 30–80 ms
    elif profile == "low_fp":
        minf = max(6, min(minf, 12))    # 96–192 ms
        hang = max(120, min(hang, 200)) # 120–200 ms
    else:  # balanced
        minf = max(4, min(minf, 8))     # 64–128 ms
        hang = max(80, min(hang, 150))  # 80–150 ms
    return int(minf), int(hang)

def main():
    ap = argparse.ArgumentParser(description="Make SoC deployable VAD config from tuned params")
    ap.add_argument("--light", required=True, help="outputs/tuning/<TAG>/best_params_<TAG>.json")
    ap.add_argument("--heavy", default=None, help="optional: outputs/tuning/<TAG>/best_params_<TAG>.json")
    ap.add_argument("--profile", choices=["low_latency","balanced","low_fp"], default="balanced")
    ap.add_argument("--outdir", default=None, help="defaults to outputs/deploy/<PROFILE>")
    args = ap.parse_args()

    light = load_params(args.light)
    heavy = load_params(args.heavy) if args.heavy else {}

    outdir = Path(args.outdir) if args.outdir else Path("outputs/deploy")/args.profile
    outdir.mkdir(parents=True, exist_ok=True)

    merged = {}
    rows = []
    for m in MODELS:
        l = light.get(m, {"min_speech_frames": 6, "hangover_ms": 0})
        h = heavy.get(m) if heavy else None
        lpair = (int(l.get("min_speech_frames", 6)), int(l.get("hangover_ms", 0)))
        hpair = (int(h.get("min_speech_frames", 6)), int(h.get("hangover_ms", 0))) if h else None

        mf, ho = pick_value(lpair, hpair, how=args.profile)
        mf_c, ho_c = clamp_profile(mf, ho, args.profile)

        merged[m] = {"min_speech_frames": mf_c, "hangover_ms": ho_c}
        rows.append({"model": m, "min_frames_in": mf, "hang_ms_in": ho,
                     "min_frames_out": mf_c, "hang_ms_out": ho_c})

    # Write JSON
    out_json = outdir / f"soc_params_{args.profile}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    # Also write a small CSV for visibility
    out_csv = outdir / f"soc_params_{args.profile}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"[deploy] wrote {out_json}")
    print(f"[deploy] wrote {out_csv}")

if __name__ == "__main__":
    main()
