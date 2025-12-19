#!/usr/bin/env python3
# scripts/make_soc_config.py
#
# Turn tuning outputs into SoC-deployable defaults.
# - Inputs: best_params_*.json from scripts/tune_vad.py (contains low_fp/low_fn/balanced per model)
# - Output: outputs/deploy/<PROFILE>/soc_params_<PROFILE>.json (+csv)
# - Also writes a table plot visualizing chosen parameters per model.

import argparse, json, csv
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = [
    "energy", "zcr", "energy_zcr", "energy_logvar", "energy_zer",
    "webrtc_l2", "webrtc_l3",
    "zrmse", "nrmse", "par", "logvar", "zer",
    "band_snr", "lsfm", "ltsv", "ltsd",
]

PROFILES = ["low_fn", "balanced", "low_fp", "low_latency"]


def load_params(p):
    if not p:
        return {}
    pth = Path(p)
    if not pth.exists():
        return {}
    with open(pth, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile_pair(j, model: str, profile: str):
    """
    Returns (min_frames, hang_ms) for model/profile from the tuning json.
    Falls back to balanced, then a safe default.
    """
    d = j.get(model, {})
    if isinstance(d, dict) and profile in d and isinstance(d[profile], dict):
        mf = int(d[profile].get("min_speech_frames", 6))
        ho = int(d[profile].get("hangover_ms", 0))
        return mf, ho

    if isinstance(d, dict) and "balanced" in d and isinstance(d["balanced"], dict):
        mf = int(d["balanced"].get("min_speech_frames", 6))
        ho = int(d["balanced"].get("hangover_ms", 0))
        return mf, ho

    return 6, 0


def pick_value(light_pair, heavy_pair, profile: str):
    lf, lhang = light_pair
    if heavy_pair is None:
        return lf, lhang
    hf, hhang = heavy_pair

    if profile == "low_fp":
        return max(lf, hf), max(lhang, hhang)
    if profile == "low_fn":
        return min(lf, hf), min(lhang, hhang)
    if profile == "low_latency":
        return min(lf, hf), min(lhang, hhang)
    return round((lf + hf) / 2), round((lhang + hhang) / 2)


def clamp_profile(minf: int, hang: int, profile: str):
    # Hardware-friendly caps (tuned for ~hop=16 ms for most models; WebRTC uses hop=10 but we store ms)
    if profile == "low_latency":
        minf = max(1, min(minf, 4))
        hang = max(30, min(hang, 80))
    elif profile == "low_fp":
        minf = max(6, min(minf, 12))
        hang = max(120, min(hang, 220))
    elif profile == "low_fn":
        minf = max(1, min(minf, 5))
        hang = max(0, min(hang, 120))
    else:  # balanced
        minf = max(3, min(minf, 8))
        hang = max(60, min(hang, 180))

    return int(minf), int(hang)


def save_params_table_plot(rows, out_png: Path, title: str):
    """
    rows: list of dicts with keys: model, min_frames_out, hang_ms_out
    Writes a PNG table plot.
    """
    # Build table data
    col_labels = ["Model", "min_speech_frames", "hangover_ms"]
    cell_text = [[r["model"], str(r["min_frames_out"]), str(r["hang_ms_out"])] for r in rows]

    # Figure sizing: scale with number of rows
    n = max(len(cell_text), 1)
    fig_h = 0.35 * n + 1.2
    fig_w = 10.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=12)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Make SoC deployable VAD config from tuned params")
    ap.add_argument("--light", required=True, help="outputs/tuning/<TAG>/best_params_<TAG>.json")
    ap.add_argument("--heavy", default=None, help="optional: outputs/tuning/<TAG>/best_params_<TAG>.json")
    ap.add_argument("--profile", choices=PROFILES, default="balanced")
    ap.add_argument("--outdir", default=None, help="defaults to outputs/deploy/<PROFILE>")
    args = ap.parse_args()

    light = load_params(args.light)
    heavy = load_params(args.heavy) if args.heavy else {}

    outdir = Path(args.outdir) if args.outdir else Path("outputs") / "deploy" / args.profile
    outdir.mkdir(parents=True, exist_ok=True)

    merged = {}
    rows = []

    for m in MODELS:
        lpair = get_profile_pair(light, m, args.profile)
        hpair = get_profile_pair(heavy, m, args.profile) if heavy else None

        mf, ho = pick_value(lpair, hpair, profile=args.profile)
        mf_c, ho_c = clamp_profile(mf, ho, args.profile)

        merged[m] = {"min_speech_frames": mf_c, "hangover_ms": ho_c}

        rows.append({
            "model": m,
            "min_frames_light": lpair[0], "hang_ms_light": lpair[1],
            "min_frames_heavy": (hpair[0] if hpair else ""), "hang_ms_heavy": (hpair[1] if hpair else ""),
            "min_frames_merged": mf, "hang_ms_merged": ho,
            "min_frames_out": mf_c, "hang_ms_out": ho_c,
        })

    out_json = outdir / f"soc_params_{args.profile}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    out_csv = outdir / f"soc_params_{args.profile}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Table plot
    out_png = outdir / f"soc_params_{args.profile}_table.png"
    save_params_table_plot(
        rows=rows,
        out_png=out_png,
        title=f"SoC VAD policy parameters ({args.profile})",
    )

    print(f"[deploy] wrote {out_json}")
    print(f"[deploy] wrote {out_csv}")
    print(f"[deploy] wrote {out_png}")


if __name__ == "__main__":
    main()
