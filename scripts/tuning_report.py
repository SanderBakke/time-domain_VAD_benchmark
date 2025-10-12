#!/usr/bin/env python

"""
tuning_report.py — Summarize VAD tuning CSVs and suggest presets.

Reads one or more tuning result CSVs produced by scripts/grid_search_vad.py
and generates:
 - Top-5 table (per CSV)
 - Pareto-ish scatter of F1 vs sec_per_hour (annotated)
 - Suggested presets (Conservative / Balanced / Aggressive)
 - A Markdown report tying it all together

Outputs go to: outputs/tuning_report/

Usage examples (PowerShell):
  python scripts/tuning_report.py --csv outputs/tuning/combo_grid.csv --title "Combo VAD Tuning"
  python scripts/tuning_report.py --csv outputs/tuning/energy_grid.csv --csv outputs/tuning/zcr_grid.csv --title "Energy & ZCR Tuning"
"""
import argparse
import csv
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

def read_tuning_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def coerce_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def select_top_k(rows, k=5):
    # rows expected sorted by F1 desc already; if not, sort here
    rows2 = sorted(rows, key=lambda r: (-coerce_float(r.get("f1", 0)), coerce_float(r.get("sec_per_hour", 1e9))))
    return rows2[:k]

def suggest_presets(rows):
    """
    Return three presets (dicts): Conservative, Balanced, Aggressive.
    Heuristic:
      - Sort by F1 desc, then sec_per_hour asc
      - Conservative: among top 20%, choose one with lowest FP proxy (favor higher on_ratio, lower zcr_max if present)
      - Aggressive: among top 20%, choose one with highest recall proxy (lower on_ratio, higher zcr_max)
      - Balanced: the top-1 by the sort
    If fields missing (e.g., zcr_max for energy), ignore gracefully.
    """
    if not rows:
        return None, None, None
    # Coerce numeric
    for r in rows:
        for k in ["f1","sec_per_hour","accuracy","precision","recall",
                  "on_ratio","off_ratio","zcr_max","ema_alpha","hangover_ms"]:
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except:
                    pass
    # Primary sort
    rows_sorted = sorted(rows, key=lambda r: (-r.get("f1", 0.0), r.get("sec_per_hour", 1e9)))
    if len(rows_sorted) <= 3:
        # just map roughly
        bal = rows_sorted[0]
        cons = rows_sorted[min(1, len(rows_sorted)-1)]
        aggr = rows_sorted[min(2, len(rows_sorted)-1)]
        return cons, bal, aggr

    topN = max(3, int(0.2 * len(rows_sorted)))
    pool = rows_sorted[:topN]

    # Balanced: best by sort
    balanced = pool[0]

    # Conservative: pick with high on_ratio (if present) and low zcr_max
    def cons_score(r):
        score = 0.0
        if "on_ratio" in r and isinstance(r["on_ratio"], float):
            score += r["on_ratio"]
        if "zcr_max" in r and isinstance(r["zcr_max"], float):
            score += (0.3 - r["zcr_max"])  # lower zcr_max is more conservative
        if "off_ratio" in r and isinstance(r["off_ratio"], float):
            score += 0.2 * r["off_ratio"]
        return score
    conservative = max(pool, key=cons_score)

    # Aggressive: low on_ratio, high zcr_max
    def aggr_score(r):
        score = 0.0
        if "on_ratio" in r and isinstance(r["on_ratio"], float):
            score += -r["on_ratio"]
        if "zcr_max" in r and isinstance(r["zcr_max"], float):
            score += r["zcr_max"]
        if "off_ratio" in r and isinstance(r["off_ratio"], float):
            score += -0.1 * r["off_ratio"]
        return score
    aggressive = max(pool, key=aggr_score)

    return conservative, balanced, aggressive

def plot_f1_vs_speed(rows, out_png: Path, title: str):
    f1 = np.array([coerce_float(r.get("f1", np.nan)) for r in rows])
    sp = np.array([coerce_float(r.get("sec_per_hour", np.nan)) for r in rows])
    plt.figure(figsize=(6,4))
    plt.scatter(sp, f1, s=18)
    plt.xlabel("Seconds per hour of audio (lower is faster)")
    plt.ylabel("F1 score")
    plt.title(title + "\nF1 vs runtime proxy")
    # annotate top few
    idx = np.argsort(-f1)[:8]
    for i in idx:
        r = rows[i]
        tag = []
        for k in ["on_ratio","off_ratio","zcr_max","ema_alpha","hangover_ms"]:
            v = r.get(k,"")
            if v != "":
                tag.append(f"{k}={float(v):.2f}")
        txt = ", ".join(tag[:3])  # limit clutter
        plt.annotate(txt, (sp[i], f1[i]), fontsize=7, xytext=(2,2), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def write_top_table_md(rows, out_md: Path, title: str, algo_name: str):
    top = select_top_k(rows, k=5)
    with open(out_md, "a", encoding="utf-8") as f:
        f.write(f"\n### Top-5 configurations — {algo_name}\n\n")
        f.write("| Rank | F1 | Precision | Recall | sec/hour | on_ratio | off_ratio | zcr_max | ema_alpha | hangover_ms |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(top, 1):
            f.write(f"| {i} | {coerce_float(r.get('f1')):.4f} | {coerce_float(r.get('precision')):.4f} | {coerce_float(r.get('recall')):.4f} | {coerce_float(r.get('sec_per_hour')):.2f} | {r.get('on_ratio','')} | {r.get('off_ratio','')} | {r.get('zcr_max','')} | {r.get('ema_alpha','')} | {r.get('hangover_ms','')} |\n")

def write_presets_md(cons, bal, aggr, out_md: Path, algo_name: str):
    def fmt(r, name):
        if r is None:
            return f"- **{name}**: (not available)\n"
        return (f"- **{name}**: on_ratio={r.get('on_ratio','—')}, off_ratio={r.get('off_ratio','—')}, "
                f"zcr_max={r.get('zcr_max','—')}, ema_alpha={r.get('ema_alpha','—')}, "
                f"hangover_ms={r.get('hangover_ms','—')}  "
                f"(F1={coerce_float(r.get('f1')):.4f}, sec/hr={coerce_float(r.get('sec_per_hour')):.2f})\n")
    with open(out_md, "a", encoding="utf-8") as f:
        f.write(f"\n### Suggested presets — {algo_name}\n\n")
        f.write(fmt(cons, "Conservative"))
        f.write(fmt(bal, "Balanced (default)"))
        f.write(fmt(aggr, "Aggressive"))
        f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="One or more tuning CSVs from grid_search_vad.py")
    ap.add_argument("--title", type=str, default="VAD Tuning Report")
    ap.add_argument("--outdir", type=str, default="outputs/tuning_report")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report_md = outdir / "report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.title}\n")
        f.write("This report summarizes tuning results and recommends three deployable presets.\n")

    for csv_path in args.csv:
        csv_path = Path(csv_path)
        algo_name = csv_path.stem
        rows = read_tuning_csv(csv_path)
        if not rows:
            with open(report_md, "a", encoding="utf-8") as f:
                f.write(f"\n## {algo_name}\nNo rows found.\n")
            continue

        # Plots
        fig_path = outdir / f"{algo_name}_f1_vs_speed.png"
        plot_f1_vs_speed(rows, fig_path, title=algo_name)

        # Top-5 table
        with open(report_md, "a", encoding="utf-8") as f:
            f.write(f"\n## {algo_name}\n")
            f.write(f"![F1 vs speed]({fig_path.name})\n")

        write_top_table_md(rows, report_md, title=args.title, algo_name=algo_name)

        # Presets
        cons, bal, aggr = suggest_presets(rows)
        write_presets_md(cons, bal, aggr, report_md, algo_name)

    print(f"Report written to: {report_md}")
    print(f"Images saved under: {outdir}")

if __name__ == "__main__":
    main()
