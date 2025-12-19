#!/usr/bin/env python3
import argparse
import csv
import datetime
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Report-friendly plotting defaults (tuned for side-by-side figs)
# ============================================================
PLOT_STYLE = {
    # Font sizes (points)
    "base": 20,
    "labels": 24,
    "title": 26,
    "ticks": 20,
    "legend": 18,

    # Line/marker sizing
    "linewidth": 3.2,
    "star_size": 360,      # scatter "s" (area) for deployed OP star
    "extra_marker_size": 180,
    "star_legend_ms": 14,  # markersize in legend
    "extra_legend_ms": 11,

    # Figure export resolution
    "dpi": 300,
}

plt.rcParams.update(
    {
        "figure.autolayout": True,

        # Global font sizes
        "font.size": PLOT_STYLE["base"],
        "axes.titlesize": PLOT_STYLE["title"],
        "axes.labelsize": PLOT_STYLE["labels"],
        "xtick.labelsize": PLOT_STYLE["ticks"],
        "ytick.labelsize": PLOT_STYLE["ticks"],
        "legend.fontsize": PLOT_STYLE["legend"],

        # Thicker default lines (also overridden explicitly in ROC/PR)
        "lines.linewidth": PLOT_STYLE["linewidth"],

        # Save figures at high resolution by default
        "savefig.dpi": PLOT_STYLE["dpi"],
    }
)

# ----------------------------
# Utils
# ----------------------------
def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def confusion(y_true, y_pred):
    """
    Accepts lists or numpy arrays.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    fpr = fp / max(fp + tn, 1)
    tpr = rec
    return dict(precision=prec, recall=rec, f1=f1, acc=acc, fpr=fpr, tpr=tpr)


def parse_tag(tag: str) -> Tuple[str, str, str]:
    """
    Expected examples:
      - "light_low_fp"
      - "heavy_balanced"
      - "none_low_fn"
      - "light_low_fp_cascade"
    Returns: (dataset, profile, suffix)
    """
    t = tag.strip()
    parts = t.split("_")
    dataset = parts[0] if parts else t
    profile = "unknown"
    suffix = ""
    if len(parts) >= 2:
        profile = parts[1]
    if len(parts) >= 3:
        suffix = "_".join(parts[2:])
    return dataset, profile, suffix


def infer_model_from_path(p: Path) -> str:
    """
    Robust model-name inference from paths.

    Primary rule: parse the model token from filenames of the form:
      - clip_results_<model>_<tag>.csv
      - frame_scores_<model>_<tag>.csv

    Also supports eval 'inputs' copies which are prefixed like:
      - <tag>__clip_results_<model>_<tag>.csv

    Fallback: longest-first substring matching (prevents 'energy' from
    stealing 'energy_logvar', etc.).
    """
    name = p.name.lower()

    # WebRTC special-cases (keep explicit)
    if "webrtc_l2" in name or "webrtc-l2" in name:
        return "webrtc_l2"
    if "webrtc_l3" in name or "webrtc-l3" in name:
        return "webrtc_l3"

    known = (
        "energy_logvar",
        "energy_zcr",
        "energy_zer",
        "band_snr",
        "ltsv",
        "ltsd",
        "lsfm",
        "zrmse",
        "nrmse",
        "logvar",
        "energy",
        "zcr",
        "zer",
        "par",
        "webrtc_l2",
        "webrtc_l3",
    )
    known_set = set(known)

    # 1) Try strict filename parse for clip_results_... and frame_scores_...
    m = re.search(r"(?:^|__)clip_results_([a-z0-9_]+)_", name)
    if m:
        cand = m.group(1)
        if cand in known_set:
            return cand

    m = re.search(r"(?:^|__)frame_scores_([a-z0-9_]+)_", name)
    if m:
        cand = m.group(1)
        if cand in known_set:
            return cand

    # 2) Fallback: longest-first substring match (critical for energy_* combos)
    for k in sorted(known_set, key=len, reverse=True):
        if k in name:
            return k

    # 3) Last resort
    return p.stem.lower()


def tag_from_clip_path(p: Path) -> Optional[str]:
    """
    Your convention:
      outputs/clips/<tag>/clip_results_<model>_<tag>.csv
    So we trust the parent folder name.
    """
    try:
        return p.parent.name
    except Exception:
        return None


# ----------------------------
# Curves from clip scores (clip-score = max over frames)
# ----------------------------
def compute_curves(labels: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)

    if labels.size == 0:
        return dict(
            fpr=np.array([0.0, 1.0]),
            tpr=np.array([0.0, 1.0]),
            auc=np.array(np.nan),
            recall=np.array([0.0, 1.0]),
            precision=np.array([1.0, 0.0]),
            ap=np.array(np.nan),
        )

    order = np.argsort(scores)[::-1]
    y = labels[order]
    sc = scores[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    P = float(tp[-1])
    N = float(fp[-1])

    # keep only change points in score
    ch = np.r_[True, sc[1:] != sc[:-1]]
    tp = tp[ch]
    fp = fp[ch]

    tpr = tp / max(P, 1.0)
    fpr = fp / max(N, 1.0)

    auc = float(np.trapezoid(tpr, fpr)) if len(tpr) >= 2 else float("nan")

    recall = tpr
    precision = tp / np.maximum(tp + fp, 1)

    # Average precision (stepwise)
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else float("nan")

    return dict(fpr=fpr, tpr=tpr, auc=auc, recall=recall, precision=precision, ap=ap)


# ----------------------------
# Files discovery
# ----------------------------
def discover_frame_files(tag: str) -> List[Path]:
    base = Path("outputs/frames") / tag
    if not base.exists():
        return []
    files = sorted(base.rglob(f"frame_scores_*_{tag}.csv"))
    if not files:
        files = sorted(base.rglob("frame_scores_*.csv"))
    return files


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_conf_bar(rows: List[Dict], out_png: Path, title: str):
    names = [r["model"] for r in rows]
    tp = np.array([r["tp"] for r in rows])
    fp = np.array([r["fp"] for r in rows])
    fn = np.array([r["fn"] for r in rows])
    tn = np.array([r["tn"] for r in rows])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13.5, 6.5), dpi=PLOT_STYLE["dpi"])

    ax.bar(x, tp, label="True Positive (TP)")
    ax.bar(x, fp, bottom=tp, label="False Positive (FP)")
    ax.bar(x, fn, bottom=tp + fp, label="False Negative (FN)")
    ax.bar(x, tn, bottom=tp + fp + fn, label="True Negative (TN)")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in names], rotation=25, ha="right")
    ax.set_ylabel("Number of 1 s clips", fontsize=PLOT_STYLE["labels"])
    ax.set_title(title, fontsize=PLOT_STYLE["title"])
    ax.tick_params(axis="both", which="major", labelsize=PLOT_STYLE["ticks"])
    ax.legend(fontsize=PLOT_STYLE["legend"], framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_table_png(df, title: str, out_png: Path, figsize=(22, 6), fontsize=14):
    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_STYLE["dpi"])
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.15, 1.30)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    ax.set_title(title, pad=18, fontsize=PLOT_STYLE["title"])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    kind: str,
    dataset: str,
    profile: str,
    tag: str,
    models: List[str],
    curves: Dict[str, Dict[str, np.ndarray]],
    deployed_ops_roc: Dict[str, Tuple[float, float]],
    deployed_ops_pr: Dict[str, Tuple[float, float]],
    extra_ops_roc: Dict[str, Dict[str, Tuple[float, float]]],  # profile -> model -> (FPR,TPR)
    extra_ops_pr: Dict[str, Dict[str, Tuple[float, float]]],   # profile -> model -> (Recall,Precision)
    out_png: Path,
):
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=PLOT_STYLE["dpi"])

    handles = []
    labels = []
    plotted_models = []

    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for i, m in enumerate(models):
        c = curves.get(m)
        if c is None:
            continue

        ls = linestyles[i % len(linestyles)]
        z = 2 + i

        if kind == "roc":
            h, = ax.plot(
                c["fpr"],
                c["tpr"],
                linewidth=PLOT_STYLE["linewidth"],
                linestyle=ls,
                alpha=0.85,
                zorder=z,
                label=f"{m} (AUC={c['auc']:.3f})",
            )
        else:
            h, = ax.plot(
                c["recall"],
                c["precision"],
                linewidth=PLOT_STYLE["linewidth"],
                linestyle=ls,
                alpha=0.85,
                zorder=z,
                label=f"{m} (AP={c['ap']:.3f})",
            )
        handles.append(h)
        labels.append(h.get_label())
        plotted_models.append(m)

    marker_map = {
        "balanced": ("o", "Balanced SoC operating point"),
        "low_fp": ("s", "Low-FP SoC operating point"),
        "low_fn": ("^", "Low-FN SoC operating point"),
    }

    for m, h in zip(plotted_models, handles):
        color = h.get_color()

        # Deployed OP (this run) = star
        if kind == "roc" and m in deployed_ops_roc:
            x, y = deployed_ops_roc[m]
            ax.scatter([x], [y], marker="*", s=PLOT_STYLE["star_size"], c=[color], edgecolors="k", zorder=6)
        if kind == "pr" and m in deployed_ops_pr:
            r, p = deployed_ops_pr[m]
            ax.scatter([r], [p], marker="*", s=PLOT_STYLE["star_size"], c=[color], edgecolors="k", zorder=6)

        # Extra profile OPs (only if those profiles were included in --clips)
        for prof, (mk, _) in marker_map.items():
            if prof == profile:
                continue
            if kind == "roc":
                if prof in extra_ops_roc and m in extra_ops_roc[prof]:
                    x, y = extra_ops_roc[prof][m]
                    ax.scatter([x], [y], marker=mk, s=PLOT_STYLE["extra_marker_size"], c=[color],
                               edgecolors="k", zorder=5)
            else:
                if prof in extra_ops_pr and m in extra_ops_pr[prof]:
                    r, p = extra_ops_pr[prof][m]
                    ax.scatter([r], [p], marker=mk, s=PLOT_STYLE["extra_marker_size"], c=[color],
                               edgecolors="k", zorder=5)

    if kind == "roc":
        ax.set_xlabel("False Positive Rate (FPR) — clip level", fontsize=18)
        ax.set_ylabel("True Positive Rate (TPR / Recall) — clip level", fontsize=18)
        ax.set_title(f"ROC curves (clip level) — {dataset} — {profile}  [{tag}]", fontsize=PLOT_STYLE["title"])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
    else:
        ax.set_xlabel("Recall — clip level", fontsize=18)
        ax.set_ylabel("Precision — clip level", fontsize=18)
        ax.set_title(f"PR curves (clip level) — {dataset} — {profile}  [{tag}]", fontsize=PLOT_STYLE["title"])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)

    ax.tick_params(axis="both", which="major", labelsize=PLOT_STYLE["ticks"])
    ax.grid(alpha=0.25)

    leg1 = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        fontsize=PLOT_STYLE["legend"],
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=2.2,
        handletextpad=0.6,
    )
    ax.add_artist(leg1)

    marker_handles = [
        plt.Line2D(
            [0], [0],
            marker="*",
            color="w",
            label="Deployed operating point",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=PLOT_STYLE["star_legend_ms"],
        ),
    ]

    for prof, (mk, lab) in marker_map.items():
        if prof == profile:
            continue
        if kind == "roc":
            has_any = prof in extra_ops_roc and len(extra_ops_roc[prof]) > 0
        else:
            has_any = prof in extra_ops_pr and len(extra_ops_pr[prof]) > 0
        if has_any:
            marker_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker=mk,
                    color="w",
                    label=lab,
                    markerfacecolor="gray",
                    markeredgecolor="k",
                    markersize=PLOT_STYLE["extra_legend_ms"],
                )
            )

    ax.legend(
        handles=marker_handles,
        loc="lower left",
        fontsize=PLOT_STYLE["legend"],
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=2.0,
        handletextpad=0.6,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate VAD outputs: clip metrics + clip-level ROC/PR from frame scores. "
                    "Supports cascade simulation when --tag includes 'cascade'."
    )
    ap.add_argument("--clips", nargs="+", required=True,
                    help="One or more clip_results_*.csv files. May include multiple profiles "
                         "(e.g., light_low_fp, light_low_fn, light_balanced) to overlay OPs.")
    ap.add_argument("--frame_scores", nargs="*", default=[],
                    help="Optional explicit frame_scores CSVs (else auto-discover under outputs/frames/<tag>/).")
    ap.add_argument("--tag", required=True,
                    help="Primary tag folder under outputs/* to evaluate (e.g., heavy_balanced_cascade). "
                         "If it contains 'cascade', cascade simulation is enabled.")
    ap.add_argument("--make_confbar", action="store_true",
                    help="If set, create stacked TP/FP/FN/TN bar plot.")
    ap.add_argument("--cascade_pairs", nargs="*", default=[],
                    help="Explicit cascade pairs like energy+band_snr energy+webrtc_l3 ...")
    ap.add_argument("--only_cascades", action="store_true",
                    help="If set, exclude standalone models and keep only cascades in tables/plots.")
    args = ap.parse_args()

    tag = args.tag
    dataset, profile, _ = parse_tag(tag)
    enable_cascade = ("cascade" in tag.lower())

    if args.only_cascades and not enable_cascade:
        raise SystemExit("[error] --only_cascades requires cascade mode (tag must include 'cascade').")

    out_root = Path("outputs/eval") / f"{args.tag}__{now_tag()}"
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "roc").mkdir(parents=True, exist_ok=True)
    (out_root / "pr").mkdir(parents=True, exist_ok=True)
    (out_root / "inputs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load clip results for one or more tags/profiles (same dataset)
    # ------------------------------------------------------------
    clip_labels_by_tag: Dict[str, Dict[str, int]] = {}
    clip_preds_by_tag_model: Dict[str, Dict[str, Dict[str, int]]] = {}
    tags_seen: List[str] = []

    # Track primary-tag model order from args.clips order
    primary_models_in_order: List[str] = []
    seen_primary = set()

    for cfile in args.clips:
        cpath = Path(cfile)
        if not cpath.exists():
            print(f"[warn] Missing clip file: {cpath}")
            continue

        ctag = tag_from_clip_path(cpath)
        if not ctag:
            print(f"[warn] Could not infer tag from clip file: {cpath}")
            continue

        dset_i, prof_i, _ = parse_tag(ctag)
        if dset_i != dataset:
            continue

        tags_seen.append(ctag)

        model = infer_model_from_path(cpath)

        clip_labels_by_tag.setdefault(ctag, {})
        clip_preds_by_tag_model.setdefault(ctag, {})
        clip_preds_by_tag_model[ctag].setdefault(model, {})

        # Copy inputs for audit
        try:
            dst = out_root / "inputs" / f"{ctag}__{cpath.name}"
            if not dst.exists():
                dst.write_bytes(cpath.read_bytes())
        except Exception:
            pass

        with open(cpath, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                src = row.get("source", "").strip()
                if not src:
                    continue
                lab = int(safe_float(row.get("label", 0.0)))
                pred = int(safe_float(row.get("pred", 0.0)))

                if src not in clip_labels_by_tag[ctag]:
                    clip_labels_by_tag[ctag][src] = lab
                clip_preds_by_tag_model[ctag][model][src] = pred

        if ctag == args.tag and model not in seen_primary:
            seen_primary.add(model)
            primary_models_in_order.append(model)

    if args.tag not in clip_labels_by_tag or args.tag not in clip_preds_by_tag_model:
        raise SystemExit(
            f"[error] No clip results found for primary --tag '{args.tag}'. "
            f"Make sure --clips includes files under outputs/clips/{args.tag}/"
        )

    clip_label_by_source = clip_labels_by_tag[args.tag]
    all_sources = sorted(clip_label_by_source.keys())
    clip_pred_by_model_primary = clip_preds_by_tag_model[args.tag]

    # ----------------------------
    # Standalone summary metrics (primary tag) + deployed OP
    # ----------------------------
    summary_rows: List[Dict] = []
    deployed_ops_roc: Dict[str, Tuple[float, float]] = {}
    deployed_ops_pr: Dict[str, Tuple[float, float]] = {}

    for m in primary_models_in_order:
        preds = clip_pred_by_model_primary.get(m, {})
        y_true = [clip_label_by_source[s] for s in all_sources]
        y_pred = [preds.get(s, 0) for s in all_sources]

        tp, fp, fn, tn = confusion(y_true, y_pred)
        met = metrics(tp, fp, fn, tn)

        summary_rows.append(
            dict(
                model=m,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(met["precision"], 4),
                recall=round(met["recall"], 4),
                f1=round(met["f1"], 4),
                acc=round(met["acc"], 4),
                fpr=round(met["fpr"], 4),
                tpr=round(met["tpr"], 4),
            )
        )
        deployed_ops_roc[m] = (met["fpr"], met["tpr"])
        deployed_ops_pr[m] = (met["recall"], met["precision"])

    # ----------------------------
    # Cascade simulation (primary tag)
    # ----------------------------
    cascade_models: List[str] = []
    cascade_rows: List[Dict] = []
    cascade_pair_by_name: Dict[str, Tuple[str, str]] = {}

    if enable_cascade:
        pairs: List[Tuple[str, str]] = []
        if args.cascade_pairs:
            for s in args.cascade_pairs:
                if "+" not in s:
                    continue
                a, b = s.split("+", 1)
                pairs.append((a.strip().lower(), b.strip().lower()))
        else:
            # Heuristic fallback (you normally pass explicit pairs)
            stage2_candidates = [m for m in primary_models_in_order if m in ("band_snr", "webrtc_l2", "webrtc_l3")]
            stage1_candidates = [m for m in primary_models_in_order if m not in stage2_candidates]
            for s1 in stage1_candidates:
                for s2 in stage2_candidates:
                    pairs.append((s1, s2))

        for s1, s2 in pairs:
            if s1 not in clip_pred_by_model_primary or s2 not in clip_pred_by_model_primary:
                continue

            name = f"cascade_{s1}_{s2}"
            cascade_pair_by_name[name] = (s1, s2)
            cascade_models.append(name)

            y_true = []
            y_pred = []
            for src in all_sources:
                y_true.append(clip_label_by_source[src])
                p1 = clip_pred_by_model_primary[s1].get(src, 0)
                p2 = clip_pred_by_model_primary[s2].get(src, 0)
                y_pred.append(1 if (p1 == 1 and p2 == 1) else 0)

            tp, fp, fn, tn = confusion(y_true, y_pred)
            met = metrics(tp, fp, fn, tn)

            cascade_rows.append(
                dict(
                    model=name,
                    tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=round(met["precision"], 4),
                    recall=round(met["recall"], 4),
                    f1=round(met["f1"], 4),
                    acc=round(met["acc"], 4),
                    fpr=round(met["fpr"], 4),
                    tpr=round(met["tpr"], 4),
                )
            )
            deployed_ops_roc[name] = (met["fpr"], met["tpr"])
            deployed_ops_pr[name] = (met["recall"], met["precision"])

    # ----------------------------
    # Extra profile operating points (if included in --clips)
    # Compute BOTH standalone + cascades for those profiles.
    # ----------------------------
    extra_ops_roc: Dict[str, Dict[str, Tuple[float, float]]] = {}  # prof -> model -> (FPR,TPR)
    extra_ops_pr: Dict[str, Dict[str, Tuple[float, float]]] = {}   # prof -> model -> (Recall,Precision)

    for ctag in sorted(set(tags_seen)):
        dset_i, prof_i, _ = parse_tag(ctag)
        if dset_i != dataset:
            continue
        if prof_i == profile:
            continue
        if ctag not in clip_labels_by_tag or ctag not in clip_preds_by_tag_model:
            continue

        labels_other = clip_labels_by_tag[ctag]
        preds_other_by_model = clip_preds_by_tag_model[ctag]

        common_sources = [s for s in all_sources if s in labels_other]
        if not common_sources:
            continue

        extra_ops_roc.setdefault(prof_i, {})
        extra_ops_pr.setdefault(prof_i, {})

        # Standalone OPs for this profile
        for m in primary_models_in_order:
            if m not in preds_other_by_model:
                continue
            y_true = [labels_other[s] for s in common_sources]
            y_pred = [preds_other_by_model[m].get(s, 0) for s in common_sources]

            tp, fp, fn, tn = confusion(y_true, y_pred)
            met = metrics(tp, fp, fn, tn)

            extra_ops_roc[prof_i][m] = (met["fpr"], met["tpr"])
            extra_ops_pr[prof_i][m] = (met["recall"], met["precision"])

        # Cascade OPs for this profile (same cascade names as primary)
        if enable_cascade:
            for cname, (s1, s2) in cascade_pair_by_name.items():
                if s1 not in preds_other_by_model or s2 not in preds_other_by_model:
                    continue
                y_true = [labels_other[s] for s in common_sources]
                y_pred = []
                for s in common_sources:
                    p1 = preds_other_by_model[s1].get(s, 0)
                    p2 = preds_other_by_model[s2].get(s, 0)
                    y_pred.append(1 if (p1 == 1 and p2 == 1) else 0)

                tp, fp, fn, tn = confusion(y_true, y_pred)
                met = metrics(tp, fp, fn, tn)
                extra_ops_roc[prof_i][cname] = (met["fpr"], met["tpr"])
                extra_ops_pr[prof_i][cname] = (met["recall"], met["precision"])

    # ----------------------------
    # Apply "only cascades" filtering
    # ----------------------------
    if args.only_cascades:
        rows_for_summary = list(cascade_rows)
        models_for_outputs = list(cascade_models)

        deployed_ops_roc = {k: v for k, v in deployed_ops_roc.items() if k in models_for_outputs}
        deployed_ops_pr = {k: v for k, v in deployed_ops_pr.items() if k in models_for_outputs}

        for prof_i in list(extra_ops_roc.keys()):
            extra_ops_roc[prof_i] = {k: v for k, v in extra_ops_roc[prof_i].items() if k in models_for_outputs}
        for prof_i in list(extra_ops_pr.keys()):
            extra_ops_pr[prof_i] = {k: v for k, v in extra_ops_pr[prof_i].items() if k in models_for_outputs}
    else:
        rows_for_summary = list(summary_rows) + (list(cascade_rows) if enable_cascade else [])
        models_for_outputs = list(primary_models_in_order) + (list(cascade_models) if enable_cascade else [])

    # ----------------------------
    # Write summary CSV + summary PNG table
    # ----------------------------
    out_summary_csv = out_root / "tables" / f"summary_metrics_{args.tag}.csv"
    summary_headers = ["model", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "acc", "fpr", "tpr"]

    with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_headers)
        w.writeheader()
        for r in rows_for_summary:
            w.writerow({k: r.get(k, "") for k in summary_headers})

    try:
        import pandas as pd
        df = pd.read_csv(out_summary_csv)
        plot_table_png(
            df,
            title=f"VAD clip-level summary — {dataset} — {profile}  [{args.tag}]",
            out_png=out_root / "tables" / f"summary_table_{args.tag}.png",
            figsize=(22, 6),
            fontsize=14,
        )
    except Exception as e:
        print(f"[warn] Could not render summary table PNG: {e}")

    # ----------------------------
    # Optional confusion bar plot
    # ----------------------------
    if args.make_confbar:
        try:
            plot_conf_bar(
                rows_for_summary,
                out_png=out_root / "tables" / f"confusion_bar_{args.tag}.png",
                title=f"TP/FP/FN/TN: dataset={dataset} profile={profile} tag={args.tag}",
            )
        except Exception as e:
            print(f"[warn] Could not render confusion bar: {e}")

    # ----------------------------
    # Operating points table (deployed + any extra profiles present in --clips)
    # ----------------------------
    op_headers = ["model", "deployed_FPR", "deployed_TPR", "deployed_precision", "deployed_recall"]

    # Add columns only for profiles that actually exist in extra_ops
    for prof_i in ("balanced", "low_fp", "low_fn"):
        if prof_i == profile:
            continue
        if prof_i in extra_ops_roc and len(extra_ops_roc[prof_i]) > 0:
            op_headers += [f"{prof_i}_FPR", f"{prof_i}_TPR", f"{prof_i}_precision", f"{prof_i}_recall"]

    op_rows = []
    for m in models_for_outputs:
        fpr_dep, tpr_dep = deployed_ops_roc.get(m, (np.nan, np.nan))
        rec_dep, prec_dep = deployed_ops_pr.get(m, (np.nan, np.nan))

        row = dict(
            model=m,
            deployed_FPR=round(float(fpr_dep), 4) if np.isfinite(fpr_dep) else "",
            deployed_TPR=round(float(tpr_dep), 4) if np.isfinite(tpr_dep) else "",
            deployed_precision=round(float(prec_dep), 4) if np.isfinite(prec_dep) else "",
            deployed_recall=round(float(rec_dep), 4) if np.isfinite(rec_dep) else "",
        )

        for prof_i in ("balanced", "low_fp", "low_fn"):
            if prof_i == profile:
                continue
            if prof_i in extra_ops_roc and m in extra_ops_roc[prof_i] and prof_i in extra_ops_pr and m in extra_ops_pr[prof_i]:
                fpr_i, tpr_i = extra_ops_roc[prof_i][m]
                rec_i, prec_i = extra_ops_pr[prof_i][m]
                row[f"{prof_i}_FPR"] = round(float(fpr_i), 4)
                row[f"{prof_i}_TPR"] = round(float(tpr_i), 4)
                row[f"{prof_i}_precision"] = round(float(prec_i), 4)
                row[f"{prof_i}_recall"] = round(float(rec_i), 4)

        op_rows.append(row)

    out_op_csv = out_root / "tables" / f"operating_points_{args.tag}.csv"
    with open(out_op_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=op_headers)
        w.writeheader()
        for r in op_rows:
            w.writerow({k: r.get(k, "") for k in op_headers})

    try:
        import pandas as pd
        dfop = pd.read_csv(out_op_csv)
        plot_table_png(
            dfop,
            title=f"Operating points (SoC profiles) — {dataset} — primary={profile}  [{args.tag}]",
            out_png=out_root / "tables" / f"operating_points_{args.tag}.png",
            figsize=(22, 5.5),
            fontsize=14,
        )
    except Exception as e:
        print(f"[warn] Could not render operating-points table PNG: {e}")

    # ----------------------------
    # Frame scores -> clip scores (primary tag only)
    # ----------------------------
    frame_files = [Path(p) for p in args.frame_scores] if args.frame_scores else discover_frame_files(args.tag)
    frame_files = [p for p in frame_files if p.exists()]

    if not frame_files:
        print("[info] No frame_scores found; ROC/PR skipped.")
        print(f"[outputs] {out_root}")
        return

    # frame_scores_by_model[model][src] = list of per-frame scores
    frame_scores_by_model: Dict[str, Dict[str, List[float]]] = {}

    for fp in frame_files:
        model = infer_model_from_path(fp)
        frame_scores_by_model.setdefault(model, {})

        with open(fp, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                src = row.get("source", "").strip()
                if not src:
                    continue
                if src not in clip_label_by_source:
                    continue

                v = row.get("prob", None)
                if v is None or v == "":
                    v = row.get("score", 0.0)
                score = safe_float(v, 0.0)

                frame_scores_by_model[model].setdefault(src, []).append(score)

    # Convert to clip-scores via max-over-frames
    clip_scores: Dict[str, Dict[str, float]] = {}
    for m, per_clip in frame_scores_by_model.items():
        if not per_clip:
            continue
        clip_scores.setdefault(m, {})
        for src, vals in per_clip.items():
            clip_scores[m][src] = float(np.max(vals))

    # ----------------------------
    # Build curves for standalone + cascades
    # ----------------------------
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    # Standalone curves (only if not only_cascades)
    if not args.only_cascades:
        for m in primary_models_in_order:
            if m not in clip_scores:
                continue
            scores = []
            labels = []
            for src in all_sources:
                if src in clip_scores[m]:
                    scores.append(clip_scores[m][src])
                    labels.append(clip_label_by_source[src])
            if scores:
                curves[m] = compute_curves(np.array(labels, np.int32), np.array(scores, np.float64))

    # Cascade curves: stage2 score gated by stage1 deployed decision
    if enable_cascade:
        for cname, (s1, s2) in cascade_pair_by_name.items():
            if s1 not in clip_pred_by_model_primary:
                continue
            if s2 not in clip_scores:
                continue  # can't build ROC/PR without stage2 frame-derived scores

            scores = []
            labels = []
            for src in all_sources:
                if src not in clip_scores[s2]:
                    continue
                gate = clip_pred_by_model_primary[s1].get(src, 0)
                s2_score = clip_scores[s2][src]
                scores.append(s2_score if gate == 1 else 0.0)
                labels.append(clip_label_by_source[src])

            if scores:
                curves[cname] = compute_curves(np.array(labels, np.int32), np.array(scores, np.float64))

    # ----------------------------
    # ROC + PR plots
    # ----------------------------
    model_list_for_plots = [m for m in models_for_outputs if m in curves]
    if model_list_for_plots:
        plot_roc_pr(
            kind="roc",
            dataset=dataset,
            profile=profile,
            tag=args.tag,
            models=model_list_for_plots,
            curves=curves,
            deployed_ops_roc=deployed_ops_roc,
            deployed_ops_pr=deployed_ops_pr,
            extra_ops_roc=extra_ops_roc,
            extra_ops_pr=extra_ops_pr,
            out_png=out_root / "roc" / f"roc_all_{args.tag}.png",
        )
        plot_roc_pr(
            kind="pr",
            dataset=dataset,
            profile=profile,
            tag=args.tag,
            models=model_list_for_plots,
            curves=curves,
            deployed_ops_roc=deployed_ops_roc,
            deployed_ops_pr=deployed_ops_pr,
            extra_ops_roc=extra_ops_roc,
            extra_ops_pr=extra_ops_pr,
            out_png=out_root / "pr" / f"pr_all_{args.tag}.png",
        )

    print(f"[outputs] {out_root}")


if __name__ == "__main__":
    main()
