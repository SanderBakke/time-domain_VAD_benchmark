# time_domain_vad_benchmark

A compact, reproducible benchmark for **time-domain VAD** on MCUs/SoCs.  
It includes classic baselines (Energy, ZCR, Energy∩ZCR “Combo”) and **WebRTC VAD (L2/L3)**, uses **SoC-aligned framing** (32 ms frames, 16 ms hop by default), and provides a tidy pipeline for  
**augmentation → run → evaluate (ROC/PR) → tune policy → export SoC config**.

---

## Quick start

```powershell
# 1) Create venv and install deps
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) (Optional) augment a subset
python .\scripts\augment_subset.py --in data --out data_aug_light --mode light
python .\scripts\augment_subset.py --in data --out data_aug_heavy --mode heavy
```

---

### Audio data and the outputs/ directory are not committed.
Use the provided `.gitignore`.

---

## Folder structure

```
vad/
  datasets.py     # iter_dataset(...)
  features.py     # framing + STE + ZCR
  algorithms.py   # EnergyVAD, ZCRVAD, ComboVAD

scripts/          # CLI tools
  run_vad.py          # Energy/ZCR/Combo
  run_vad_webrtc.py   # WebRTC L2/L3 + min_frames policy
  evaluate_vad.py     # Clip metrics + ROC/PR + confusion bars
  tune_vad.py         # Policy sweep on saved frame streams
  make_soc_config.py  # Export chosen params to JSON profiles
  augment_subset.py   # Lightweight augmentation helper

outputs/          # Generated; not committed
  clips/<TAG>/clip_results_*.csv
  frames/<TAG>/<Model>/frame_scores_*.csv
  runtime/<TAG>/{timing_*.csv, runtime_summary_<TAG>.csv}
  eval/<TAG>__<STAMP>/* (tables, confusion bars, ROC/PR plots)
  tuning/<TAG>/* (tuning tables + best_params)
  deploy/<profile>/* (SoC config JSONs)
```

---

## Models included

- **Energy** — Short-time energy with adaptive noise floor (for plotting), decision by threshold + clip policy.  
- **ZCR** — Zero-crossing rate (lower → more voiced), decision + clip policy.  
- **Combo** — Conservative AND: Energy and (not-high ZCR).  
- **WebRTC VAD** — Compiled library with **Level 2** (sensitive) and **Level 3** (stricter).  
  Wrapped in the same clip policy: `median3` (optional), `hangover_ms`, and `min_speech_frames`.

### Clip policy (common)

- `median3` (optional) smooths frame decisions.  
- `hangover_ms` extends positives by N frames.  
- `min_speech_frames` = number of positive frames required to label the clip as speech.

---

## Datasets and augmentation

- Use your base `data/` (not committed).  
- `augment_subset.py` creates `data_aug_light` and `data_aug_heavy` with configurable noise/reverb/pitch/time-stretch augmentations.  
- Typically, **tune on light** and **validate on heavy** to reduce overfitting and test robustness.

---

## Running models (PowerShell examples)

### Set shared variables once

```powershell
$TAG = "light"      # or "heavy"
$DS  = "data_aug_light"   # or "data_aug_heavy"

# SoC-aligned defaults: 32 ms frames, 16 ms hop, low-FP policy
$COMMON = @(
  "--dataset_root", $DS,
  "--tag", $TAG,
  "--emit_scores",
  "--frame_ms", 32, "--hop_ms", 16,
  "--median3",
  "--hangover_ms", 100,
  "--min_speech_frames", 3,
  "--repeat", 5, "--timing_note", "soc_32_16_lowFP"
)
```

### Run Energy / ZCR / Combo

```powershell
python .\scripts\run_vad.py --algo energy $COMMON
python .\scripts\run_vad.py --algo zcr $COMMON
python .\scripts\run_vad.py --algo combo $COMMON
```

**Outputs:**  
- `outputs/clips/<TAG>/clip_results_<algo>_<TAG>.csv`  
- `outputs/frames/<TAG>/<Model>/frame_scores_<Model>_<TAG>.csv`  
- `outputs/runtime/<TAG>/{timing_*.csv, runtime_summary_<TAG>.csv}`

---

## Run WebRTC (L2/L3)

WebRTC requires **10/20/30 ms frames**; use 30/10 ms to approximate 32/16 ms.

```powershell
$COMMON_WB = @(
  "--dataset_root", $DS,
  "--tag", $TAG,
  "--emit_scores",
  "--frame_ms", 30, "--hop_ms", 10,
  "--median3",
  "--hangover_ms", 100,
  "--min_speech_frames", 3,
  "--repeat", 5, "--timing_note", "soc_30_10_lowFP"
)

python .\scripts\run_vad_webrtc.py --level 2 $COMMON_WB
python .\scripts\run_vad_webrtc.py --level 3 $COMMON_WB
```

---

## Evaluate (tables + confusion bars + ROC/PR)

```powershell
$clips = @(
  "outputs/clips/$TAG/clip_results_energy_$TAG.csv",
  "outputs/clips/$TAG/clip_results_zcr_$TAG.csv",
  "outputs/clips/$TAG/clip_results_combo_$TAG.csv",
  "outputs/clips/$TAG/clip_results_webrtc_l2_$TAG.csv",
  "outputs/clips/$TAG/clip_results_webrtc_l3_$TAG.csv"
)
$rt = "outputs/runtime/$TAG"

python .\scripts\evaluate_vad.py --clips $clips --tag $TAG --runtime_summary_dir $rt --make_confbar
```

**Outputs:**  
- CSV summary + table (PNG)  
- Confusion bar  
- ROC/PR plots with:
  - AUC/AP per curve  
  - OP star (clip operating point from policy)  
  - Low-FP diamond (min FPR at Recall ≥ target)

---

## Tune clip policy (hangover & min_speech_frames)

```powershell
python .\scripts\tune_vad.py `
  --tag $TAG `
  --hang_grid "0,50,100,150,200,250,300,400" `
  --min_frames_grid "1,2,3,4,5,6,8,10,12,16,20" `
  --hop_ms 16 `
  --recall_target 0.80
```

**Outputs:**  
- `outputs/tuning/<TAG>/tuning_results.csv`  
- `outputs/tuning/<TAG>/best_params_<TAG>.json`

---

## Export SoC config profiles

```powershell
python .\scripts\make_soc_config.py `
  --light_json outputs/tuning/light/best_params_light.json `
  --heavy_json outputs/tuning/heavy/best_params_heavy.json `
  --out_dir outputs/deploy
```

**Outputs:**  
- `outputs/deploy/balanced/soc_params_balanced.json`  
- `outputs/deploy/lowfp/soc_params_lowfp.json`

---

## Notes & tips

- **Operating point (OP):** star markers in ROC/PR come from your clip policy (`median3`, `hangover`, `min_frames`).  
  They show where the system actually runs.  
- **Low-FP point:** diamond on ROC = minimum FPR that achieves `Recall ≥ recall_target`. Useful for avoiding unnecessary SoC wake-ups.  
- **WebRTC & probabilities:** WebRTC produces binary decisions; synthetic scores are used only for ROC/PR plots, OP remains true to policy.  
- **Framing & MCU parity:** defaults (32 ms / 16 ms; WebRTC 30 / 10) mirror embedded KWS/VAD pipelines for realistic evaluation.
