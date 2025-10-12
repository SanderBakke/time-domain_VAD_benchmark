# *NOT UPDATED* 
# Time-Domain VAD Benchmark (Simple & Educational)

This is a **beginner-friendly** project to implement and compare **time‑domain Voice Activity Detection (VAD)** variants:
- Short‑Time Energy (STE) thresholding
- Zero‑Crossing Rate (ZCR) thresholding
- A combined, adaptive approach with **noise floor**, **hysteresis** (separate on/off thresholds), and **hangover**

You can run **clip‑level** tests on the **Google Speech Commands v0.02** dataset using **noise-only** clips from `_background_noise_` as non‑speech and the word folders as speech.

> Goal: let you try different VADs, understand each step, and compare accuracy/precision/recall/F1 and runtime. The code is small, heavily commented, and easy to tweak.

---

## 0) Quick start

### Install (Python 3.9+ recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the dataset (Speech Commands v0.02)
You have **three options** — pick **one**:

**A. Manual download (recommended for this project)**
1. Download the zip from the official storage (Google) or a mirror (e.g., Kaggle).
2. Unzip to a folder, e.g. `~/data/speech_commands_v0.02`.
3. You should see many word folders (e.g., `yes/`, `no/`, ...) **and** a folder named **`_background_noise_`**.
4. Pass that path to the script via `--data_dir`.

**B. TensorFlow Datasets (TFDS)**  
If you already use TFDS, you can adapt `datasets.py` to load from TFDS. (Not enabled by default here to keep deps light.)

**C. Torchaudio**  
Also possible, but requires PyTorch/Torchaudio. We keep this project light-weight and file‑system based by default.

> If you go with (A), you do **not** need TFDS/PyTorch.

### Run a baseline (energy VAD with adaptive threshold + default params)
```bash
python scripts/run_vad.py --data_dir ~/data/speech_commands_v0.02 --algo energy_adaptive
```

### Try ZCR or the combined VAD
```bash
# ZCR only
python scripts/run_vad.py --data_dir ~/data/speech_commands_v0.02 --algo zcr

# Combined (energy + zcr + hysteresis + hangover)
python scripts/run_vad.py --data_dir ~/data/speech_commands_v0.02 --algo combo
```

### Common tweaks
```bash
# Frame/hop (ms), hysteresis (dB-like ratio), hangover (ms)
python scripts/run_vad.py --data_dir ~/data/speech_commands_v0.02 --algo combo \
  --frame_ms 20 --hop_ms 10 --on_ratio 3.0 --off_ratio 1.5 --zcr_max 0.12 --hangover_ms 200
```

The script will print:
- **Clip-level metrics**: Accuracy, Precision, Recall, F1
- **Runtime**: processing seconds per hour of audio
- **A CSV** with per-file decisions in `outputs/`

---

## 1) What each step does (plain words)

- **Framing**: We slice audio into small **frames** (e.g., 20 ms) with overlap (e.g., hop 10 ms). Speech is not constant; frame‑wise stats capture **when** speech starts/stops.

- **Short‑Time Energy (STE)**: Sum of squares in a frame — a simple loudness measure. Speech frames usually have **higher energy** than silence/noise.

- **Zero‑Crossing Rate (ZCR)**: How often the signal crosses zero in a frame. Voiced vowels are smoother (lower ZCR), while unvoiced consonants and some noises flip more (higher ZCR).

- **Adaptive threshold**: Instead of a fixed cutoff, we estimate a **noise floor** on the fly (e.g., an exponential moving average of low‑energy frames) and compare frames to that.

- **Hysteresis**: Two thresholds — a **higher one to turn ON** speech and a **lower one to turn OFF** speech — to reduce flickering.

- **Hangover**: After detecting speech, **stay ON** for a short time to avoid chopping syllables.

---

## 2) Project layout

```
time_domain_vad_benchmark/
├─ vad/
│  ├─ __init__.py
│  ├─ features.py        # framing, STE, ZCR
│  ├─ algorithms.py      # EnergyVAD, ZCRVAD, ComboVAD (adaptive+hysteresis+hangover)
│  ├─ datasets.py        # file-system loader for Speech Commands + background noise slicing
│  └─ metrics.py         # clip-level metrics & timing helpers
├─ scripts/
│  └─ run_vad.py         # CLI to run evaluations
├─ outputs/              # metrics and CSVs will be saved here
├─ requirements.txt
└─ README.md
```

---

## 3) What counts as speech/non‑speech in this project?

- **Speech**: any 1‑second file from a **word folder** (e.g., `yes/`, `no/`, `left/`, `right/`, etc.).
- **Non‑speech**: 1‑second segments cut from the files inside **`_background_noise_/`**.

That’s enough to compare VAD variants on a large, varied set of short clips.

---

## 4) Tips

- Start with `--algo energy_fixed` to feel the effect of a single threshold (expect it to be brittle).  
- Switch to `--algo energy_adaptive` with `--on_ratio/--off_ratio` to see stability improve.  
- Add ZCR via `--algo combo` and tune `--zcr_max`.  
- Keep `--hangover_ms` around 150–300 ms to avoid word chopping.
- For reproducible splits, you can pin a random seed and cap the number of clips using `--max_files` during iteration.

---

## 5) License

MIT, educational use.
