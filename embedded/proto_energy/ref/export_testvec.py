#!/usr/bin/env python3
"""
export_testvec.py

Generates a C header file (test_vectors/test_vectors.h) containing a small
set of int16 test frames and their expected int64 frame-level energies.

The energy computation uses short_time_energy_i16() from vad/features.py,
which is the Python golden reference for the embedded energy kernel.

Usage (run from repo root OR from this file's directory):
    python embedded/proto_energy/ref/export_testvec.py

Output:
    embedded/proto_energy/test_vectors/test_vectors.h

Parameters:
    FRAME_SIZE = 512 samples  (32 ms at 16 kHz)
    No hopping needed here; each test frame is independent.
"""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# Locate the repo root and import the benchmark's vad package.
# This script can be run from the repo root or from any subdirectory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vad.features import float_to_q15, short_time_energy_i16  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_SIZE = 512   # 32 ms at 16 kHz
SR = 16000
RNG_SEED = 42

OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "test_vectors")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_vectors.h")


# ---------------------------------------------------------------------------
# Frame construction helpers
# ---------------------------------------------------------------------------

def make_zero_frame() -> np.ndarray:
    """All-zero frame (complete silence)."""
    return np.zeros(FRAME_SIZE, dtype=np.int16)


def make_constant_frame(value: int) -> np.ndarray:
    """Frame filled with a constant int16 value."""
    v = np.clip(int(value), -32768, 32767)
    return np.full(FRAME_SIZE, v, dtype=np.int16)


def make_sine_frame(freq_hz: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sine wave at freq_hz with float amplitude in [0, 1].
    Quantized to int16 via float_to_q15.
    Phase is randomized (non-deterministic appearance, but seed-controlled).
    """
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(FRAME_SIZE, dtype=np.float64) / SR
    x = amplitude * np.sin(2 * np.pi * freq_hz * t + phase)
    return float_to_q15(x.astype(np.float32))


def make_sine_plus_noise_frame(
    freq_hz: float, sine_amplitude: float, noise_amplitude: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Sine + additive white Gaussian noise, both in float [-1, 1], then quantized."""
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(FRAME_SIZE, dtype=np.float64) / SR
    sine = sine_amplitude * np.sin(2 * np.pi * freq_hz * t + phase)
    noise = noise_amplitude * rng.standard_normal(FRAME_SIZE)
    x = np.clip(sine + noise, -1.0, 1.0)
    return float_to_q15(x.astype(np.float32))


def make_noise_frame(amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at float amplitude in [0, 1], quantized to int16."""
    x = np.clip(amplitude * rng.standard_normal(FRAME_SIZE), -1.0, 1.0)
    return float_to_q15(x.astype(np.float32))


def make_alternating_frame() -> np.ndarray:
    """Alternating +32767 / -32768 — stress test for accumulator."""
    data = np.empty(FRAME_SIZE, dtype=np.int16)
    data[0::2] = 32767
    data[1::2] = -32768
    return data


# ---------------------------------------------------------------------------
# Build the list of test frames
# ---------------------------------------------------------------------------

def build_test_frames(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """
    Returns a list of (description, int16_frame) tuples.
    The description is used as a comment in the generated header.
    """
    frames = [
        ("zero / silence",
         make_zero_frame()),

        ("constant +100 (very low amplitude)",
         make_constant_frame(100)),

        ("constant +1000 (moderate constant)",
         make_constant_frame(1000)),

        ("constant -1000 (negative constant, same energy as +1000)",
         make_constant_frame(-1000)),

        ("sine 440 Hz, amplitude 0.30 (typical speech fundamental region)",
         make_sine_frame(440.0, 0.30, rng)),

        ("sine 1000 Hz, amplitude 0.50 (mid-band, moderate level)",
         make_sine_frame(1000.0, 0.50, rng)),

        ("sine 440 Hz + AWGN, sine_amp=0.25 noise_amp=0.05 (realistic speech+noise)",
         make_sine_plus_noise_frame(440.0, 0.25, 0.05, rng)),

        ("white noise, amplitude 0.10 (quiet background noise)",
         make_noise_frame(0.10, rng)),

        ("all +32767 (max positive stress test)",
         make_constant_frame(32767)),

        ("all -32768 (max negative stress test)",
         make_constant_frame(-32768)),

        ("alternating +32767 / -32768 (worst-case accumulator stress test)",
         make_alternating_frame()),
    ]
    return frames


# ---------------------------------------------------------------------------
# C header generation
# ---------------------------------------------------------------------------

def format_int16_array(data: np.ndarray, values_per_line: int = 16) -> str:
    """Format a 1-D int16 numpy array as a C initializer list."""
    values = [str(int(v)) for v in data]
    lines = []
    for i in range(0, len(values), values_per_line):
        chunk = values[i:i + values_per_line]
        lines.append("        " + ", ".join(chunk))
    return ",\n".join(lines)


def generate_header(frames: list[tuple[str, np.ndarray]]) -> str:
    n = len(frames)
    frame_size = FRAME_SIZE

    # Compute reference energies using the benchmark golden reference.
    energies: list[int] = []
    for desc, frame in frames:
        e = short_time_energy_i16(frame.reshape(1, -1))  # shape [1, N] -> [1]
        energies.append(int(e[0]))

    # Header comment block
    lines = [
        "/*",
        " * test_vectors.h",
        " *",
        " * AUTO-GENERATED by embedded/proto_energy/ref/export_testvec.py",
        " * DO NOT EDIT MANUALLY.",
        " *",
        f" * Frame size : {frame_size} samples (32 ms at {SR} Hz)",
        f" * Num frames : {n}",
        " *",
        " * Reference energies computed by vad.features.short_time_energy_i16().",
        " * Formula: energy = sum(int32(x[i])^2)  for i in [0, N-1],  accumulated in int64.",
        " */",
        "",
        "#ifndef TEST_VECTORS_H",
        "#define TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define TV_N_FRAMES   {n}",
        f"#define TV_FRAME_SIZE {frame_size}",
        "",
    ]

    # Per-frame descriptions as comments
    lines.append("/* Frame index descriptions:")
    for i, (desc, _) in enumerate(frames):
        lines.append(f" *   [{i:2d}]  {desc}")
    lines.append(" */")
    lines.append("")

    # Frame data array
    lines.append("static const int16_t tv_frames[TV_N_FRAMES][TV_FRAME_SIZE] = {")
    for i, (desc, frame) in enumerate(frames):
        lines.append(f"    /* [{i:2d}] {desc} */")
        lines.append("    {")
        lines.append(format_int16_array(frame))
        comma = "," if i < n - 1 else ""
        lines.append(f"    }}{comma}")
        lines.append("")
    lines.append("};")
    lines.append("")

    # Reference energy array
    lines.append("/* Reference frame energies: sum(int32(x[i])^2) in int64 */")
    lines.append("static const int64_t tv_energy_ref[TV_N_FRAMES] = {")
    for i, (e, (desc, _)) in enumerate(zip(energies, frames)):
        comma = "," if i < n - 1 else ""
        lines.append(f"    {e}LL{comma}  /* [{i:2d}] {desc} */")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* TEST_VECTORS_H */")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    frames = build_test_frames(rng)
    header = generate_header(frames)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)

    print(f"[ok] wrote {OUTPUT_FILE}")
    print(f"     {len(frames)} frames, frame_size={FRAME_SIZE}")

    # Print a summary of energies for quick visual sanity check
    print()
    print("  idx  energy_ref                 description")
    print("  ---  -------------------------  -----------")
    for i, (desc, frame) in enumerate(frames):
        e = short_time_energy_i16(frame.reshape(1, -1))[0]
        print(f"  [{i:2d}]  {int(e):>25d}  {desc}")


if __name__ == "__main__":
    main()