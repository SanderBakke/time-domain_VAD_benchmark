# embedded/proto_energy — Minimal Energy VAD Prototype

## What this is

A minimal embedded prototype for testing one specific kernel from the VAD pipeline:
**frame-level energy computation over int16 audio samples**.

This is the first step toward running a fixed-point VAD on a RISC-V devkit.
The Python benchmark in `vad/` and `scripts/` is the golden reference and is not modified here.

## What this is NOT

- Not a port of the full benchmark framework
- Not a full VAD decision system (no thresholding, no noise floor, no hangover)
- Not a clip-level pipeline
- Not a KWS system

Those steps come later, after this kernel is verified on the devkit.

## Folder structure

```
proto_energy/
├── ref/
│   └── export_testvec.py     Python script: generates test_vectors.h from the benchmark reference
├── c/
│   ├── energy_kernel.h       C header: function prototype
│   ├── energy_kernel.c       C implementation: vad_energy_i16()
│   ├── test_energy.c         C test harness: runs kernel, checks against reference
│   └── Makefile              Host build (gcc)
├── test_vectors/
│   └── test_vectors.h        Auto-generated — do not edit manually
└── README.md                 This file
```

## The kernel

```c
int64_t vad_energy_i16(const int16_t *frame, int n);
```

Computes `sum( (int32_t)frame[i]^2 )` over `n` samples, accumulated in `int64_t`.

This matches `vad.features.short_time_energy_i16()` in the Python benchmark exactly:
```python
x = frames.astype(np.int32)
e = (x * x).sum(axis=1, dtype=np.int64)
```

Parameters for this prototype:
- Sample rate: 16 kHz
- Frame size: 512 samples (32 ms)
- Hop size: 256 samples (16 ms) — not used here; each test frame is independent

## Step-by-step: how to verify

### Step 1 — Generate test vectors (run once, from the repo root)

```bash
python embedded/proto_energy/ref/export_testvec.py
```

This produces `test_vectors/test_vectors.h` with 11 test frames and their
reference energies computed by the Python benchmark's `short_time_energy_i16()`.

You will see a console summary like:
```
  idx  energy_ref                 description
  [ 0]                          0  zero / silence
  [ 1]                    5120000  constant +100 (very low amplitude)
  ...
  [10]               549739036928  alternating +32767 / -32768 (stress test)
```

### Step 2 — Build the host test (on your laptop)

```bash
cd embedded/proto_energy/c
make
```

Or without make:
```bash
cd embedded/proto_energy/c
gcc -O0 -Wall -std=c99 -I../test_vectors -o test_energy test_energy.c energy_kernel.c
```

### Step 3 — Run the host test

```bash
./test_energy          # Linux/Mac
test_energy.exe        # Windows
```

Expected output when everything is correct:
```
VAD frame-level energy kernel test
Frames: 11   Frame size: 512 samples

[PASS]  frame  0  energy=                    0  ref=                    0
[PASS]  frame  1  energy=              5120000  ref=              5120000
[PASS]  frame  2  energy=            512000000  ref=            512000000
...
[PASS]  frame 10  energy=        549739036928  ref=        549739036928

Max absolute error across all frames: 0
Result: ALL PASSED (11/11)
```

If `max absolute error = 0` and all frames pass, the C kernel is numerically
identical to the Python reference on this host.

### Step 4 — Move to the devkit

1. Cross-compile `energy_kernel.c` and `test_energy.c` for your RISC-V target.
2. Enable timing by compiling with `-DENABLE_TIMING` and implementing `read_cycles()`
   in `test_energy.c` (the placeholder is already marked with a `TODO` comment).
3. Flash and run. The PASS/FAIL output and cycle counts are printed over UART.

Example RISC-V cross-compile (adjust toolchain prefix as needed):
```bash
riscv64-unknown-elf-gcc -O0 -Wall -std=c99 -I../test_vectors \
    -DENABLE_TIMING \
    -o test_energy.elf test_energy.c energy_kernel.c
```

## Adding cycle measurement (devkit step)

In `c/test_energy.c`, find the section marked `TODO: replace with devkit cycle counter`.

For a standard RISC-V core, replace the placeholder with:
```c
static inline uint64_t read_cycles(void) {
    uint64_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}
```

Then compile with `-DENABLE_TIMING` to activate the timing output.

## What the next step is

After this prototype runs correctly on the devkit (PASS + cycle count):

1. Measure cycles per frame and compare against theoretical op-count
   (512 widening casts + 512 multiplies + 511 adds ≈ 1535 operations).
2. Add a simple fixed threshold decision on top of the energy output
   (no adaptive noise floor yet — just a static threshold for initial testing).
3. Extend to processing a short sequence of frames from a known input buffer
   (not full dataset playback — just a small in-memory buffer).
4. At that point, the noise-floor EMA and hangover logic from `EnergyVADFixed`
   in `vad/algorithms.py` can be translated to C incrementally.

## Regenerating test vectors

If you change the Python benchmark or want different test signals, just re-run:
```bash
python embedded/proto_energy/ref/export_testvec.py
```

The header is always regenerated from scratch. Rebuild the C test after regenerating.