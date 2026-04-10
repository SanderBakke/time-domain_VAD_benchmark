/*
 * energy_kernel.h
 *
 * Frame-level energy kernel for embedded VAD prototype.
 *
 * Computes:
 *   energy = sum( (int32_t)frame[i] * (int32_t)frame[i] )   for i in [0, n-1]
 *
 * Accumulator is int64 to avoid overflow even for worst-case int16 inputs.
 *
 * Worst-case single-sample contribution:  32768^2 = 1,073,741,824  (fits in int32)
 * Worst-case full-frame (512 samples):    512 * 32768^2 = 549,755,813,888  (fits in int64)
 *
 * Input:
 *   frame  - pointer to int16_t samples for one frame
 *   n      - number of samples in the frame (e.g. 512 for 32 ms at 16 kHz)
 *
 * Output:
 *   int64_t energy value (>= 0)
 *
 * This function has no side effects and no internal state.
 * It is safe to call repeatedly on successive frames.
 */

#ifndef ENERGY_KERNEL_H
#define ENERGY_KERNEL_H

#include <stdint.h>

int64_t vad_energy_i16(const int16_t *frame, int n);

#endif /* ENERGY_KERNEL_H */