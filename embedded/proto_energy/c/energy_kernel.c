/*
 * energy_kernel.c
 *
 * Frame-level energy kernel for embedded VAD prototype.
 *
 * See energy_kernel.h for full documentation.
 *
 * Implementation notes:
 *   - frame[i] is int16.  Multiplying two int16 values can overflow int16
 *     (e.g. -32768 * -32768 = 1,073,741,824, which exceeds INT16_MAX = 32767).
 *   - We widen each sample to int32 before multiplying.  The product of two
 *     int16 values fits in int32 (max: 32768^2 = 1,073,741,824 < INT32_MAX).
 *   - The accumulator is int64 to handle up to 512 * 32768^2 = 549,755,813,888
 *     without overflow.
 *
 * This mirrors vad.features.short_time_energy_i16() in the Python benchmark:
 *     x = frames.astype(np.int32)
 *     e = (x * x).sum(axis=1, dtype=np.int64)
 */

#include "energy_kernel.h"

int64_t vad_energy_i16(const int16_t *frame, int n)
{
    int64_t acc = 0;
    int i;

    for (i = 0; i < n; i++) {
        int32_t s = (int32_t)frame[i];  /* widen to int32 before multiply */
        acc += (int64_t)(s * s);        /* s*s fits in int32; cast to int64 before accumulating */
    }

    return acc;
}