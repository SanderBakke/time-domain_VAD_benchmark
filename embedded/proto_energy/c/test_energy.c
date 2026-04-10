/*
 * test_energy.c
 *
 * Host / devkit test harness for the VAD frame-level energy kernel.
 *
 * Runs vad_energy_i16() over all test frames defined in test_vectors.h and
 * compares the results against the Python golden reference values.
 *
 * Build for host (default):
 *   gcc -O0 -Wall -std=c99 -I../test_vectors -o test_energy test_energy.c energy_kernel.c
 *   — or: make run
 *
 * Build for devkit:
 *   Fill in the TODOs in platform.h, then:
 *   <cross-gcc> -O0 -Wall -std=c99 -I../test_vectors -DPLATFORM_DEVKIT \
 *       -o test_energy.elf test_energy.c energy_kernel.c
 *   — or: see Makefile for a template command.
 *
 * Expected output when all tests pass:
 *   [PASS]  frame  0  energy=                    0  ref=                    0
 *   [PASS]  frame  1  energy=              5120000  ref=              5120000
 *   ...
 *   Max absolute error across all frames: 0
 *   Result: ALL PASSED (11/11)
 *
 * On devkit, each passing line also shows cycles= once platform_read_cycles()
 * is implemented in platform.h. On host, cycles= is suppressed (returns 0).
 */

#include <stdint.h>
#include "platform.h"    /* platform_print, platform_read_cycles */
#include "energy_kernel.h"
#include "test_vectors.h"


int main(void)
{
    int n_pass = 0;
    int n_fail = 0;
    int64_t max_abs_err = 0;

    platform_print("VAD frame-level energy kernel test\n");
    platform_print("Frames: %d   Frame size: %d samples\n\n",
                   TV_N_FRAMES, TV_FRAME_SIZE);

    for (int i = 0; i < TV_N_FRAMES; i++) {

        uint64_t t0     = platform_read_cycles();
        int64_t  result = vad_energy_i16(tv_frames[i], TV_FRAME_SIZE);
        uint64_t t1     = platform_read_cycles();
        uint64_t cycles = t1 - t0;

        int64_t ref    = tv_energy_ref[i];
        int64_t err    = result - ref;
        int64_t abserr = err < 0 ? -err : err;

        if (abserr > max_abs_err)
            max_abs_err = abserr;

        const char *status = (abserr == 0) ? "PASS" : "FAIL";
        if (abserr == 0)
            n_pass++;
        else
            n_fail++;

        platform_print("[%s]  frame %2d  energy=%20" PRId64 "  ref=%20" PRId64,
                       status, i, result, ref);

        if (abserr != 0)
            platform_print("  ERROR=%" PRId64, err);

        /* cycles= is printed only when non-zero, so host output is unchanged.
         * On devkit, platform_read_cycles() returns real cycle counts. */
        if (cycles > 0)
            platform_print("  cycles=%" PRIu64, cycles);

        platform_print("\n");
    }

    platform_print("\nMax absolute error across all frames: %" PRId64 "\n",
                   max_abs_err);

    if (n_fail == 0)
        platform_print("Result: ALL PASSED (%d/%d)\n", n_pass, TV_N_FRAMES);
    else
        platform_print("Result: %d FAILED, %d passed (out of %d)\n",
                       n_fail, n_pass, TV_N_FRAMES);

    return (n_fail == 0) ? 0 : 1;
}
