/*
 * platform.h
 *
 * Minimal host/devkit portability shim for the energy VAD prototype.
 *
 * Isolates exactly two things that differ between host and devkit:
 *   1. Print function
 *   2. Cycle counter read
 *
 * Build for host (default — no extra flags needed):
 *   gcc ... test_energy.c energy_kernel.c
 *
 * Build for devkit:
 *   Add -DPLATFORM_DEVKIT and fill in the two TODOs below.
 */

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include <inttypes.h>  /* PRId64, PRIu64 */

/* -------------------------------------------------------------------------
 * NOTE on 64-bit printf and newlib-nano:
 *
 * Some bare-metal RISC-V toolchains default to newlib-nano (-specs=nano.specs),
 * which may lack support for 64-bit printf format specifiers (PRId64 / PRIu64).
 * Symptom: large energy values (frames 4-10) print as 0 or garbage.
 *
 * If you see this, check your linker flags. Options:
 *   a) Drop -specs=nano.specs (uses full newlib; larger binary)
 *   b) Add -u _printf_float if your BSP provides a 64-bit printf patch
 *   c) Print high/low halves manually:
 *        uint32_t hi = (uint32_t)(val >> 32), lo = (uint32_t)(val & 0xFFFFFFFF);
 *        platform_print("%lu%08lu", (unsigned long)hi, (unsigned long)lo);
 * -------------------------------------------------------------------------
 */

#if defined(PLATFORM_DEVKIT)

    /* ---- DEVKIT: fill in these two TODOs before compiling for target ---- */

    /* TODO: include your BSP UART / printf header here, for example:
     *   #include "bsp/uart.h"
     *   #include "printf.h"
     *   #include "nuclei_sdk_soc.h"
     */

    /* TODO: map platform_print to your BSP's printf-compatible function.
     * If your BSP already retargets printf to UART, the line below is correct.
     * Otherwise replace printf with e.g. uart_printf or tfp_printf.
     */
    #define platform_print  printf

    static inline uint64_t platform_read_cycles(void)
    {
#if __riscv_xlen == 64
        /* RV64: rdcycle reads the full 64-bit counter in one instruction. */
        uint64_t c;
        __asm__ volatile ("rdcycle %0" : "=r"(c));
        return c;
#else
        /* RV32: 64-bit cycle counter is split across two CSRs (cycle / cycleh).
         * Read high, low, high again; retry on wrap-around. */
        uint32_t lo, hi, hi2;
        do {
            __asm__ volatile ("rdcycleh %0" : "=r"(hi));
            __asm__ volatile ("rdcycle  %0" : "=r"(lo));
            __asm__ volatile ("rdcycleh %0" : "=r"(hi2));
        } while (hi != hi2);
        return ((uint64_t)hi << 32) | lo;
#endif
    }

#else  /* PLATFORM_HOST — default when PLATFORM_DEVKIT is not defined */

    #include <stdio.h>
    #define platform_print  printf

    static inline uint64_t platform_read_cycles(void)
    {
        return 0;  /* no cycle counter on host */
    }

#endif /* PLATFORM_DEVKIT */

#endif /* PLATFORM_H */
