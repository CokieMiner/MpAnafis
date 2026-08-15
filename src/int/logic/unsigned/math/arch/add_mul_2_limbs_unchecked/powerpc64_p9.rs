//! `PowerPC64` (POWER9 / POWER10 ISA 3.0+) fused dual-row multiply-add kernel.
//!
//! Uses 3-operand hardware fused multiply-add instructions (`maddld`/`maddhdu`),
//! eliminating separate addition stages and store-to-load forwarding stalls.

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for PowerPC 64-bit (POWER9 ISA 3.0).
///
/// Computes:
///
/// ```text
///   dst[0..len] += src[0..len] * s0 + c0
///   dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// # Microarchitectural Strategy
///
/// POWER9's `maddld` and `maddhdu` compute $a \times b + c$ into 64-bit low and high halves respectively.
/// The kernel fuses the destination limb additions directly into the multiplication step, chains carries
/// via `addc`/`addze`, and carries forward `dst[j+1]` in `{d_cur}` across hardware `CTR` iterations.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical inner loop for 2-row multi-precision Karatsuba and basecase multiplication"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`) remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmpldi {len}, 0",                           // Compare length with 0
            "beq 2f",                                    // If len == 0, skip to end (2f)
            "ld {d_cur}, 0({dst})",                      // Prime pipeline: load dst[0]
            "mtctr {len}",                               // Load loop counter into hardware CTR register

            ".p2align 4",
            // Main dual-row accumulation loop
            "1:",
            "ld {s}, 0({src})",                          // Load src[j]
            "ld {d_next}, 8({dst})",                     // Pre-load dst[j+1] for row 1 accumulation

            // [Row 0 Fused MAC: (s * s0 + d_cur)]
            "maddld {t0}, {s}, {s0}, {d_cur}",           // t0 = (src[j] * s0 + d_cur).lo
            "maddhdu {hi0}, {s}, {s0}, {d_cur}",         // hi0 = (src[j] * s0 + d_cur).hi
            "addc {d_cur}, {t0}, {c0}",                  // d_cur = t0 + c0, set CA bit in XER
            "addze {c0}, {hi0}",                         // c0 = hi0 + CA bit (row 0 carry)
            "std {d_cur}, 0({dst})",                     // Store finalized dst[j]

            // [Row 1 Fused MAC: (s * s1 + d_next)]
            "maddld {t1}, {s}, {s1}, {d_next}",          // t1 = (src[j] * s1 + d_next).lo
            "maddhdu {hi1}, {s}, {s1}, {d_next}",        // hi1 = (src[j] * s1 + d_next).hi
            "addc {d_cur}, {t1}, {c1}",                  // d_cur = t1 + c1, set CA bit in XER
            "addze {c1}, {hi1}",                         // c1 = hi1 + CA bit (row 1 carry)

            // Advance pointers and loop via CTR
            "addi {src}, {src}, 8",                      // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                      // Advance dst pointer by 8 bytes
            "bdnz 1b",                                   // Decrement CTR and branch if CTR != 0

            // [Final Store: Write high accumulated limb dst[len]]
            "std {d_cur}, 0({dst})",                     // Flush carry-forwarded dst[len]

            // Completion
            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d_cur = out(reg) _,
            d_next = out(reg) _,
            t0 = out(reg) _,
            hi0 = out(reg) _,
            t1 = out(reg) _,
            hi1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (c0, c1)
}
