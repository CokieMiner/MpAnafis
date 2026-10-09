//! `AArch64` (ARMv8-A / ARMv9-A) write-only dual-row multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! paired memory operations (`ldp`/`stp`) and two-limb unrolling.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] x (s0 + s1 x 2^64)
/// ```
///
/// Evaluates two simultaneous multiplication rows in registers without memory reads of `dst`.
/// The loop is 2-way unrolled, hoists all 8 `mul`/`umulh` multiplications upfront across dual
/// both rows, merges the carry chains, and writes destination limbs
/// in 128-bit paired stores (`stp`).
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
#[expect(
    clippy::inline_always,
    reason = "inlining joins the two-row initializer to complete basecase multiplication"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    let carry0: Limb = 0;
    let carry1: Limb = 0;
    let pending1: Limb = 0;
    let chunks = len >> 1;

    // SAFETY: disjoint aligned spans provide len initialized inputs and len+2
    // writable outputs. Pair blocks and the odd tail consume exactly len
    // inputs before the final two stores. Products plus two limbs are <= B^2-1;
    // the final high carry is <= s1. Destination values are kept in registers
    // until stored, so uninitialized output is never read. All clobbers are outputs.
    unsafe {
        asm!(
            "cbz {chunks}, 2f",                          // If chunks == 0, skip to remainder (2f)

            // Main 2-way unrolled loop body
            "1:",
            "ldp {v0}, {v1}, [{src}], #16",              // Paired load: v0 = src[0], v1 = src[1]

            // Four products split into eight low/high multiplication instructions.
            "mul {lo0_a}, {v0}, {s0}",                   // Limb A: low product row 0
            "mul {lo1_a}, {v0}, {s1}",                   // Limb A: low product row 1
            "umulh {hi0_a}, {v0}, {s0}",                 // Limb A: high product row 0
            "umulh {hi1_a}, {v0}, {s1}",                 // Limb A: high product row 1

            "mul {lo0_b}, {v1}, {s0}",                   // Limb B: low product row 0
            "mul {lo1_b}, {v1}, {s1}",                   // Limb B: low product row 1
            "umulh {hi0_b}, {v1}, {s0}",                 // Limb B: high product row 0
            "umulh {hi1_b}, {v1}, {s1}",                 // Limb B: high product row 1

            // [Limb A Accumulation & Merging]
            "adds {lo0_a}, {lo0_a}, {carry0}",           // lo0_a += carry0, set C flag
            "adc {hi0_a}, {hi0_a}, xzr",                 // hi0_a += C flag + 0
            "adds {out0}, {lo0_a}, {pending1}",          // out0 = lo0_a + pending1 (final dst limb A)
            "adc {carry0}, {hi0_a}, xzr",                // carry0 = hi0_a + C flag
            "adds {pending1}, {lo1_a}, {carry1}",        // pending1 = lo1_a + carry1
            "adc {carry1}, {hi1_a}, xzr",                // carry1 = hi1_a + C flag

            // [Limb B Accumulation & Merging]
            "adds {lo0_b}, {lo0_b}, {carry0}",           // lo0_b += carry0
            "adc {hi0_b}, {hi0_b}, xzr",                 // hi0_b += C flag
            "adds {out1}, {lo0_b}, {pending1}",          // out1 = lo0_b + pending1 (final dst limb B)
            "adc {carry0}, {hi0_b}, xzr",                // carry0 = hi0_b + C flag
            "adds {pending1}, {lo1_b}, {carry1}",        // pending1 = lo1_b + carry1
            "adc {carry1}, {hi1_b}, xzr",                // carry1 = hi1_b + C flag

            // [Paired Store: Write 2 finalized limbs to destination]
            "stp {out0}, {out1}, [{dst}], #16",          // Store dst[0..2] and advance pointer by 16 bytes

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "b.ne 1b",                                   // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",
            "tbz {len}, #0, 3f",                         // If len is even, skip remainder (3f)

            // 1-limb tail
            "ldr {v0}, [{src}], #8",                     // Load single src limb
            "mul {lo0_a}, {v0}, {s0}",                   // Low product row 0
            "mul {lo1_a}, {v0}, {s1}",                   // Low product row 1
            "umulh {hi0_a}, {v0}, {s0}",                 // High product row 0
            "umulh {hi1_a}, {v0}, {s1}",                 // High product row 1

            "adds {lo0_a}, {lo0_a}, {carry0}",           // Add row 0 carry
            "adc {hi0_a}, {hi0_a}, xzr",                 // Propagate carry
            "adds {out0}, {lo0_a}, {pending1}",          // Merge with previous pending limb
            "adc {carry0}, {hi0_a}, xzr",                // Update carry0
            "adds {pending1}, {lo1_a}, {carry1}",        // Update pending1
            "adc {carry1}, {hi1_a}, xzr",                // Update carry1
            "str {out0}, [{dst}], #8",                   // Store single finalized limb

            // Epilogue: Flush trailing high row 1 limb + remaining carry
            "3:",
            "adds {pending1}, {pending1}, {carry0}",     // pending1 += carry0
            "adc {carry1}, {carry1}, xzr",               // carry1 += C flag
            "stp {pending1}, {carry1}, [{dst}]",         // Store final 2 limbs (dst[len..len+2])

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = in(reg) len,
            chunks = inout(reg) chunks => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            v0 = out(reg) _,
            v1 = out(reg) _,
            out0 = out(reg) _,
            out1 = out(reg) _,
            lo0_a = out(reg) _,
            hi0_a = out(reg) _,
            lo1_a = out(reg) _,
            hi1_a = out(reg) _,
            lo0_b = out(reg) _,
            hi0_b = out(reg) _,
            lo1_b = out(reg) _,
            hi1_b = out(reg) _,
            options(nostack)
        );
    }
}
