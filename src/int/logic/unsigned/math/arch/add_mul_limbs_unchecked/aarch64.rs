//! `AArch64` (ARMv8-A / ARMv9-A) fused multiply-add limb kernel.
//!
//! Uses 64-bit paired memory loads (`ldp`), 64×64→128-bit unsigned multipliers (`mul`/`umulh`),
//! carry-chain accumulators (`adds`/`adc`), and paired stores (`stp`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// `AArch64` execution pipelines (e.g. Apple Silicon M1-M4, ARM Cortex-X, Neoverse V1/V2):
/// - Quad-issue memory pipeline loads 4 source limbs and 4 destination limbs using dual `ldp` pairs.
/// - Dual multiply execution pipes compute `mul` (low product) and `umulh` (high product) concurrently.
/// - Carry chains are accumulated via `adds` and folded into high limbs with `adc ..., xzr`.
/// - Dual `stp` instructions stream accumulated results back through store buffers.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized limbs.
/// - `src` must point to a readable buffer of at least `len` initialized limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Critical for peak assembly performance in multi-precision hot paths; 4-way unrolled"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cbz {chunks}, 1f",                          // If chunks == 0, skip directly to remainder loop (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            "ldp {src_val0}, {src_val1}, [{src}], #16",  // Load src[0..2] and advance src pointer by 16 bytes
            "ldp {src_val2}, {src_val3}, [{src}], #16",  // Load src[2..4] and advance src pointer by 16 bytes
            "ldp {dst_val0}, {dst_val1}, [{dst}]",       // Load dst[0..2]
            "ldp {dst_val2}, {dst_val3}, [{dst}, #16]",  // Load dst[2..4]

            // [Limb 0 Multiply-Accumulate]
            "mul {p_lo0}, {src_val0}, {scalar}",         // p_lo0 = low 64 bits of (src[0] * scalar)
            "umulh {p_hi0}, {src_val0}, {scalar}",       // p_hi0 = high 64 bits of (src[0] * scalar)
            "adds {p_lo0}, {p_lo0}, {carry}",            // p_lo0 += carry, set C flag
            "adc {p_hi0}, {p_hi0}, xzr",                 // p_hi0 += C flag + 0
            "adds {dst_val0}, {dst_val0}, {p_lo0}",      // dst_val0 += p_lo0, set C flag
            "adc {carry}, {p_hi0}, xzr",                 // carry = p_hi0 + C flag

            // [Limb 1 Multiply-Accumulate]
            "mul {p_lo1}, {src_val1}, {scalar}",         // p_lo1 = low 64 bits of (src[1] * scalar)
            "umulh {p_hi1}, {src_val1}, {scalar}",       // p_hi1 = high 64 bits of (src[1] * scalar)
            "adds {p_lo1}, {p_lo1}, {carry}",            // p_lo1 += carry, set C flag
            "adc {p_hi1}, {p_hi1}, xzr",                 // p_hi1 += C flag + 0
            "adds {dst_val1}, {dst_val1}, {p_lo1}",      // dst_val1 += p_lo1, set C flag
            "adc {carry}, {p_hi1}, xzr",                 // carry = p_hi1 + C flag

            // [Limb 2 Multiply-Accumulate]
            "mul {p_lo0}, {src_val2}, {scalar}",         // p_lo0 = low 64 bits of (src[2] * scalar)
            "umulh {p_hi0}, {src_val2}, {scalar}",       // p_hi0 = high 64 bits of (src[2] * scalar)
            "adds {p_lo0}, {p_lo0}, {carry}",            // p_lo0 += carry, set C flag
            "adc {p_hi0}, {p_hi0}, xzr",                 // p_hi0 += C flag + 0
            "adds {dst_val2}, {dst_val2}, {p_lo0}",      // dst_val2 += p_lo0, set C flag
            "adc {carry}, {p_hi0}, xzr",                 // carry = p_hi0 + C flag

            // [Limb 3 Multiply-Accumulate]
            "mul {p_lo1}, {src_val3}, {scalar}",         // p_lo1 = low 64 bits of (src[3] * scalar)
            "umulh {p_hi1}, {src_val3}, {scalar}",       // p_hi1 = high 64 bits of (src[3] * scalar)
            "adds {p_lo1}, {p_lo1}, {carry}",            // p_lo1 += carry, set C flag
            "adc {p_hi1}, {p_hi1}, xzr",                 // p_hi1 += C flag + 0
            "adds {dst_val3}, {dst_val3}, {p_lo1}",      // dst_val3 += p_lo1, set C flag
            "adc {carry}, {p_hi1}, xzr",                 // carry = p_hi1 + C flag

            // [Paired Vectorized Memory Stores]
            "stp {dst_val0}, {dst_val1}, [{dst}], #16", // Store dst[0..2] and advance dst pointer by 16 bytes
            "stp {dst_val2}, {dst_val3}, [{dst}], #16", // Store dst[2..4] and advance dst pointer by 16 bytes
            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "b.ne 2b",                                   // Repeat while chunks != 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "cbz {rem}, 3f",                             // If rem == 0, skip to completion (3f)

            // 1-limb unrolled tail loop
            "4:",                                        // Tail loop label
            "ldr {src_val0}, [{src}], #8",               // Load single src limb and advance pointer by 8 bytes
            "ldr {dst_val0}, [{dst}]",                   // Load single dst limb
            "mul {p_lo0}, {src_val0}, {scalar}",         // Low 64-bit product
            "umulh {p_hi0}, {src_val0}, {scalar}",       // High 64-bit product
            "adds {p_lo0}, {p_lo0}, {carry}",            // Add incoming carry
            "adc {p_hi0}, {p_hi0}, xzr",                 // Propagate carry bit
            "adds {dst_val0}, {dst_val0}, {p_lo0}",      // Accumulate into destination limb
            "adc {carry}, {p_hi0}, xzr",                 // Update carry out
            "str {dst_val0}, [{dst}], #8",               // Store updated limb and advance pointer by 8 bytes
            "subs {rem}, {rem}, #1",                     // Decrement remainder counter
            "b.ne 4b",                                   // Repeat while rem != 0

            // Tail completion
            "3:",                                        // Completion label
            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            src_val2 = out(reg) _,
            src_val3 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            dst_val2 = out(reg) _,
            dst_val3 = out(reg) _,
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            options(nostack)
        );
    }
    carry
}
