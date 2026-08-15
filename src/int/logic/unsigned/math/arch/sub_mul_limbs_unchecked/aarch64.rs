//! `AArch64` (ARMv8-A / ARMv9-A) fused multiply-subtract limb kernel.
//!
//! Uses 64×64→128-bit unsigned multipliers (`mul`/`umulh`), paired memory operations
//! (`ldp`/`stp`), and condition-code borrow extraction (`subs`/`cset cc`/`orr`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// Multi-precision multiply-subtract combines high-word carry tracking and multi-precision subtraction
/// borrows. The kernel is 2-way unrolled (16 bytes per iteration), utilizing dual `ldp` pairs to load
/// source and destination limbs concurrently. 128-bit products are assembled with `mul`/`umulh`, and
/// borrows are captured with `cset cc` (Condition Set if Carry Clear / Borrow).
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 64-bit ARM multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb = 0;
    let mut borrow: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cbz {chunks}, 1f",                          // If chunks == 0, skip to remainder loop (1f)

            // Main 2-way unrolled loop body
            "2:",

            // [Paired Vectorized Memory Loads]
            "ldp {src_val0}, {src_val1}, [{src}], #16",  // Load src[0..2] and advance pointer by 16 bytes
            "ldp {dst_val0}, {dst_val1}, [{dst}]",        // Load dst[0..2]

            // [Limb 0 Multiply-Subtract]
            "mul {p_lo0}, {src_val0}, {scalar}",          // p_lo0 = low 64 bits of src[0] * scalar
            "umulh {p_hi0}, {src_val0}, {scalar}",        // p_hi0 = high 64 bits of src[0] * scalar
            "adds {p_lo0}, {p_lo0}, {carry_hi}",          // p_lo0 += carry_hi, set C flag
            "adc {p_hi0}, {p_hi0}, xzr",                  // p_hi0 += C flag + 0
            "subs {dst_val0}, {dst_val0}, {p_lo0}",       // dst[0] -= p_lo0, set flags (C=0 if borrow)
            "cset {b1}, cc",                              // b1 = 1 if borrow occurred, else 0
            "subs {dst_val0}, {dst_val0}, {borrow}",      // dst[0] -= incoming borrow
            "cset {b2}, cc",                              // b2 = 1 if second borrow occurred, else 0
            "orr {borrow}, {b1}, {b2}",                   // borrow = b1 | b2 (combined borrow bit)
            "mov {carry_hi}, {p_hi0}",                    // Update running carry_hi

            // [Limb 1 Multiply-Subtract]
            "mul {p_lo1}, {src_val1}, {scalar}",          // Low 64 bits of src[1] * scalar
            "umulh {p_hi1}, {src_val1}, {scalar}",        // High 64 bits of src[1] * scalar
            "adds {p_lo1}, {p_lo1}, {carry_hi}",          // p_lo1 += carry_hi
            "adc {p_hi1}, {p_hi1}, xzr",                  // p_hi1 += C flag + 0
            "subs {dst_val1}, {dst_val1}, {p_lo1}",       // dst[1] -= p_lo1
            "cset {b1}, cc",                              // b1 = 1 if borrow occurred
            "subs {dst_val1}, {dst_val1}, {borrow}",      // dst[1] -= incoming borrow
            "cset {b2}, cc",                              // b2 = 1 if borrow occurred
            "orr {borrow}, {b1}, {b2}",                   // Combined borrow bit
            "mov {carry_hi}, {p_hi1}",                    // Update carry_hi

            // [Paired Vectorized Memory Store]
            "stp {dst_val0}, {dst_val1}, [{dst}], #16",  // Store updated limbs and advance dst pointer
            "sub {chunks}, {chunks}, #1",                 // Decrement chunk counter
            "cbnz {chunks}, 2b",                          // Loop if chunks != 0

            // Remainder processing (0 or 1 limb)
            "1:",
            "cbz {rem}, 3f",                              // If rem == 0, skip to end (3f)

            // 1-limb tail
            "ldr {src_val0}, [{src}], #8",                // Load single src limb
            "ldr {dst_val0}, [{dst}]",                    // Load single dst limb
            "mul {p_lo0}, {src_val0}, {scalar}",          // Low 64 bits
            "umulh {p_hi0}, {src_val0}, {scalar}",        // High 64 bits
            "adds {p_lo0}, {p_lo0}, {carry_hi}",          // Add carry_hi
            "adc {p_hi0}, {p_hi0}, xzr",                  // Propagate carry
            "subs {dst_val0}, {dst_val0}, {p_lo0}",       // Subtract product
            "cset {b1}, cc",                              // Capture borrow
            "subs {dst_val0}, {dst_val0}, {borrow}",      // Subtract previous borrow
            "cset {b2}, cc",                              // Capture borrow
            "orr {borrow}, {b1}, {b2}",                   // Combine borrow bits
            "mov {carry_hi}, {p_hi0}",                    // Update carry_hi
            "str {dst_val0}, [{dst}], #8",                // Store updated limb

            // Tail completion
            "3:",

            carry_hi = inout(reg) carry_hi,
            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            b1 = out(reg) _,
            b2 = out(reg) _,
            options(nostack)
        );
    }
    (carry_hi, borrow)
}
