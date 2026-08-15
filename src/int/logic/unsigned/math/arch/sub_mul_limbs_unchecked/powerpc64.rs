//! `PowerPC64` (POWER8 / POWER9 / POWER10) fused multiply-subtract limb kernel.
//!
//! Uses 64-bit dual-issue multipliers (`mulld`/`mulhdu`), addition with carry (`addc`/`addze`)
//! for the high product row, and subtraction with borrow (`subfic`/`subfe`) for destination updates.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// PowerPC models borrow inversion using `subfic` (Subtract from Immediate Carrying) and `subfe`
/// (Subtract from Extended). In PowerPC, `CA = 1` indicates *no borrow*, while `CA = 0` indicates
/// *borrow occurred*. This kernel hoists the 8 independent `mulld`/`mulhdu` multipliers across 4 limbs,
/// chains `addc`/`addze` to assemble product limbs, and then executes `subfic`/`subfe` to subtract from
/// destination memory across hardware CTR loop iterations.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Critical for peak assembly performance in 64-bit PowerPC multi-precision hot paths; 4-way unrolled"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb = 0;
    let mut borrow_reg: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmpldi {chunks}, 0",         // Compare chunks count with 0
            "beq 1f",                     // If chunks == 0, skip to remainder loop (1f)
            "mtctr {chunks}",             // Load loop counter into hardware CTR register
            ".p2align 4",

            // Main 4-way unrolled loop body
            "2:",

            // [Load 4 Source and 4 Destination Limbs]
            "ld {src_v0}, 0({src})",      // Load src[0]
            "ld {src_v1}, 8({src})",      // Load src[1]
            "ld {src_v2}, 16({src})",     // Load src[2]
            "ld {src_v3}, 24({src})",     // Load src[3]
            "ld {dst_v0}, 0({dst})",      // Load dst[0]
            "ld {dst_v1}, 8({dst})",      // Load dst[1]
            "ld {dst_v2}, 16({dst})",     // Load dst[2]
            "ld {dst_v3}, 24({dst})",     // Load dst[3]

            // [Hoisted Superscalar Multipliers: 4 independent 64x64->128 product pairs]
            "mulld {p_lo0}, {src_v0}, {scalar}",          // Low 64 bits of src[0] * scalar
            "mulhdu {p_hi0}, {src_v0}, {scalar}",         // High 64 bits of src[0] * scalar
            "mulld {p_lo1}, {src_v1}, {scalar}",          // Low 64 bits of src[1] * scalar
            "mulhdu {p_hi1}, {src_v1}, {scalar}",         // High 64 bits of src[1] * scalar
            "mulld {p_lo2}, {src_v2}, {scalar}",          // Low 64 bits of src[2] * scalar
            "mulhdu {p_hi2}, {src_v2}, {scalar}",         // High 64 bits of src[2] * scalar
            "mulld {p_lo3}, {src_v3}, {scalar}",          // Low 64 bits of src[3] * scalar
            "mulhdu {p_hi3}, {src_v3}, {scalar}",         // High 64 bits of src[3] * scalar

            // [Limb 0 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo0}, {p_lo0}, {carry_hi}",          // p_lo0 += carry_hi, set CA
            "addze {carry_hi}, {p_hi0}",                  // carry_hi = p_hi0 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",          // dst_v0 = dst_v0 - p_lo0 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "std {dst_v0}, 0({dst})",                     // Store updated dst[0]

            // [Limb 1 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo1}, {p_lo1}, {carry_hi}",          // p_lo1 += carry_hi
            "addze {carry_hi}, {p_hi1}",                  // carry_hi = p_hi1 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v1}, {p_lo1}, {dst_v1}",          // dst_v1 = dst_v1 - p_lo1 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "std {dst_v1}, 8({dst})",                     // Store updated dst[1]

            // [Limb 2 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo2}, {p_lo2}, {carry_hi}",          // p_lo2 += carry_hi
            "addze {carry_hi}, {p_hi2}",                  // carry_hi = p_hi2 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v2}, {p_lo2}, {dst_v2}",          // dst_v2 = dst_v2 - p_lo2 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "std {dst_v2}, 16({dst})",                    // Store updated dst[2]

            // [Limb 3 Multiply-Carry & Subtraction-Borrow]
            "addc {p_lo3}, {p_lo3}, {carry_hi}",          // p_lo3 += carry_hi
            "addze {carry_hi}, {p_hi3}",                  // carry_hi = p_hi3 + CA
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v3}, {p_lo3}, {dst_v3}",          // dst_v3 = dst_v3 - p_lo3 - borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Capture new borrow mask
            "std {dst_v3}, 24({dst})",                    // Store updated dst[3]

            // Advance pointers by 4 limbs (32 bytes) and loop via CTR
            "addi {src}, {src}, 32",
            "addi {dst}, {dst}, 32",
            "bdnz 2b",                                    // Decrement CTR and loop if CTR != 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",
            "cmpldi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",                                // Load remainder count into CTR

            // 1-limb unrolled tail loop
            "4:",
            "ld {src_v0}, 0({src})",                      // Load single src limb
            "ld {dst_v0}, 0({dst})",                      // Load single dst limb
            "mulld {p_lo0}, {src_v0}, {scalar}",          // Low 64-bit product
            "mulhdu {p_hi0}, {src_v0}, {scalar}",         // High 64-bit product
            "addc {p_lo0}, {p_lo0}, {carry_hi}",          // Add carry_hi
            "addze {carry_hi}, {p_hi0}",                  // Update carry_hi
            "subfic {temp}, {borrow_reg}, 0",             // Convert borrow mask to CA flag
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",          // Subtract product + borrow
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}", // Update borrow mask
            "std {dst_v0}, 0({dst})",                     // Store updated limb
            "addi {src}, {src}, 8",                       // Advance src pointer
            "addi {dst}, {dst}, 8",                       // Advance dst pointer
            "bdnz 4b",                                    // Loop if != 0

            // Tail completion
            "3:",
            "neg {borrow_reg}, {borrow_reg}",             // Convert mask (-1 -> 1, 0 -> 0)

            carry_hi = inout(reg) carry_hi,
            borrow_reg = inout(reg) borrow_reg,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _,
            src_v1 = out(reg) _,
            src_v2 = out(reg) _,
            src_v3 = out(reg) _,
            dst_v0 = out(reg) _,
            dst_v1 = out(reg) _,
            dst_v2 = out(reg) _,
            dst_v3 = out(reg) _,
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            p_lo2 = out(reg) _,
            p_hi2 = out(reg) _,
            p_lo3 = out(reg) _,
            p_hi3 = out(reg) _,
            temp = out(reg) _,
            options(nostack)
        );
    }
    (carry_hi, borrow_reg)
}
