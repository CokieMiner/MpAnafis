//! RISC-V 32-bit (RV32GC / RV32IM) fused multiply-add limb kernel.
//!
//! Uses 32×32→64-bit unsigned multipliers (`mul`/`mulhu`) and branchless carry
//! capture using `sltu` (set less than unsigned) for explicit overflow tracking.

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, add the result into `dst`,
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
/// Uses standard RV32M extension hardware instructions: `mul` (low 32 bits) and `mulhu`
/// (high 32 bits). Carry detection after addition is achieved branchlessly with `sltu`.
/// The loop is 2-way unrolled (8 bytes per iteration).
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit RISC-V multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry_in: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `4`, `8`) remain within `len * 4` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder handler (2f)

            // Main 2-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0 Multiply-Accumulate]
            "lw {s0}, 0({src})",                         // Load src[0] (32 bits)
            "lw {s1}, 4({src})",                         // Load src[1] (32 bits)
            "lw {d0}, 0({dst})",                         // Load dst[0] (32 bits)
            "lw {d1}, 4({dst})",                         // Load dst[1] (32 bits)

            "mul {p_lo0}, {s0}, {scalar}",               // Low 32 bits of src[0] * scalar
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 32 bits of src[0] * scalar
            "add {t_lo}, {p_lo0}, {carry_in}",           // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = (t_lo < p_lo0) ? 1 : 0
            "add {p_hi0}, {p_hi0}, {ca}",                // p_hi0 += ca
            "add {t0}, {t_lo}, {d0}",                    // t0 = t_lo + dst[0]
            "sltu {cb}, {t0}, {d0}",                     // cb = (t0 < dst[0]) ? 1 : 0
            "add {carry_in}, {p_hi0}, {cb}",             // carry_in = p_hi0 + cb
            "sw {t0}, 0({dst})",                         // Store accumulated result to dst[0]

            // [Limb 1 Multiply-Accumulate]
            "mul {p_lo1}, {s1}, {scalar}",               // Low 32 bits of src[1] * scalar
            "mulhu {p_hi1}, {s1}, {scalar}",             // High 32 bits of src[1] * scalar
            "add {t_lo}, {p_lo1}, {carry_in}",           // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if carry occurred
            "add {p_hi1}, {p_hi1}, {ca}",                // p_hi1 += ca
            "add {t0}, {t_lo}, {d1}",                    // t0 = t_lo + dst[1]
            "sltu {cb}, {t0}, {d1}",                     // cb = 1 if destination carry occurred
            "add {carry_in}, {p_hi1}, {cb}",             // Update running carry
            "sw {t0}, 4({dst})",                         // Store to dst[1]

            "addi {src}, {src}, 8",                      // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                      // Advance dst pointer by 8 bytes
            "addi {chunks}, {chunks}, -1",               // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",                                        // Remainder entry label
            "beqz {rem}, 4f",                            // If rem == 0, skip to completion (4f)

            // 1-limb remainder
            "3:",                                        // Tail loop label
            "lw {s0}, 0({src})",                         // Load single src limb
            "lw {d0}, 0({dst})",                         // Load single dst limb
            "mul {p_lo0}, {s0}, {scalar}",               // Low 32-bit product
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 32-bit product
            "add {t_lo}, {p_lo0}, {carry_in}",           // Add running carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry
            "add {p_hi0}, {p_hi0}, {ca}",                // Propagate carry
            "add {t0}, {t_lo}, {d0}",                    // Accumulate into destination limb
            "sltu {cb}, {t0}, {d0}",                     // Detect destination carry
            "add {carry_in}, {p_hi0}, {cb}",             // Update running carry
            "sw {t0}, 0({dst})",                         // Store updated limb

            // Tail completion
            "4:",                                        // Completion label

            carry_in = inout(reg) carry_in,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            s0 = out(reg) _,
            s1 = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo = out(reg) _,
            t0 = out(reg) _,
            ca = out(reg) _,
            cb = out(reg) _,
            options(nostack)
        );
    }
    carry_in
}
