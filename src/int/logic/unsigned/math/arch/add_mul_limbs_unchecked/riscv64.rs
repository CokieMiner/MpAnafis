//! RISC-V 64-bit (RV64GC / RV64IM) fused multiply-add limb kernel.
//!
//! Uses 64×64→128-bit unsigned multipliers (`mul`/`mulhu`) and branchless carry
//! capture using `sltu` (set less than unsigned) for explicit overflow tracking.

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
/// RISC-V features an orthogonal, flag-free instruction set. 128-bit products are generated
/// using dual `mul` (low 64 bits) and `mulhu` (high 64 bits). Multi-precision carry propagation
/// is computed branchlessly using `sltu` (Set Less Than Unsigned): an addition $A + B = S$
/// overflows modulo $2^{64}$ if and only if $S < A$. The loop is 4-way unrolled (32 bytes
/// per iteration), minimizing pipeline bubbles on out-of-order RV64 cores (e.g. `SiFive` U74/P550).
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
    reason = "Critical for peak assembly performance in 64-bit RISC-V multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry_in: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder loop (2f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0 Multiply-Accumulate]
            "ld {s0}, 0({src})",                         // Load src[0]
            "ld {d0}, 0({dst})",                         // Load dst[0]
            "mul {p_lo0}, {s0}, {scalar}",               // Low 64 bits of src[0] * scalar
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 64 bits of src[0] * scalar
            "add {t_lo}, {p_lo0}, {carry_in}",           // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = (t_lo < p_lo0) ? 1 : 0
            "add {p_hi0}, {p_hi0}, {ca}",                // p_hi0 += ca
            "add {t0}, {t_lo}, {d0}",                    // t0 = t_lo + dst[0]
            "sltu {cb}, {t0}, {d0}",                     // cb = (t0 < dst[0]) ? 1 : 0
            "add {carry_in}, {p_hi0}, {cb}",             // carry_in = p_hi0 + cb
            "sd {t0}, 0({dst})",                         // Store accumulated result to dst[0]

            // [Limb 1 Multiply-Accumulate]
            "ld {s1}, 8({src})",                         // Load src[1]
            "ld {d1}, 8({dst})",                         // Load dst[1]
            "mul {p_lo1}, {s1}, {scalar}",               // Low 64 bits of src[1] * scalar
            "mulhu {p_hi1}, {s1}, {scalar}",             // High 64 bits of src[1] * scalar
            "add {t_lo}, {p_lo1}, {carry_in}",           // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if carry occurred
            "add {p_hi1}, {p_hi1}, {ca}",                // p_hi1 += ca
            "add {t0}, {t_lo}, {d1}",                    // t0 = t_lo + dst[1]
            "sltu {cb}, {t0}, {d1}",                     // cb = 1 if destination carry occurred
            "add {carry_in}, {p_hi1}, {cb}",             // Update running carry
            "sd {t0}, 8({dst})",                         // Store to dst[1]

            // [Limb 2 Multiply-Accumulate]
            "ld {s0}, 16({src})",                        // Load src[2]
            "ld {d0}, 16({dst})",                        // Load dst[2]
            "mul {p_lo0}, {s0}, {scalar}",               // Low 64 bits of src[2] * scalar
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 64 bits of src[2] * scalar
            "add {t_lo}, {p_lo0}, {carry_in}",           // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if carry occurred
            "add {p_hi0}, {p_hi0}, {ca}",                // p_hi0 += ca
            "add {t0}, {t_lo}, {d0}",                    // t0 = t_lo + dst[2]
            "sltu {cb}, {t0}, {d0}",                     // cb = 1 if destination carry occurred
            "add {carry_in}, {p_hi0}, {cb}",             // Update running carry
            "sd {t0}, 16({dst})",                        // Store to dst[2]

            // [Limb 3 Multiply-Accumulate]
            "ld {s1}, 24({src})",                        // Load src[3]
            "ld {d1}, 24({dst})",                        // Load dst[3]
            "mul {p_lo1}, {s1}, {scalar}",               // Low 64 bits of src[3] * scalar
            "mulhu {p_hi1}, {s1}, {scalar}",             // High 64 bits of src[3] * scalar
            "add {t_lo}, {p_lo1}, {carry_in}",           // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if carry occurred
            "add {p_hi1}, {p_hi1}, {ca}",                // p_hi1 += ca
            "add {t0}, {t_lo}, {d1}",                    // t0 = t_lo + dst[3]
            "sltu {cb}, {t0}, {d1}",                     // cb = 1 if destination carry occurred
            "add {carry_in}, {p_hi1}, {cb}",             // Update running carry
            "sd {t0}, 24({dst})",                        // Store to dst[3]

            "addi {src}, {src}, 32",                     // Advance src pointer by 32 bytes
            "addi {dst}, {dst}, 32",                     // Advance dst pointer by 32 bytes
            "addi {chunks}, {chunks}, -1",               // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder limbs processing (0 to 3 limbs)
            "2:",                                        // Remainder entry label
            "beqz {rem}, 4f",                            // If rem == 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "ld {s0}, 0({src})",                         // Load single src limb
            "ld {d0}, 0({dst})",                         // Load single dst limb
            "mul {p_lo0}, {s0}, {scalar}",               // Low 64-bit product
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 64-bit product
            "add {t_lo}, {p_lo0}, {carry_in}",           // Add running carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry
            "add {p_hi0}, {p_hi0}, {ca}",                // Propagate carry
            "add {t0}, {t_lo}, {d0}",                    // Accumulate into destination limb
            "sltu {cb}, {t0}, {d0}",                     // Detect destination carry
            "add {carry_in}, {p_hi0}, {cb}",             // Update running carry
            "sd {t0}, 0({dst})",                         // Store updated limb
            "addi {src}, {src}, 8",                      // Advance src (+8)
            "addi {dst}, {dst}, 8",                      // Advance dst (+8)
            "addi {rem}, {rem}, -1",                     // Decrement remainder
            "bnez {rem}, 3b",                            // Repeat while rem != 0

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
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
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
