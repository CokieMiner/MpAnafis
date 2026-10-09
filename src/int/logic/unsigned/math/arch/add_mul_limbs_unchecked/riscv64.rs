//! RISC-V 64-bit (RV64GC / RV64IM) fused multiply-add limb kernel.
//!
//! Uses 64x64->128-bit unsigned multipliers (`mul`/`mulhu`) and branchless carry
//! capture using `sltu` (set less than unsigned) for explicit overflow tracking.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Four limbs are processed per block. Unsigned addition overflows exactly
/// when its modular result is below either addend; `sltu` captures that bit.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[expect(
    clippy::inline_always,
    reason = "Keep the unrolled hardware row recurrence in the selected multiplication caller"
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

    // SAFETY: 4*chunks + rem == len bounds every aligned load and store in
    // the initialized spans. Eight-byte pointer advances stay within the live
    // allocations. Stores target only the disjoint writable destination.
    // Zero chunks and remainder skip all pointer accesses.
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
