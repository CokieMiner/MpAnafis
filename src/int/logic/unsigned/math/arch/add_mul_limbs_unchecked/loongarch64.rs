//! `LoongArch64` fused multiply-add limb kernel.
//!
//! Uses 64x64->128-bit unsigned multipliers (`mul.d`/`mulh.du`), branchless carry
//! propagation via `sltu`, and 2-way unrolling.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Two limbs are processed per block, followed by an optional scalar tail.
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
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: 2*chunks + rem == len bounds every aligned load and store in
    // the initialized spans. Eight-byte pointer advances stay within the live
    // allocations. Stores target only the disjoint writable destination.
    // Zero chunks and remainder skip all pointer accesses.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder loop (2f)

            // Main 2-way unrolled loop body
            "1:",                                        // Loop head label
            // [Load 2 Source and 2 Destination Limbs]
            "ld.d {s0}, {src}, 0",                       // Load src[0]
            "ld.d {s1}, {src}, 8",                       // Load src[1]
            "ld.d {d0}, {dst}, 0",                       // Load dst[0]
            "ld.d {d1}, {dst}, 8",                       // Load dst[1]

            // [Limb 0 Multiply-Add]
            "mul.d {p_lo0}, {s0}, {scalar}",             // Low 64 bits of src[0] * scalar
            "mulh.du {p_hi0}, {s0}, {scalar}",           // High 64 bits of src[0] * scalar
            "add.d {t_lo}, {p_lo0}, {carry_in}",         // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if addition wrapped
            "add.d {p_hi0}, {p_hi0}, {ca}",              // p_hi0 += ca
            "add.d {t0}, {t_lo}, {d0}",                  // t0 = t_lo + dst[0]
            "sltu {cb}, {t0}, {d0}",                     // cb = 1 if addition wrapped
            "add.d {carry_in}, {p_hi0}, {cb}",           // carry_in = p_hi0 + cb
            "st.d {t0}, {dst}, 0",                       // Store updated dst[0]

            // [Limb 1 Multiply-Add]
            "mul.d {p_lo1}, {s1}, {scalar}",             // Low 64 bits of src[1] * scalar
            "mulh.du {p_hi1}, {s1}, {scalar}",           // High 64 bits of src[1] * scalar
            "add.d {t_lo}, {p_lo1}, {carry_in}",         // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if addition wrapped
            "add.d {p_hi1}, {p_hi1}, {ca}",              // p_hi1 += ca
            "add.d {t0}, {t_lo}, {d1}",                  // t0 = t_lo + dst[1]
            "sltu {cb}, {t0}, {d1}",                     // cb = 1 if addition wrapped
            "add.d {carry_in}, {p_hi1}, {cb}",           // carry_in = p_hi1 + cb
            "st.d {t0}, {dst}, 8",                       // Store updated dst[1]

            // Advance pointers by 2 limbs (16 bytes) and loop
            "addi.d {src}, {src}, 16",                   // Advance src pointer
            "addi.d {dst}, {dst}, 16",                   // Advance dst pointer
            "addi.d {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",                                        // Remainder entry label
            "beqz {rem}, 4f",                            // If rem == 0, skip to finish (4f)

            // 1-limb tail
            "3:",                                        // Tail loop label
            "ld.d {s0}, {src}, 0",                       // Load single src limb
            "ld.d {d0}, {dst}, 0",                       // Load single dst limb
            "mul.d {p_lo0}, {s0}, {scalar}",             // Low 64-bit product
            "mulh.du {p_hi0}, {s0}, {scalar}",           // High 64-bit product
            "add.d {t_lo}, {p_lo0}, {carry_in}",         // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add.d {p_hi0}, {p_hi0}, {ca}",              // Propagate carry
            "add.d {t0}, {t_lo}, {d0}",                  // Accumulate into destination limb
            "sltu {cb}, {t0}, {d0}",                     // Detect second overflow
            "add.d {carry_in}, {p_hi0}, {cb}",           // Final carry for limb
            "st.d {t0}, {dst}, 0",                       // Store updated limb

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
