//! `LoongArch32` fused multiply-add limb kernel.
//!
//! Uses 32×32→64-bit unsigned multipliers (`mul.w`/`mulh.wu`), branchless carry
//! propagation via `sltu`, and 2-way unrolling.

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
/// `LoongArch32` computes 64-bit products via `mul.w` and `mulh.wu`. Carries are captured branchlessly
/// using `sltu` (Set Less Than Unsigned) and accumulated into high products. The loop is 2-way unrolled.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit LoongArch multi-precision hot paths"
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
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder loop (2f)

            // Main 2-way unrolled loop body
            "1:",                                        // Loop head label
            // [Load 2 Source and 2 Destination Limbs (32-bit each)]
            "ld.w {s0}, {src}, 0",                       // Load src[0]
            "ld.w {s1}, {src}, 4",                       // Load src[1]
            "ld.w {d0}, {dst}, 0",                       // Load dst[0]
            "ld.w {d1}, {dst}, 4",                       // Load dst[1]

            // [Limb 0 Multiply-Add]
            "mul.w {p_lo0}, {s0}, {scalar}",             // Low 32 bits of src[0] * scalar
            "mulh.wu {p_hi0}, {s0}, {scalar}",           // High 32 bits of src[0] * scalar
            "add.w {t_lo}, {p_lo0}, {carry_in}",         // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if addition wrapped
            "add.w {p_hi0}, {p_hi0}, {ca}",              // p_hi0 += ca
            "add.w {t0}, {t_lo}, {d0}",                  // t0 = t_lo + dst[0]
            "sltu {cb}, {t0}, {d0}",                     // cb = 1 if addition wrapped
            "add.w {carry_in}, {p_hi0}, {cb}",           // carry_in = p_hi0 + cb
            "st.w {t0}, {dst}, 0",                       // Store updated dst[0]

            // [Limb 1 Multiply-Add]
            "mul.w {p_lo1}, {s1}, {scalar}",             // Low 32 bits of src[1] * scalar
            "mulh.wu {p_hi1}, {s1}, {scalar}",           // High 32 bits of src[1] * scalar
            "add.w {t_lo}, {p_lo1}, {carry_in}",         // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if addition wrapped
            "add.w {p_hi1}, {p_hi1}, {ca}",              // p_hi1 += ca
            "add.w {t0}, {t_lo}, {d1}",                  // t0 = t_lo + dst[1]
            "sltu {cb}, {t0}, {d1}",                     // cb = 1 if addition wrapped
            "add.w {carry_in}, {p_hi1}, {cb}",           // carry_in = p_hi1 + cb
            "st.w {t0}, {dst}, 4",                       // Store updated dst[1]

            // Advance pointers by 2 limbs (8 bytes) and loop
            "addi.w {src}, {src}, 8",                    // Advance src pointer
            "addi.w {dst}, {dst}, 8",                    // Advance dst pointer
            "addi.w {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",                                        // Remainder entry label
            "beqz {rem}, 4f",                            // If rem == 0, skip to finish (4f)

            // 1-limb tail
            "3:",                                        // Tail loop label
            "ld.w {s0}, {src}, 0",                       // Load single src limb
            "ld.w {d0}, {dst}, 0",                       // Load single dst limb
            "mul.w {p_lo0}, {s0}, {scalar}",             // Low 32-bit product
            "mulh.wu {p_hi0}, {s0}, {scalar}",           // High 32-bit product
            "add.w {t_lo}, {p_lo0}, {carry_in}",         // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add.w {p_hi0}, {p_hi0}, {ca}",              // Propagate carry
            "add.w {t0}, {t_lo}, {d0}",                  // Accumulate into destination limb
            "sltu {cb}, {t0}, {d0}",                     // Detect second overflow
            "add.w {carry_in}, {p_hi0}, {cb}",           // Final carry for limb
            "st.w {t0}, {dst}, 0",                       // Store updated limb

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
