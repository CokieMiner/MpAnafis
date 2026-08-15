//! RISC-V 64-bit fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using 64×64→128-bit multipliers (`mul`/`mulhu`) and branchless `sltu` carry capture.

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for RISC-V 64-bit.
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
/// RISC-V 64-bit evaluates four 64×64→128-bit products per limb across both rows (`s0` and `s1`),
/// captures two independent carry chains branchlessly with `sltu`, and updates memory in place.
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

    if len == 0 {
        return (0, 0);
    }

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`) remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            // Main dual-row accumulation loop
            "1:",
            "ld {s}, 0({src})",                          // Load src[j]
            "ld {d0}, 0({dst})",                         // Load dst[j]
            "ld {d1}, 8({dst})",                         // Load dst[j+1]

            // [Row 0 Carry Chain: dst[j] += src[j] * s0 + c0]
            "mul {p_lo0}, {s}, {s0}",                    // Low 64 bits of src[j] * s0
            "mulhu {p_hi0}, {s}, {s0}",                  // High 64 bits of src[j] * s0
            "add {t_lo0}, {p_lo0}, {c0}",                // t_lo0 = p_lo0 + c0
            "sltu {ca0}, {t_lo0}, {c0}",                 // ca0 = 1 if addition wrapped
            "add {p_hi0}, {p_hi0}, {ca0}",               // p_hi0 += ca0
            "add {t0}, {t_lo0}, {d0}",                   // t0 = t_lo0 + d0
            "sltu {cb0}, {t0}, {d0}",                    // cb0 = 1 if addition wrapped
            "add {c0}, {p_hi0}, {cb0}",                  // c0 = p_hi0 + cb0 (row 0 carry)

            // [Row 1 Carry Chain: dst[j+1] += src[j] * s1 + c1]
            "mul {p_lo1}, {s}, {s1}",                    // Low 64 bits of src[j] * s1
            "mulhu {p_hi1}, {s}, {s1}",                  // High 64 bits of src[j] * s1
            "add {t_lo1}, {p_lo1}, {c1}",                // t_lo1 = p_lo1 + c1
            "sltu {ca1}, {t_lo1}, {c1}",                 // ca1 = 1 if addition wrapped
            "add {p_hi1}, {p_hi1}, {ca1}",               // p_hi1 += ca1
            "add {t1}, {t_lo1}, {d1}",                   // t1 = t_lo1 + d1
            "sltu {cb1}, {t1}, {d1}",                    // cb1 = 1 if addition wrapped
            "add {c1}, {p_hi1}, {cb1}",                  // c1 = p_hi1 + cb1 (row 1 carry)

            // Store updated destination limbs
            "sd {t0}, 0({dst})",                         // Store finalized dst[j]
            "sd {t1}, 8({dst})",                         // Store intermediate dst[j+1]

            // Advance pointers and loop
            "addi {src}, {src}, 8",                      // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                      // Advance dst pointer by 8 bytes
            "addi {len}, {len}, -1",                     // Decrement remaining limbs
            "bnez {len}, 1b",                            // Repeat while len != 0

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            t_lo0 = out(reg) _,
            ca0 = out(reg) _,
            t0 = out(reg) _,
            cb0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo1 = out(reg) _,
            ca1 = out(reg) _,
            t1 = out(reg) _,
            cb1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
