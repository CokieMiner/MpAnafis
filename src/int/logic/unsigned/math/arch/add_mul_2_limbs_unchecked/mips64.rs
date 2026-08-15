//! `MIPS64` fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using `dmultu`/`mflo`/`mfhi` and branchless `sltu` carry capture.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for MIPS 64-bit.
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
/// MIPS 64-bit uses dedicated `HI` and `LO` registers for 64×64→128-bit multiplication (`dmultu`).
/// Both carry chains are accumulated branchlessly using `sltu` and non-trapping `daddu`.
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
            ".set noat",
            // Main dual-row accumulation loop
            "1:",
            "ld {s}, 0({src})",                          // Load src[j]
            "ld {d0}, 0({dst})",                         // Load dst[j]
            "ld {d1}, 8({dst})",                         // Load dst[j+1]

            // [Row 0 Carry Chain: dst[j] += src[j] * s0 + c0]
            "dmultu {s}, {s0}",                          // HI:LO = src[j] * s0 (128-bit product)
            "mflo {p_lo0}",                              // Extract low 64 bits from LO
            "mfhi {p_hi0}",                              // Extract high 64 bits from HI
            "daddu {t_lo0}, {p_lo0}, {c0}",              // t_lo0 = p_lo0 + c0
            "sltu {ca0}, {t_lo0}, {c0}",                 // ca0 = 1 if addition wrapped
            "daddu {p_hi0}, {p_hi0}, {ca0}",             // p_hi0 += ca0
            "daddu {t0}, {t_lo0}, {d0}",                 // t0 = t_lo0 + d0
            "sltu {cb0}, {t0}, {d0}",                    // cb0 = 1 if addition wrapped
            "daddu {c0}, {p_hi0}, {cb0}",                // c0 = p_hi0 + cb0 (row 0 carry)

            // [Row 1 Carry Chain: dst[j+1] += src[j] * s1 + c1]
            "dmultu {s}, {s1}",                          // HI:LO = src[j] * s1
            "mflo {p_lo1}",                              // Extract low 64 bits
            "mfhi {p_hi1}",                              // Extract high 64 bits
            "daddu {t_lo1}, {p_lo1}, {c1}",              // t_lo1 = p_lo1 + c1
            "sltu {ca1}, {t_lo1}, {c1}",                 // ca1 = 1 if addition wrapped
            "daddu {p_hi1}, {p_hi1}, {ca1}",             // p_hi1 += ca1
            "daddu {t1}, {t_lo1}, {d1}",                 // t1 = t_lo1 + d1
            "sltu {cb1}, {t1}, {d1}",                    // cb1 = 1 if addition wrapped
            "daddu {c1}, {p_hi1}, {cb1}",                // c1 = p_hi1 + cb1 (row 1 carry)

            // Store updated destination limbs
            "sd {t0}, 0({dst})",                         // Store finalized dst[j]
            "sd {t1}, 8({dst})",                         // Store intermediate dst[j+1]

            // Advance pointers and loop
            "daddiu {src}, {src}, 8",                    // Advance src pointer by 8 bytes
            "daddiu {dst}, {dst}, 8",                    // Advance dst pointer by 8 bytes
            "daddiu {len}, {len}, -1",                   // Decrement remaining count
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
