//! `MIPS64` fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using `dmultu`/`mflo`/`mfhi` and branchless `sltu` carry capture.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
#[expect(
    clippy::inline_always,
    reason = "Keep both hardware row recurrences in the selected multiplication caller"
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

    // SAFETY: len > 0 gives exactly len iterations. Each reads source[j]
    // and the initialized destination[j..=j+1], then writes only that disjoint
    // destination. Eight-byte advances preserve alignment within the live spans.
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
