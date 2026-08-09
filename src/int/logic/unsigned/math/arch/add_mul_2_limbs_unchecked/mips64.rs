//! `MIPS64` fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for MIPS 64-bit.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len + 1` limbs: the second
///   row writes one limb ahead of the first, so the last store lands at
///   `dst[len]`.
/// - `src` must be valid for reads of `len` limbs.
/// - `dst` and `src` must not overlap, even partially: the loop reads `src`
///   while it writes `dst`, so any overlap is a data race.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
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

    // SAFETY: Caller guarantees dst and src are valid for len elements.
    unsafe {
        asm!(
            ".set noat",
            "1:",
            "ld {s}, 0({src})",
            "ld {d0}, 0({dst})",
            "ld {d1}, 8({dst})",

            // --- s0 chain: dst[j] += src[j] * s0 + c0 ---
            "dmultu {s}, {s0}",
            "mflo {p_lo0}",
            "mfhi {p_hi0}",
            "daddu {t_lo0}, {p_lo0}, {c0}",
            "sltu {ca0}, {t_lo0}, {c0}",
            "daddu {p_hi0}, {p_hi0}, {ca0}",
            "daddu {t0}, {t_lo0}, {d0}",
            "sltu {cb0}, {t0}, {d0}",
            "daddu {c0}, {p_hi0}, {cb0}",

            // --- s1 chain: dst[j+1] += src[j] * s1 + c1 ---
            "dmultu {s}, {s1}",
            "mflo {p_lo1}",
            "mfhi {p_hi1}",
            "daddu {t_lo1}, {p_lo1}, {c1}",
            "sltu {ca1}, {t_lo1}, {c1}",
            "daddu {p_hi1}, {p_hi1}, {ca1}",
            "daddu {t1}, {t_lo1}, {d1}",
            "sltu {cb1}, {t1}, {d1}",
            "daddu {c1}, {p_hi1}, {cb1}",

            "sd {t0}, 0({dst})",
            "sd {t1}, 8({dst})",

            "daddiu {src}, {src}, 8",
            "daddiu {dst}, {dst}, 8",
            "daddiu {len}, {len}, -1",
            "bnez {len}, 1b",

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
