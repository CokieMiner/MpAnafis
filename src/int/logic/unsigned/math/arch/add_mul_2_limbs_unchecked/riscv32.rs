//! RISC-V 32-bit fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for RISC-V 32-bit.
///
/// Computes:
/// ```text
/// dst[0..len] += src[0..len] * s0 + c0
/// dst[1..len+1] += src[0..len] * s1 + c1
/// ```
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
            "1:",
            "lw {s}, 0({src})",
            "lw {d0}, 0({dst})",
            "lw {d1}, 4({dst})",

            // --- s0 chain: dst[j] += src[j] * s0 + c0 ---
            "mul {p_lo0}, {s}, {s0}",
            "mulhu {p_hi0}, {s}, {s0}",
            "add {t_lo0}, {p_lo0}, {c0}",
            "sltu {ca0}, {t_lo0}, {c0}",
            "add {p_hi0}, {p_hi0}, {ca0}",
            "add {t0}, {t_lo0}, {d0}",
            "sltu {cb0}, {t0}, {d0}",
            "add {c0}, {p_hi0}, {cb0}",

            // --- s1 chain: dst[j+1] += src[j] * s1 + c1 ---
            "mul {p_lo1}, {s}, {s1}",
            "mulhu {p_hi1}, {s}, {s1}",
            "add {t_lo1}, {p_lo1}, {c1}",
            "sltu {ca1}, {t_lo1}, {c1}",
            "add {p_hi1}, {p_hi1}, {ca1}",
            "add {t1}, {t_lo1}, {d1}",
            "sltu {cb1}, {t1}, {d1}",
            "add {c1}, {p_hi1}, {cb1}",

            "sw {t0}, 0({dst})",
            "sw {t1}, 4({dst})",

            "addi {src}, {src}, 4",
            "addi {dst}, {dst}, 4",
            "addi {len}, {len}, -1",
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
