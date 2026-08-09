//! MIPS 32-bit architecture-specific limb operations.
//!
//! Uses `multu`/`mflo`/`mfhi` for efficient 32x32->64 multiplication and
//! `addu` with `sltu` for carry tracking.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
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
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",
            "1:",
            "lw {s0}, 0({src})",
            "lw {s1}, 4({src})",
            "lw {d0}, 0({dst})",
            "lw {d1}, 4({dst})",

            "multu {s0}, {scalar}",
            "mflo {p_lo0}",
            "mfhi {p_hi0}",
            "addu {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "addu {p_hi0}, {p_hi0}, {ca}",
            "addu {t0}, {t_lo}, {d0}",
            "sltu {cb}, {t0}, {d0}",
            "addu {carry_in}, {p_hi0}, {cb}",
            "sw {t0}, 0({dst})",

            "multu {s1}, {scalar}",
            "mflo {p_lo1}",
            "mfhi {p_hi1}",
            "addu {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            "addu {p_hi1}, {p_hi1}, {ca}",
            "addu {t0}, {t_lo}, {d1}",
            "sltu {cb}, {t0}, {d1}",
            "addu {carry_in}, {p_hi1}, {cb}",
            "sw {t0}, 4({dst})",

            "addiu {src}, {src}, 8",
            "addiu {dst}, {dst}, 8",
            "addiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            "3:",
            "lw {s0}, 0({src})",
            "lw {d0}, 0({dst})",
            "multu {s0}, {scalar}",
            "mflo {p_lo0}",
            "mfhi {p_hi0}",
            "addu {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "addu {p_hi0}, {p_hi0}, {ca}",
            "addu {t0}, {t_lo}, {d0}",
            "sltu {cb}, {t0}, {d0}",
            "addu {carry_in}, {p_hi0}, {cb}",
            "sw {t0}, 0({dst})",
            "4:",

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
