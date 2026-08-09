//! `LoongArch32` architecture-specific limb operations.
//!
//! Uses `mul.w`/`mulh.wu` for efficient 32x32->64 multiplication and
//! `add.w`/`sub.w` with `sltu` for carry/borrow tracking.

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
            "beqz {chunks}, 2f",
            "1:",
            "ld.w {s0}, {src}, 0",
            "ld.w {s1}, {src}, 4",
            "ld.w {d0}, {dst}, 0",
            "ld.w {d1}, {dst}, 4",

            "mul.w {p_lo0}, {s0}, {scalar}",
            "mulh.wu {p_hi0}, {s0}, {scalar}",
            "add.w {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add.w {p_hi0}, {p_hi0}, {ca}",
            "add.w {t0}, {t_lo}, {d0}",
            "sltu {cb}, {t0}, {d0}",
            "add.w {carry_in}, {p_hi0}, {cb}",
            "st.w {t0}, {dst}, 0",

            "mul.w {p_lo1}, {s1}, {scalar}",
            "mulh.wu {p_hi1}, {s1}, {scalar}",
            "add.w {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            "add.w {p_hi1}, {p_hi1}, {ca}",
            "add.w {t0}, {t_lo}, {d1}",
            "sltu {cb}, {t0}, {d1}",
            "add.w {carry_in}, {p_hi1}, {cb}",
            "st.w {t0}, {dst}, 4",

            "addi.w {src}, {src}, 8",
            "addi.w {dst}, {dst}, 8",
            "addi.w {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            "3:",
            "ld.w {s0}, {src}, 0",
            "ld.w {d0}, {dst}, 0",
            "mul.w {p_lo0}, {s0}, {scalar}",
            "mulh.wu {p_hi0}, {s0}, {scalar}",
            "add.w {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add.w {p_hi0}, {p_hi0}, {ca}",
            "add.w {t0}, {t_lo}, {d0}",
            "sltu {cb}, {t0}, {d0}",
            "add.w {carry_in}, {p_hi0}, {cb}",
            "st.w {t0}, {dst}, 0",
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
