//! MIPS 64-bit architecture-specific limb operations.
//!
//! Uses `dmultu`/`mflo`/`mfhi` for efficient 64x64->128 multiplication and
//! `daddu`/`dsubu` with `sltu` for carry/borrow tracking.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements and `src`
///   for reads of `len` elements.
/// - `dst` and `src` must not overlap, even partially: the kernel reads
///   `src` while it writes `dst`.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_in: Limb = 0;
    let mut borrow_out: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",
            "1:",
            "ld {s0}, 0({src})",
            "ld {s1}, 8({src})",
            "ld {d0}, 0({dst})",
            "ld {d1}, 8({dst})",

            "dmultu {s0}, {scalar}",
            "mflo {p_lo0}",
            "mfhi {p_hi0}",
            "daddu {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "daddu {carry_in}, {p_hi0}, {ca}",
            "dsubu {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",
            "dsubu {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sd {t1}, 0({dst})",

            "dmultu {s1}, {scalar}",
            "mflo {p_lo1}",
            "mfhi {p_hi1}",
            "daddu {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            "daddu {carry_in}, {p_hi1}, {ca}",
            "dsubu {t0}, {d1}, {t_lo}",
            "sltu {b0}, {d1}, {t_lo}",
            "dsubu {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sd {t1}, 8({dst})",

            "daddiu {src}, {src}, 16",
            "daddiu {dst}, {dst}, 16",
            "daddiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            "3:",
            "ld {s0}, 0({src})",
            "ld {d0}, 0({dst})",
            "dmultu {s0}, {scalar}",
            "mflo {p_lo0}",
            "mfhi {p_hi0}",
            "daddu {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "daddu {carry_in}, {p_hi0}, {ca}",
            "dsubu {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",
            "dsubu {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sd {t1}, 0({dst})",
            "4:",

            carry_in = inout(reg) carry_in,
            borrow_out = inout(reg) borrow_out,
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
            t1 = out(reg) _,
            ca = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_out)
}
