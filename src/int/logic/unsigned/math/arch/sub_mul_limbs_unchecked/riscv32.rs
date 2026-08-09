//! RISC-V 32-bit architecture-specific limb operations.
//!
//! Uses `mul`/`mulhu` for efficient 32x32->64 multiplication and
//! `add`/`sub` with `sltu` for carry/borrow tracking.

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
            "beqz {chunks}, 2f",
            "1:",
            "lw {s0}, 0({src})",
            "lw {s1}, 4({src})",
            "lw {d0}, 0({dst})",
            "lw {d1}, 4({dst})",

            "mul {p_lo0}, {s0}, {scalar}",
            "mulhu {p_hi0}, {s0}, {scalar}",
            "add {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add {carry_in}, {p_hi0}, {ca}",
            "sub {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",
            "sub {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sw {t1}, 0({dst})",

            "mul {p_lo1}, {s1}, {scalar}",
            "mulhu {p_hi1}, {s1}, {scalar}",
            "add {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            "add {carry_in}, {p_hi1}, {ca}",
            "sub {t0}, {d1}, {t_lo}",
            "sltu {b0}, {d1}, {t_lo}",
            "sub {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sw {t1}, 4({dst})",

            "addi {src}, {src}, 8",
            "addi {dst}, {dst}, 8",
            "addi {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            "3:",
            "lw {s0}, 0({src})",
            "lw {d0}, 0({dst})",
            "mul {p_lo0}, {s0}, {scalar}",
            "mulhu {p_hi0}, {s0}, {scalar}",
            "add {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add {carry_in}, {p_hi0}, {ca}",
            "sub {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",
            "sub {t1}, {t0}, {borrow_out}",
            "sltu {b1}, {t0}, {borrow_out}",
            "or {borrow_out}, {b0}, {b1}",
            "sw {t1}, 0({dst})",
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
