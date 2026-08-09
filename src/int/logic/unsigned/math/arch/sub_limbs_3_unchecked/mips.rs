//! `MIPS32` implementation of `sub_limbs_3_unchecked`.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs, returning
/// the final borrow.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Caller guarantees pointers are valid for `len` elements.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            "lw {t0}, 0({src2})",
            "lw {t1}, 0({src1})",
            "sltu {c0}, {t1}, {t0}",
            "subu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "subu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 0({dst})",

            "lw {t0}, 4({src2})",
            "lw {t1}, 4({src1})",
            "sltu {c0}, {t1}, {t0}",
            "subu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "subu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 4({dst})",

            "lw {t0}, 8({src2})",
            "lw {t1}, 8({src1})",
            "sltu {c0}, {t1}, {t0}",
            "subu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "subu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 8({dst})",

            "lw {t0}, 12({src2})",
            "lw {t1}, 12({src1})",
            "sltu {c0}, {t1}, {t0}",
            "subu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "subu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 12({dst})",

            "addiu {src1}, {src1}, 16",
            "addiu {src2}, {src2}, 16",
            "addiu {dst}, {dst}, 16",
            "addiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "lw {t0}, 0({src2})",
            "lw {t1}, 0({src1})",
            "sltu {c0}, {t1}, {t0}",
            "subu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "subu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 0({dst})",

            "addiu {src1}, {src1}, 4",
            "addiu {src2}, {src2}, 4",
            "addiu {dst}, {dst}, 4",
            "addiu {rem}, {rem}, -1",
            "bnez {rem}, 3b",
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
