//! `MIPS64` implementation of `sub_limbs_3_unchecked`.

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
            "ld {t0}, 0({src2})",
            "ld {t1}, 0({src1})",
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 0({dst})",

            "ld {t0}, 8({src2})",
            "ld {t1}, 8({src1})",
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 8({dst})",

            "ld {t0}, 16({src2})",
            "ld {t1}, 16({src1})",
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 16({dst})",

            "ld {t0}, 24({src2})",
            "ld {t1}, 24({src1})",
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 24({dst})",

            "daddiu {src1}, {src1}, 32",
            "daddiu {src2}, {src2}, 32",
            "daddiu {dst}, {dst}, 32",
            "daddiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld {t0}, 0({src2})",
            "ld {t1}, 0({src1})",
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 0({dst})",

            "daddiu {src1}, {src1}, 8",
            "daddiu {src2}, {src2}, 8",
            "daddiu {dst}, {dst}, 8",
            "daddiu {rem}, {rem}, -1",
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
