//! `LoongArch64` subtraction kernels (inline assembly).
//!
//! Like RISC-V, `LoongArch64` has no borrow flag.  Borrow is detected via
//! `sltu` (set-less-than unsigned) with a two-stage comparison:
//!
//! ```text
//!   sltu    b0, dst, src         // b0 = 1 if dst < src  (first borrow)
//!   sub.d   t, dst, src          // t = dst - src
//!   sltu    b1, t, borrow        // b1 = 1 if (dst-src) < old_borrow
//!   sub.d   t, t, borrow         // t -= old_borrow
//!   or      borrow, b0, b1       // combined borrow for next limb
//! ```
//!
//! Note: `b1` must use the value of `t` BEFORE the second subtraction
//! (the `sltu` is placed before `sub.d` for this reason).
//!
//! ## Loop structure
//!
//! 4-way unrolled (`len >> 2`) with a single-limb tail for the remainder.

use core::arch::asm;

use super::Limb;

// ── sub_n_3 (3-operand) ───────────────────────────────────────────────────

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
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
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0: t2 = src1[0] − src2[0] − borrow
            "ld.d {t0}, {src1}, 0",         // t0 = src1[0]
            "ld.d {t1}, {src2}, 0",         // t1 = src2[0]
            "sltu {b0}, {t0}, {t1}",         // b0 = 1 if src1 < src2
            "sub.d {t2}, {t0}, {t1}",        // t2 = src1 − src2
            "sltu {b1}, {t2}, {borrow}",     // b1 = 1 if diff < borrow
            "sub.d {t2}, {t2}, {borrow}",    // t2 −= borrow
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 0",

            // Limb 1
            "ld.d {t0}, {src1}, 8",
            "ld.d {t1}, {src2}, 8",
            "sltu {b0}, {t0}, {t1}",
            "sub.d {t2}, {t0}, {t1}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 8",

            // Limb 2
            "ld.d {t0}, {src1}, 16",
            "ld.d {t1}, {src2}, 16",
            "sltu {b0}, {t0}, {t1}",
            "sub.d {t2}, {t0}, {t1}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 16",

            // Limb 3
            "ld.d {t0}, {src1}, 24",
            "ld.d {t1}, {src2}, 24",
            "sltu {b0}, {t0}, {t1}",
            "sub.d {t2}, {t0}, {t1}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 24",

            "addi.d {src1}, {src1}, 32",
            "addi.d {src2}, {src2}, 32",
            "addi.d {dst}, {dst}, 32",
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld.d {t0}, {src1}, 0",
            "ld.d {t1}, {src2}, 0",
            "sltu {b0}, {t0}, {t1}",
            "sub.d {t2}, {t0}, {t1}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 0",
            "addi.d {src1}, {src1}, 8",
            "addi.d {src2}, {src2}, 8",
            "addi.d {dst}, {dst}, 8",
            "addi.d {rem}, {rem}, -1",
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
            t2 = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
