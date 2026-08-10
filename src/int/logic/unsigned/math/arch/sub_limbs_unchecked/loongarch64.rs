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

// ── sub_n ──────────────────────────────────────────────────────────────────

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0: t2 = dst[0] − src[0] − borrow
            "ld.d {t0}, {src}, 0",           // t0 = src[0]
            "ld.d {t1}, {dst}, 0",           // t1 = dst[0]
            "sltu {b0}, {t1}, {t0}",          // b0 = 1 if dst < src
            "sub.d {t2}, {t1}, {t0}",         // t2 = dst − src
            "sltu {b1}, {t2}, {borrow}",      // b1 = 1 if (dst−src) < old_borrow
            "sub.d {t2}, {t2}, {borrow}",     // t2 −= old_borrow
            "or {borrow}, {b0}, {b1}",        // combined borrow
            "st.d {t2}, {dst}, 0",
            // Limb 1
            "ld.d {t0}, {src}, 8",
            "ld.d {t1}, {dst}, 8",
            "sltu {b0}, {t1}, {t0}",
            "sub.d {t2}, {t1}, {t0}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 8",
            // Limb 2
            "ld.d {t0}, {src}, 16",
            "ld.d {t1}, {dst}, 16",
            "sltu {b0}, {t1}, {t0}",
            "sub.d {t2}, {t1}, {t0}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 16",
            // Limb 3
            "ld.d {t0}, {src}, 24",
            "ld.d {t1}, {dst}, 24",
            "sltu {b0}, {t1}, {t0}",
            "sub.d {t2}, {t1}, {t0}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 24",
            "addi.d {src}, {src}, 32",
            "addi.d {dst}, {dst}, 32",
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",
            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld.d {t0}, {src}, 0",
            "ld.d {t1}, {dst}, 0",
            "sltu {b0}, {t1}, {t0}",
            "sub.d {t2}, {t1}, {t0}",
            "sltu {b1}, {t2}, {borrow}",
            "sub.d {t2}, {t2}, {borrow}",
            "or {borrow}, {b0}, {b1}",
            "st.d {t2}, {dst}, 0",
            "addi.d {src}, {src}, 8",
            "addi.d {dst}, {dst}, 8",
            "addi.d {rem}, {rem}, -1",
            "bnez {rem}, 3b",
            "4:",
            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
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
