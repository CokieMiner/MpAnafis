//! `s390x` (IBM Z) subtraction kernels (inline assembly).
//!
//! Uses the condition code (CC) to track borrow across limbs:
//!
//! ```text
//!   lghi   borrow, 0
//!   slgr   borrow, borrow     // CC = 0  (result zero, no borrow)
//!   slbgr  dst, src           // dst = dst − src − borrow_from_CC
//! ```
//!
//! ## Borrow propagation
//!
//! `SLGR` sets CC = 0/1 when no borrow occurs, CC = 2/3 when borrow occurs.
//! `SLBGR` uses the inverse: a borrow (CC = 2/3) causes 1 to be subtracted
//! from the result, while no borrow (CC = 0/1) subtracts 0.
//!
//! ## Loop structure & Unrolling Rationale (Why 2-way instead of 4-way)
//!
//! Both 2-operand and 3-operand kernels are 2-way unrolled (`len >> 1`) with a single-limb
//! tail for the remainder (`len & 1`).
//!
//! Unlike `x86_64` or `aarch64` which use 4-way unrolling, `s390x` kernels use 2-way unrolling.
//! This is because remainder handling (`len & 1`) with 2-way unrolling can be executed
//! using a single `brctg` (branch on count) instruction, which branches without clobbering
//! CC. Unrolling 4-way would leave up to 3 remainder limbs (`len & 3`), requiring additional
//! compare and jump instructions (`cgij`/`clgij`) or loops that would clobber CC or require
//! CC-saving overhead, negating any benefits of higher unrolling factors.

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
    let mut borrow: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    // `cgij` clobbers CC; we reset CC=0 after the guard before entering the loop.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: reset CC=0 before entering main loop
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",          // CC = 0 (reset borrow flag)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lg {src1_val0}, 0({src1})",        // load src1[0]
            "lg {src2_val0}, 0({src2})",        // load src2[0]
            "slbgr {src1_val0}, {src2_val0}",   // src1_val0 = src1[0] − src2[0] − borrow
            "stg {src1_val0}, 0({dst})",        // store dst[0]
            "lg {src1_val1}, 8({src1})",        // load src1[1]
            "lg {src2_val1}, 8({src2})",        // load src2[1]
            "slbgr {src1_val1}, {src2_val1}",   // src1_val1 = src1[1] − src2[1] − borrow
            "stg {src1_val1}, 8({dst})",        // store dst[1]
            "la {src1}, 16({src1})",            // src1 += 16
            "la {src2}, 16({src2})",            // src2 += 16
            "la {dst}, 16({dst})",              // dst  += 16
            "brctg {chunks}, 2b",               // --chunks; branch if != 0 (preserves CC)
            "j 4f",                              // skip CC re-init (main loop done, CC already set)

            "1:",                                 // chunks == 0: set CC=0 for tail
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",           // CC = 0 (reset borrow flag)

            "4:",                                 // common: CC is ready
            "brctg {rem}, 3f",                  // if rem was 0 → skip tail
            "lg {src1_val0}, 0({src1})",        // load last limb
            "lg {src2_val0}, 0({src2})",        // load last src2
            "slbgr {src1_val0}, {src2_val0}",   // dst += src1 - src2 - borrow
            "stg {src1_val0}, 0({dst})",        // store result
            "3:",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",         // borrow = −borrow_from_CC (0 or −1)
            "lcgr {borrow}, {borrow}",          // two's complement → 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_addr) dst => _,
            src1 = inout(reg_addr) src1 => _,
            src2 = inout(reg_addr) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_val0 = out(reg) _, src2_val0 = out(reg) _,
            src1_val1 = out(reg) _, src2_val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
