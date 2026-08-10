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
//! ## Loop structure
//!
//! 2-way unrolled (`len >> 1`) with a single-limb tail for the remainder.

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
    let mut borrow: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    //
    // CC-carrier pitfalls handled:
    //   - `cgij` clobbers CC, so we reset CC=0 *after* the chunks guard and
    //     before the main loop.
    //   - The remainder skip uses `brctg` (which does NOT touch CC): when
    //     `rem == 0`, `brctg` wraps to `u64::MAX` and branches (skipping the
    //     tail); when `rem == 1`, it decrements to 0 and falls through into
    //     the tail, preserving the main-loop borrow in CC.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: reset CC=0 before entering main loop
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",          // CC = 0 (reset borrow flag)

            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            // Process limb 0
            "lg {src_val0}, 0({src})",          // load src[0]
            "lg {dst_val0}, 0({dst})",          // load dst[0]
            "slbgr {dst_val0}, {src_val0}",     // dst[0] = dst[0] − src[0] − borrow
            "stg {dst_val0}, 0({dst})",         // store dst[0]
            // Process limb 1
            "lg {src_val1}, 8({src})",          // load src[1]
            "lg {dst_val1}, 8({dst})",          // load dst[1]
            "slbgr {dst_val1}, {src_val1}",     // dst[1] = dst[1] − src[1] − borrow
            "stg {dst_val1}, 8({dst})",         // store dst[1]
            "la {src}, 16({src})",              // src += 16
            "la {dst}, 16({dst})",              // dst += 16
            "brctg {chunks}, 2b",               // --chunks; branch if != 0 (preserves CC)
            "j 4f",                              // skip CC re-init (main loop done, CC already set)

            "1:",                                 // chunks == 0: set CC=0 for tail
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",           // CC = 0 (reset borrow flag)

            "4:",                                 // common: CC is ready
            "brctg {rem}, 3f",                  // if rem was 0 → wrap to MAX, branch (skip tail); preserves CC
            // Process remainder limb
            "lg {src_val0}, 0({src})",
            "lg {dst_val0}, 0({dst})",
            "slbgr {dst_val0}, {src_val0}",
            "stg {dst_val0}, 0({dst})",
            "3:",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",         // borrow = −borrow_from_CC (0 or −1)
            "lcgr {borrow}, {borrow}",          // two's complement → 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_addr) dst => _,
            src = inout(reg_addr) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
