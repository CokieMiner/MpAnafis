//! `s390x` (IBM Z) borrow propagation kernel (inline assembly).
//!
//! Uses CC via `slbgr` (subtract logical with borrow) in a 2-way unrolled loop.

use core::arch::asm;

use super::Limb;

/// Propagate borrow through `dst` slice.
///
/// # Safety
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `borrow` must be `0` or `1` (the CC is derived from `0 - borrow`).
#[allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    if borrow == 0 || len == 0 {
        return borrow;
    }
    let chunks = len >> 1;
    let rem = len & 1;
    let zero_const: Limb = 0;
    // SAFETY: deferred to caller per the doc comment above.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: set CC from borrow before entering main loop
            "lghi {cc_seed}, 0",
            "slgr {cc_seed}, {borrow}",         // CC = 2/3 if borrow==1, CC=0/1 if borrow==0
            ".p2align 4",
            "2:",
            "lg {val0}, 0({dst})",
            "lg {val1}, 8({dst})",
            "slbgr {val0}, {zero}",             // val0 -= 0 + borrow_from_CC
            "slbgr {val1}, {zero}",             // val1 -= 0 + borrow_from_CC
            "stg {val0}, 0({dst})",
            "stg {val1}, 8({dst})",
            "la {dst}, 16({dst})",
            "brctg {chunks}, 2b",
            "j 4f",                             // skip CC re-init (main loop done, CC already set)

            "1:",                                // chunks == 0: set CC from borrow for tail/exit
            "lghi {cc_seed}, 0",
            "slgr {cc_seed}, {borrow}",

            "4:",                                // common: CC is ready (set either above or by main loop)
            "brctg {rem}, 3f",
            "lg {val0}, 0({dst})",
            "slbgr {val0}, {zero}",
            "stg {val0}, 0({dst})",
            "3:",
            "5:",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",         // borrow = -borrow_from_CC (0 or -1)
            "lcgr {borrow}, {borrow}",          // two's complement -> 0 or 1
            borrow = inout(reg) borrow,
            dst = inout(reg_addr) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            zero = inout(reg) zero_const => _,
            cc_seed = out(reg) _,
            val0 = out(reg) _,
            val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
