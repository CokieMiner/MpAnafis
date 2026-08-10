//! `s390x` (IBM Z) carry propagation kernel (inline assembly).
//!
//! Uses CC via `alcgr` (add logical with carry) in a 2-way unrolled loop.

use core::arch::asm;

use super::Limb;

/// Propagate carry through `dst` slice.
///
/// # Safety
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `carry` must be `0` or `1` (the CC is derived from `-1 + carry`).
#[allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    if carry == 0 || len == 0 {
        return carry;
    }
    let chunks = len >> 1;
    let rem = len & 1;
    let zero_const: Limb = 0;
    // SAFETY: deferred to caller per the doc comment above.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: set CC from carry before entering main loop
            "lghi {cc_seed}, -1",               // cc_seed = u64::MAX
            "algr {cc_seed}, {carry}",          // CC = carry (if carry==1, CC=2/3; if carry==0, CC=0/1)
            ".p2align 4",
            "2:",
            "lg {val0}, 0({dst})",
            "lg {val1}, 8({dst})",
            "alcgr {val0}, {zero}",             // val0 += 0 + CC
            "alcgr {val1}, {zero}",             // val1 += 0 + CC
            "stg {val0}, 0({dst})",
            "stg {val1}, 8({dst})",
            "la {dst}, 16({dst})",
            "brctg {chunks}, 2b",
            "j 4f",                             // skip CC re-init (main loop done, CC already set)

            "1:",                                // chunks == 0: set CC from carry for tail/exit
            "lghi {cc_seed}, -1",
            "algr {cc_seed}, {carry}",

            "4:",                                // common: CC is ready (set either above or by main loop)
            "brctg {rem}, 3f",
            "lg {val0}, 0({dst})",
            "alcgr {val0}, {zero}",
            "stg {val0}, 0({dst})",
            "3:",
            "5:",
            "lghi {carry}, 0",
            "alcgr {carry}, {carry}",           // carry = 0 + 0 + CC
            carry = inout(reg) carry,
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
    carry
}
