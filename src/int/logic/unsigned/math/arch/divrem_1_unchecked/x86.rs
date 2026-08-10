//! 32-bit x86 two-limb division kernel.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << 32) | limb` by `divisor`.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor`; violating either
/// precondition would raise the processor's divide exception.
#[allow(
    clippy::inline_always,
    reason = "DIV performs the complete 64-by-32 operation and avoids the compiler runtime helper"
)]
#[inline(always)]
pub unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    let mut quotient = limb;
    let mut remainder = rem_hi;
    // SAFETY: the caller proves divisor != 0 and rem_hi < divisor, exactly the
    // hardware conditions for a non-trapping 64-by-32 DIV with a 32-bit result.
    unsafe {
        asm!(
            "divl {divisor}",
            divisor = in(reg) divisor,
            inout("eax") quotient,
            inout("edx") remainder,
            options(pure, nomem, nostack, att_syntax)
        );
    }
    (quotient, remainder)
}
