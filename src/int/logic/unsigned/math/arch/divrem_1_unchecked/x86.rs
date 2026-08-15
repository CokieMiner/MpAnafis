//! 32-bit x86 two-limb division kernel.
//!
//! Evaluates `(q, r) = (rem_hi:limb) / divisor` in a single hardware `divl` instruction.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << 32) | limb` by `divisor`.
///
/// Computes:
///
/// ```text
///   (quotient, remainder) = ((rem_hi << 32) | limb) / divisor
/// ```
///
/// # Microarchitectural Strategy
///
/// `divl {divisor}` divides the 64-bit unsigned value in `%edx:%eax` by the 32-bit divisor.
/// Returns quotient in `%eax` and remainder in `%edx`.
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

    // SAFETY:
    // 1. Caller guarantees `divisor != 0`.
    // 2. Caller guarantees `rem_hi < divisor`, ensuring the quotient fits in 32 bits without fault.
    unsafe {
        asm!(
            "divl {divisor}",                            // %eax = (%edx:%eax) / divisor, %edx = (%edx:%eax) % divisor
            divisor = in(reg) divisor,
            inout("eax") quotient,
            inout("edx") remainder,
            options(pure, nomem, nostack, att_syntax)
        );
    }
    (quotient, remainder)
}
