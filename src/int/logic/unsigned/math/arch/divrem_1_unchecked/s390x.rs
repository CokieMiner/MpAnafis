//! `s390x` division kernel using the `dlgr` instruction.
//!
//! Evaluates `(q, r) = (rem_hi:limb) / divisor` in a single hardware `dlgr` instruction.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << 64) | limb` by `divisor`.
///
/// Computes:
///
/// ```text
///   (quotient, remainder) = ((rem_hi << 64) | limb) / divisor
/// ```
///
/// # Microarchitectural Strategy
///
/// On `s390x`, `dlgr %r0, {d}` divides the 128-bit value in the even-odd register pair
/// `%r0:%r1` (dividend: `%r0` high, `%r1` low) by the 64-bit divisor `{d}`.
/// Returns quotient in `%r1` and remainder in `%r0`.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor` (otherwise
/// the quotient would overflow 64 bits, causing a hardware exception).
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    let mut lo: Limb = limb;   // Placed in %r1 (odd register of pair %r1:%r0)
    let mut hi: Limb = rem_hi; // Placed in %r0 (even register)

    // SAFETY:
    // 1. Caller guarantees `divisor != 0`.
    // 2. Caller guarantees `rem_hi < divisor`, ensuring the quotient fits in 64 bits without fault.
    unsafe {
        asm!(
            "dlgr %r0, {d}",                             // %r1 = (%r0:%r1) / d, %r0 = (%r0:%r1) % d
            d = in(reg) divisor,
            inout("r0") hi,
            inout("r1") lo,
            options(pure, nomem, nostack)
        );
    }
    (lo, hi)
}
