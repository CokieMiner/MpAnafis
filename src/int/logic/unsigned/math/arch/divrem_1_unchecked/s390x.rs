//! `s390x` division kernel using the `dlgr` instruction.
//!
//! # Algorithm
//!
//! The `s390x` `dlgr %r0, Rx` instruction divides the 128-bit value in the
//! even-odd register pair `%r0:%r1` (dividend, high in `%r0`, low in `%r1`)
//! by the 64-bit divisor in register `Rx`.  The quotient is placed in `%r1`
//! (odd register) and the remainder in `%r0` (even register).
//!
//! We place the limb (low 64 bits of numerator) in `%r1` and the previous
//! remainder (high 64 bits) in `%r0`, then execute `dlgr`.  The result is
//! directly the quotient and remainder.
//!
//! # Safety
//!
//! The caller must guarantee that the quotient fits in 64 bits
//! (i.e. `rem_hi < divisor`).  If this precondition is violated
//! a hardware fixed-point divide exception occurs.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << 64) | limb` by `divisor`.
///
/// On `s390x` this is a single `dlgr` instruction that computes both
/// the quotient and remainder using the hardware 128÷64 divider.
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
    let mut lo: Limb = limb; // placed in %r1 (odd register of pair %r1:%r0)
    let mut hi: Limb = rem_hi; // placed in %r0 (even register)
    // SAFETY: Caller ensures divisor ≠ 0 and rem_hi < divisor.
    unsafe {
        asm!(
            // `dlgr %r0, {d}` computes:  (%r1:%r0) ÷ {d}
            //   quotient  → %r1
            //   remainder → %r0
            //
            // Precondition: %r0 (high half) < {d}, so quotient fits in %r1.
            "dlgr %r0, {d}",
            d = in(reg) divisor,
            // %r0 holds the high 64 bits of the dividend (rem_hi).
            // %r1 holds the low 64 bits of the dividend (limb).
            // After dlgr:
            //   %r1 = quotient
            //   %r0 = remainder
            inout("r0") hi,
            inout("r1") lo,
            options(pure, nomem, nostack)
        );
    }
    (lo, hi)
}
