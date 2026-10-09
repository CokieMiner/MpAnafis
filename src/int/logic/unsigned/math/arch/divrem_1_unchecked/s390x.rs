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
/// On `s390x`, `dlgr %r0, {d}` divides the 128-bit value in the even-odd register pair
/// `%r0:%r1` (dividend: `%r0` high, `%r1` low) by the 64-bit divisor `{d}`.
/// Returns quotient in `%r1` and remainder in `%r0`.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor` (otherwise
/// the quotient would overflow 64 bits, causing a hardware exception).
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the register-pair divide at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    let mut lo: Limb = limb;   // Low numerator in the odd register %r1
    let mut hi: Limb = rem_hi; // Placed in %r0 (even register)

    // SAFETY: divisor > 0 and rem_hi < divisor make the quotient fit in
    // r1 without a divide exception. The r0:r1 pair is read/write and cannot
    // share a register with the divisor. No memory or stack is accessed, and
    // identical inputs produce identical outputs.
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
