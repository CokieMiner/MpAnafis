//! `x86_64` division kernel using the `divq` instruction.
//!
//! # Algorithm
//!
//! The x86 `divq` instruction divides the 128-bit value in `rdx:rax`
//! by a 64-bit divisor.  The quotient is placed in `rax` and the
//! remainder in `rdx`.
//!
//! We place the limb (low 64 bits of numerator) in `rax` and the
//! previous remainder (high 64 bits) in `rdx`, then execute `divq
//! divisor`.  The result is directly the quotient and remainder.
//!
//! # Safety
//!
//! The caller must guarantee that the quotient fits in 64 bits
//! (i.e. `rem_hi < divisor`).  If this precondition is violated the
//! CPU will raise a #DE (divide error) exception.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << LIMB_BITS) | limb` by `divisor`.
///
/// On `x86_64` this is a single `divq` instruction that computes both
/// the quotient and remainder.
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
    let mut q_val: Limb = limb; // will be placed in rax
    let mut r_val: Limb = rem_hi; // will be placed in rdx
    // SAFETY: Caller ensures divisor ≠ 0 and rem_hi < divisor.
    unsafe {
        asm!(
            // `divq src` computes:  (rdx:rax) ÷ src
            //   quotient → rax
            //   remainder → rdx
            //
            // Precondition: rdx (high half) < src, so quotient fits in rax.
            "divq {d}",
            d = in(reg) divisor,
            // rax holds the low 64 bits of the dividend (limb).
            // rdx holds the high 64 bits of the dividend (rem_hi).
            // After divq:
            //   rax = quotient
            //   rdx = remainder
            inout("rax") q_val,
            inout("rdx") r_val,
            // `pure` + `nomem`: the instruction reads/writes only registers.
            // `nostack`: no stack pointer usage.
            options(pure, nomem, nostack, att_syntax)
        );
    }
    (q_val, r_val)
}
