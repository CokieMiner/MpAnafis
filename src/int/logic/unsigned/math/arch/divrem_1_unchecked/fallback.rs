//! Pure Rust fallback for `divrem_1_unchecked`.
//!
//! Uses `DoubleLimb` arithmetic: the `DoubleLimb` numerator
//! `(rem_hi << LIMB_BITS) | limb` is divided by the `Limb` divisor using
//! standard integer division and remainder.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Divide `(rem_hi << LIMB_BITS) | limb` by `divisor`.
///
/// # Returns
///
/// `(quotient, remainder)` where `quotient` is guaranteed to fit in
/// `Limb` because the caller ensures `rem_hi < divisor`.
///
/// # Safety
///
/// The caller must guarantee `divisor != 0` and `rem_hi < divisor`.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Critical for peak performance on platforms without hardware asm; \
              Limb→DoubleLimb is always a widening cast and the quotient/remainder \
              always fit in Limb; LIMB_BITS (≤64) fits in u32 on all targets"
)]
#[inline(always)]
pub const unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    let n_val = (rem_hi as DoubleLimb).wrapping_shl(LIMB_BITS as u32) | (limb as DoubleLimb);
    let den_wide = divisor as DoubleLimb;
    // SAFETY: `divisor` is non-zero (caller contract), so division is safe.
    let q_val = unsafe { n_val.checked_div(den_wide).unwrap_unchecked() } as Limb;
    let q_wide = q_val as DoubleLimb;
    let r_val = n_val.wrapping_sub(q_wide.wrapping_mul(den_wide)) as Limb;
    (q_val, r_val)
}
