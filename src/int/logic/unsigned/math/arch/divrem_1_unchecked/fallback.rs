//! Pure Rust fallback for `divrem_1_unchecked`.
//!
//! Uses `DoubleLimb` arithmetic: the `DoubleLimb` numerator
//! `(rem_hi << LIMB_BITS) | limb` is divided by the `Limb` divisor using
//! standard integer division and remainder.

use core::num::NonZeroUsize;

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Evaluates `(q, r)` such that `q * divisor + r = rem_hi * B + limb`.
///
/// Divides a two-limb numerator `(rem_hi * B + limb)` by a single-limb divisor,
/// returning quotient `q` and remainder `r < divisor`.
///
/// # Safety
///
/// The caller must guarantee:
/// 1. `divisor != 0` (non-zero divisor).
/// 2. `rem_hi < divisor` (guarantees quotient fits in a single limb).
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "Inlining keeps scalar division at the caller; Limb-to-DoubleLimb casts widen on every pointer width"
)]
#[cfg_attr(
    not(target_pointer_width = "16"),
    expect(
        clippy::cast_possible_truncation,
        reason = "quotient and remainder always fit in Limb on all pointer widths"
    )
)]
#[inline(always)]
pub const unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    // SAFETY: the kernel contract requires divisor > 0 on every path.
    let denominator = unsafe { NonZeroUsize::new_unchecked(divisor) };
    // Keep scalar callers in native limb arithmetic, including targets whose
    // DoubleLimb division requires a software routine.
    if rem_hi == 0 {
        // SAFETY: the caller guarantees divisor > 0; the quotient fits Limb
        // and its product with divisor is bounded by limb.
        unsafe {
            let quotient = limb.checked_div(denominator.get()).unwrap_unchecked();
            return (
                quotient,
                limb.unchecked_sub(quotient.unchecked_mul(denominator.get())),
            );
        }
    }
    let n_val = ((rem_hi as DoubleLimb) << LIMB_BITS) | (limb as DoubleLimb);
    let den_wide = denominator.get() as DoubleLimb;
    // SAFETY: `divisor` is non-zero (caller contract), so division is safe.
    let q_val = unsafe { n_val.checked_div(den_wide).unwrap_unchecked() } as Limb;
    let q_wide = q_val as DoubleLimb;
    // SAFETY: rem_hi<divisor makes q_val the complete quotient. Thus
    // q_wide*den_wide<=n_val; the product and subtraction fit DoubleLimb.
    let r_val = unsafe { n_val.unchecked_sub(q_wide.unchecked_mul(den_wide)) } as Limb;
    (q_val, r_val)
}
