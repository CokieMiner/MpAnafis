//! Carry and borrow propagation shared across the whole SSA tier.
//!
//! Every stage of the tier — ring arithmetic, the pointwise basecase, the
//! reconstruction sweep, and the CRT merge — needs to push a single carry or
//! borrow through a limb slice and know whether it escaped off the end. These
//! are the only implementations of that loop in the tier; nothing here is
//! Fermat-specific, which is why it sits above
//! [`ring`](crate::int::logic::unsigned::math::mul::ssa::ring) rather than in
//! it.
//!
//! The `_in_place` pair additionally propagates past the end of a shorter
//! addend, which is what the `B^n - 1` CRT half needs when it folds a carry
//! back around the modulus.

use super::{Addition, Limb};

/// Namespace for carry and borrow propagation shared across the whole SSA tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaCarry;

impl SsaCarry {
    /// Adds one to `limbs`, returning `true` when the carry escapes the slice.
    ///
    /// An empty slice cannot absorb the carry, so it reports `true`.
    #[allow(
        clippy::inline_always,
        reason = "carry propagation on hot modular arithmetic path"
    )]
    #[inline(always)]
    pub fn propagate_carry(limbs: &mut [Limb]) -> bool {
        for limb in limbs {
            let (sum, overflow) = limb.overflowing_add(1);
            *limb = sum;
            if !overflow {
                return false;
            }
        }
        true
    }

    /// Subtracts one from `limbs`, returning `true` when the borrow escapes.
    ///
    /// An empty slice cannot absorb the borrow, so it reports `true`.
    #[allow(
        clippy::inline_always,
        reason = "borrow propagation on hot modular arithmetic path"
    )]
    #[inline(always)]
    pub fn propagate_borrow(limbs: &mut [Limb]) -> bool {
        for limb in limbs {
            let (difference, underflow) = limb.overflowing_sub(1);
            *limb = difference;
            if !underflow {
                return false;
            }
        }
        true
    }

    /// Adds `src` into the prefix of `dst` and propagates the carry through the
    /// remainder, returning the carry that escapes `dst` entirely.
    ///
    /// `src.len() <= dst.len()` is required; the tail range is otherwise empty and
    /// the carry is reported unchanged.
    #[allow(
        clippy::inline_always,
        reason = "hot path add-with-carry propagation used in CRT merge and fold"
    )]
    #[inline(always)]
    pub fn add_full_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        debug_assert!(
            src.len() <= dst.len(),
            "full-width addition source exceeds destination"
        );
        let carry = Addition::add_slice_in_place(dst, src);
        if carry == 0 {
            return 0;
        }
        #[allow(
            unsafe_code,
            reason = "src.len() <= dst.len() by construction; tail range is in bounds"
        )]
        // SAFETY: the caller's fixed-width contract gives `src.len() <= dst.len()`;
        // equality intentionally yields an empty tail so the escaping carry is
        // returned unchanged.
        let escaped = Self::propagate_carry(unsafe { dst.get_unchecked_mut(src.len()..) });
        Limb::from(escaped)
    }

    /// Subtracts `src` from the prefix of `dst` and propagates the borrow through
    /// the remainder, returning the borrow that escapes `dst` entirely.
    ///
    /// `src.len() <= dst.len()` is required; the tail range is otherwise empty and
    /// the borrow is reported unchanged.
    #[allow(
        clippy::inline_always,
        reason = "hot path borrow propagation used in CRT merge and fold"
    )]
    #[inline(always)]
    pub fn sub_full_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        debug_assert!(
            src.len() <= dst.len(),
            "full-width subtraction source exceeds destination"
        );
        let borrow = Addition::sub_slice_in_place(dst, src);
        if borrow == 0 {
            return 0;
        }
        #[allow(
            unsafe_code,
            reason = "src.len() <= dst.len() by construction; tail range is in bounds"
        )]
        // SAFETY: the caller's fixed-width contract gives `src.len() <= dst.len()`;
        // equality intentionally yields an empty tail so the escaping borrow is
        // returned unchanged.
        let escaped = Self::propagate_borrow(unsafe { dst.get_unchecked_mut(src.len()..) });
        Limb::from(escaped)
    }

    /// Canonicalizes a wrapped negative `ml`-limb difference modulo `2^n + 1`.
    ///
    /// The wrapped subtraction already contributes `2^n`; adding the remaining
    /// `+1` from the modulus either stays in the data limbs or carries into the
    /// canonical guard representation `2^n = -1`.
    ///
    /// # Safety
    /// `dst` contains at least `ml + 1` limbs and its data-limb subtraction
    /// borrowed exactly once.
    #[allow(
        clippy::inline_always,
        reason = "one-instruction guard write on the hot Fermat reduction path"
    )]
    #[inline(always)]
    pub unsafe fn correct_wrapped_shift_difference(dst: &mut [Limb], ml: usize) {
        // SAFETY: caller guarantees dst contains at least ml + 1 limbs.
        let carry = Self::propagate_carry(unsafe { dst.get_unchecked_mut(..ml) });
        // SAFETY: caller guarantees ml is a valid guard index.
        unsafe {
            *dst.get_unchecked_mut(ml) = Limb::from(carry);
        }
    }
}
