//! Carry and borrow propagation for modular arithmetic and reconstruction.
//!
//! Modular callers retain escaping flags; bounded reconstruction proves their
//! absorption. Full-width addition and subtraction propagate through any tail
//! beyond the shorter source.
//!
#![expect(
    unsafe_code,
    reason = "Fixed-width sources bound tail slices; accumulator contracts prove carry or borrow absorption before pointer escape"
)]

use super::{Addition, Limb, LimbOutput};

/// Namespace for carry and borrow propagation shared across the whole SSA tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaCarry;

impl SsaCarry {
    /// Adds one to `limbs`, returning `true` when the carry escapes the slice.
    ///
    /// An empty slice cannot absorb the carry, so it reports `true`.
    #[expect(
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
    #[expect(
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

    /// Absorbs a carry in an accumulator whose complete sum fits its width.
    ///
    /// # Safety
    /// The initialized slice contains a limb below `Limb::MAX`. Therefore the
    /// first such limb absorbs the carry before the cursor reaches the end.
    #[inline]
    pub unsafe fn absorb_carry(limbs: &mut [Limb]) {
        let mut cursor = limbs.as_mut_ptr();
        loop {
            // SAFETY: all visited prefix limbs were maximum; the contract
            // supplies a later initialized limb below maximum in this slice.
            let limb = unsafe { &mut *cursor };
            let (sum, overflow) = limb.overflowing_add(1);
            *limb = sum;
            if !overflow {
                return;
            }
            // SAFETY: this maximum limb did not absorb the carry, so the
            // contract's first nonmaximum limb remains strictly ahead.
            cursor = unsafe { cursor.add(1) };
        }
    }

    /// Absorbs a borrow in a strictly positive accumulator tail.
    ///
    /// # Safety
    /// The initialized slice contains a nonzero limb. The first such limb
    /// absorbs the borrow before the cursor reaches the end.
    #[inline]
    pub unsafe fn absorb_borrow(limbs: &mut [Limb]) {
        let mut cursor = limbs.as_mut_ptr();
        loop {
            // SAFETY: all visited prefix limbs were zero; the contract
            // supplies a later initialized nonzero limb in this slice.
            let limb = unsafe { &mut *cursor };
            let (difference, underflow) = limb.overflowing_sub(1);
            *limb = difference;
            if !underflow {
                return;
            }
            // SAFETY: this zero limb did not absorb the borrow, so the
            // contract's first nonzero limb remains strictly ahead.
            cursor = unsafe { cursor.add(1) };
        }
    }

    /// Adds `src` into the prefix of `dst` and propagates the carry through the
    /// remainder, returning the carry that escapes `dst` entirely.
    ///
    /// Requires `src.len() <= dst.len()`. Equal widths return the escaping carry.
    #[expect(
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
        // SAFETY: the caller's fixed-width contract gives `src.len() <= dst.len()`;
        // equality intentionally yields an empty tail so the escaping carry is
        // returned unchanged.
        let escaped = Self::propagate_carry(unsafe { dst.get_unchecked_mut(src.len()..) });
        Limb::from(escaped)
    }

    /// Subtracts `src` from the prefix of `dst` and propagates the borrow through
    /// the remainder, returning the borrow that escapes `dst` entirely.
    ///
    /// Requires `src.len() <= dst.len()`. Equal widths return the escaping borrow.
    #[expect(
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
        // SAFETY: the caller's fixed-width contract gives `src.len() <= dst.len()`;
        // equality intentionally yields an empty tail so the escaping borrow is
        // returned unchanged.
        let escaped = Self::propagate_borrow(unsafe { dst.get_unchecked_mut(src.len()..) });
        Limb::from(escaped)
    }

    /// Canonicalizes a wrapped negative `ml`-limb difference modulo `2^n + 1`
    /// and returns whether the canonical residue is `-1`.
    ///
    /// The wrapped subtraction already contributes `2^n`; adding the remaining
    /// `+1` from the modulus either stays in the data limbs or carries into the
    /// canonical guard representation `2^n = -1`.
    ///
    /// # Safety
    /// `dst` contains at least `ml + 1` elements; its `ml` data limbs are
    /// initialized and their subtraction borrowed exactly once. The guard is
    /// writable and need not yet be initialized.
    #[expect(
        clippy::inline_always,
        reason = "one-instruction guard write on the hot Fermat reduction path"
    )]
    #[inline(always)]
    pub unsafe fn correct_wrapped_shift_difference(dst: &mut [impl LimbOutput], ml: usize) -> bool {
        // SAFETY: the caller supplies ml initialized data limbs and a writable
        // guard. Only the data is borrowed as limbs before the guard's first write.
        let carry = Self::propagate_carry(unsafe {
            LimbOutput::assume_init_mut(dst.get_unchecked_mut(..ml))
        });
        // SAFETY: caller guarantees ml is a valid guard index.
        unsafe {
            *dst.get_unchecked_mut(ml) = LimbOutput::from_limb(Limb::from(carry));
        }
        carry
    }
}
