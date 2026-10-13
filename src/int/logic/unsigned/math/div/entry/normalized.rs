//! Division dispatch for callers that already own normalized limb workspaces.

#![expect(
    unsafe_code,
    reason = "normalized division validates disjoint initialized slices before entering the raw Newton quotient writer"
)]

use core::cmp::Ordering;

use super::{
    BURNIKEL_ZIEGLER_THRESHOLD, DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DivScratch, Division,
    InternalMpUint, Limb, NEWTON_RAPHSON_THRESHOLD, QuotientAndRemainder, QuotientOnly,
};

impl Division {
    /// Writes a quotient and, with `WRITE_REMAINDER`, replaces the low
    /// divisor-width numerator prefix with its exact remainder, without
    /// copying or renormalizing operands.
    ///
    /// The divisor has at least two limbs and its high bit set. The numerator
    /// has exactly `divisor.len()+quotient.len()` limbs, quotient is nonempty,
    /// and the high divisor-width numerator window is strictly below divisor.
    pub fn div_rem_normalized<const WRITE_REMAINDER: bool>(
        numerator: &mut [Limb],
        divisor: &[Limb],
        quotient: &mut [Limb],
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            divisor.len() >= 2 && !quotient.is_empty(),
            "normalized division requires a multi-limb divisor and quotient storage"
        );
        debug_assert_eq!(
            divisor.last().map(|limb| limb.leading_zeros()),
            Some(0),
            "the divisor's high bit must be set"
        );
        debug_assert_eq!(
            numerator.len().checked_sub(divisor.len()),
            Some(quotient.len()),
            "normalized numerator and output widths must agree"
        );
        debug_assert_eq!(
            InternalMpUint::cmp_limbs(numerator.split_at(quotient.len()).1, divisor),
            Ordering::Less,
            "the leading numerator window must be below the divisor"
        );
        if divisor.len() < BURNIKEL_ZIEGLER_THRESHOLD
            || quotient.len() <= DIVISION_BASECASE_QUOTIENT_MAX_LIMBS
        {
            Self::knuth_d_divide_slice(numerator, divisor, quotient, &mut []);
        } else if divisor.len() < NEWTON_RAPHSON_THRESHOLD {
            scratch.recursive_product.reset_with_capacity(divisor.len());
            // SAFETY: reservation supplies divisor.len() writable spare limbs;
            // recursive repair initializes each product span before reading it.
            let product = unsafe {
                scratch
                    .recursive_product
                    .spare_capacity_mut()
                    .get_unchecked_mut(..divisor.len())
            };
            Self::burnikel_div_rem_normalized::<WRITE_REMAINDER>(
                numerator,
                divisor,
                quotient,
                product,
                &mut scratch.mul_scratch,
            );
        } else {
            // SAFETY: the slice geometry and high-window bound establish the
            // normalized block contract. The exclusive quotient slice has
            // quotient.len() initialized writable limbs disjoint from both
            // operands and scratch; a quotient is always requested.
            unsafe {
                if WRITE_REMAINDER {
                    Self::newton_div_blocks::<QuotientAndRemainder>(
                        numerator,
                        divisor,
                        quotient.as_mut_ptr(),
                        quotient.len(),
                        scratch,
                    );
                } else {
                    Self::newton_div_blocks::<QuotientOnly>(
                        numerator,
                        divisor,
                        quotient.as_mut_ptr(),
                        quotient.len(),
                        scratch,
                    );
                }
            }
        }
    }
}
