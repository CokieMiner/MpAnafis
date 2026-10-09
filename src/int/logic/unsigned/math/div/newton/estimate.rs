//! Quotient estimation and exact-division correction for Newton blocks.

#![expect(
    unsafe_code,
    reason = "normalized block geometry bounds reciprocal digits and initialized quotient spans with a zero high guard"
)]

use core::num::NonZeroUsize;

use super::{Addition, DivScratch, Division, HighProduct, Limb};

impl Division {
    /// Forms `floor(H*V/B^k)` for a k-limb prefix and `V=B^k+V0`.
    ///
    /// Only the exact high half of H*V0 is multiplied; the leading V digit
    /// adds H. The nonempty prefix and reciprocal satisfy H*V<B^(2k), so
    /// the sum fits k limbs and its appended high guard is zero. The driver
    /// takes all k digits for an ordinary block or discards the returned low
    /// digit for a quotient-only block certified at one extra radix digit.
    pub fn newton_quotient_estimate(
        high: &[Limb],
        inverse: &[Limb],
        scratch: &mut DivScratch,
    ) -> Limb {
        // SAFETY: the block driver supplies k=quo_len+usize::from(GUARDED)>0
        // and k+1 reciprocal limbs. Its disjoint normalized window obeys
        // U<D*B^quo_len; the lower reciprocal bound implies H*V<B^(2k).
        let block = unsafe { NonZeroUsize::new_unchecked(high.len()) }.get();
        debug_assert_eq!(
            inverse.len().checked_sub(1),
            Some(block),
            "the reciprocal contains k low digits and its implicit leading one"
        );
        debug_assert_eq!(
            inverse.last(),
            Some(&1),
            "the normalized reciprocal is in [B^k, 2B^k)"
        );
        // SAFETY: inverse contains k initialized low limbs and its leading one;
        // the input owners are disjoint from the product and carry workspaces.
        let inverse_low = unsafe { inverse.get_unchecked(..block) };
        let product = HighProduct::mul(
            high,
            inverse_low,
            block,
            &mut scratch.v_padded,
            &mut scratch.q_den_low,
            &mut scratch.mul_scratch,
        );
        let carry = Addition::add_slice_in_place(product, high);
        debug_assert_eq!(carry, 0, "the lower scaled estimate fits k limbs");
        // SAFETY: the exact initialized high half contains k>0 digits.
        let low_digit = unsafe { *product.get_unchecked(0) };
        let initialized_len = scratch.v_padded.len();
        // SAFETY: the high-product writer reserved one additional carry slot;
        // H*V<B^(2k) proves its first value is zero. Initialize it before
        // extending the length; materialized widths plus one fit every target.
        unsafe {
            scratch.v_padded.as_mut_ptr().add(initialized_len).write(0);
            scratch.v_padded.set_len(initialized_len.unchecked_add(1));
        }
        low_digit
    }

    /// Corrects the last quotient block of a known exact division.
    ///
    /// Let D=2^s*d with d odd, U divisible by D, and Q-Q0 in {0,1,2,3}.
    /// Then ((U/2^s)*d-Q0) mod 4 determines Q-Q0 exactly because d*d=1
    /// modulo four. Only two bits above the divisor's valuation are read;
    /// no full-width product or remainder is formed.
    /// The quotient occupies `v_padded[start..]`, including its zero high guard.
    pub fn newton_exact_correction(
        num: &[Limb],
        den: &[Limb],
        start: usize,
        scratch: &mut DivScratch,
    ) {
        // SAFETY: a normalized nonzero divisor contains a nonzero limb.
        let index = unsafe { den.iter().position(|&limb| limb != 0).unwrap_unchecked() };
        // SAFETY: index < den.len() < num.len() by the block geometry.
        let (den_low, num_low) = unsafe { (*den.get_unchecked(index), *num.get_unchecked(index)) };
        let shift = den_low.trailing_zeros();
        let mut odd = den_low >> shift;
        let mut reduced_num = num_low >> shift;
        if shift == Limb::BITS - 1 {
            // Two bits straddle this limb boundary. The extra dividend limb
            // exists even when the divisor's first nonzero limb is its last.
            // SAFETY: index < den.len() < num.len() bounds index + 1.
            let next = unsafe { index.unchecked_add(1) };
            // SAFETY: the same strict length bounds prove next < num.len().
            reduced_num |= unsafe { *num.get_unchecked(next) } << 1;
            if next < den.len() {
                // SAFETY: the comparison proves the divisor's next limb exists.
                odd |= unsafe { *den.get_unchecked(next) } << 1;
            }
        }
        // SAFETY: the estimator retains at least one quotient limb and its
        // initialized high guard after the retained high-product workspace.
        let low_quotient = unsafe { *scratch.v_padded.get_unchecked(start) };
        // Multiplying U/2^s=Q*d by d gives Q modulo four because d^2=1.
        // The estimate error is in {0,1,2,3}, so these two bits recover it exactly.
        let correction = reduced_num.wrapping_mul(odd).wrapping_sub(low_quotient) & 3;
        if correction != 0 {
            // SAFETY: start precedes the initialized quotient and high guard.
            // The exact quotient fits its block, so the correction cannot
            // carry beyond this span, including for a zero estimate.
            let quotient = unsafe { scratch.v_padded.get_unchecked_mut(start..) };
            // SAFETY: the quotient span includes a low digit and its high
            // guard, so splitting its first digit cannot fail.
            let (low, higher) = unsafe { quotient.split_first_mut().unwrap_unchecked() };
            let (sum, overflow) = low.overflowing_add(correction);
            *low = sum;
            let carry = Addition::propagate_carry(higher, Limb::from(overflow));
            debug_assert_eq!(carry, 0, "the corrected quotient fits its block");
        }
    }
}
