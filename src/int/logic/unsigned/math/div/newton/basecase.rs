//! Exact reciprocal seed from a complemented normalized numerator.
//!
//! For F=B^(2n)-1, floor(F/D)=B^n+floor((F-D*B^n)/D).
//! The residual numerator has n low limbs equal to B-1 and high half !D.
//! Normalization gives !D<D, so its quotient fits n limbs without a high
//! guard, an operand shift, or another normalization copy.

#![expect(
    unsafe_code,
    reason = "normalized divisors bound complemented numerator windows and the reciprocal's fixed leading-one output"
)]

use core::mem::replace;

use super::{ArchKernels, DivScratch, Division, InternalMpUint, Limb, PreparedDivisor};

impl Division {
    /// Returns floor((B^(2n)-1)/D) for a normalized nonempty n-limb divisor.
    pub fn reciprocal_basecase(den: &[Limb], scratch: &mut DivScratch) -> InternalMpUint {
        let n = den.len();
        let mut quotient = replace(&mut scratch.dummy_rem, InternalMpUint::zero());
        // SAFETY: a materialized nonempty Limb slice admits one additional limb.
        let width = unsafe { n.unchecked_add(1) };
        let digits = quotient.ensure_capacity_set_len_get_limbs(width);
        // SAFETY: width=n+1 bounds the exclusive initialized output partitions.
        let (lower, leading) = unsafe { digits.split_at_mut_unchecked(n) };
        // SAFETY: leading contains exactly one initialized output limb.
        unsafe {
            *leading.get_unchecked_mut(0) = 1;
        }
        if let &[divisor] = den {
            // SAFETY: D>=B/2 implies !D<D; the hardware quotient fits one
            // limb. The n=1 output has one initialized low digit.
            unsafe {
                let (inverse, _) = ArchKernels::divrem_1_unchecked(Limb::MAX, !divisor, divisor);
                *lower.get_unchecked_mut(0) = inverse;
            }
        } else if let &[low, high] = den {
            let inverse = Self::invert_pi1(high, low);
            let (mut upper, mut next) = (!high, !low);
            for digit in lower.iter_mut().rev() {
                // !D<D establishes the first quotient-fit bound; each exact
                // 3/2 step preserves it for the following all-ones input limb.
                (*digit, upper, next) =
                    Self::udiv_qr_3by2(upper, next, Limb::MAX, high, low, inverse);
            }
        } else {
            // SAFETY: n>=3 and n<=isize::MAX/size_of::<Limb>() imply 2n fits
            // usize on every pointer width. Reservation supplies that capacity.
            let numerator_len = unsafe { n.unchecked_mul(2) };
            scratch.u_norm.reset_with_capacity(numerator_len);
            // SAFETY: the reserved aligned span covers 2n limbs, disjoint
            // from den and lower. All one bits initialize its low n limbs;
            // complementing D initializes its high n before exposing the span.
            unsafe {
                let pointer = scratch.u_norm.as_mut_ptr();
                pointer.write_bytes(u8::MAX, n);
                for (index, &limb) in den.iter().enumerate() {
                    pointer.add(n.unchecked_add(index)).write(!limb);
                }
                scratch.u_norm.set_len(numerator_len);
            }
            // !D<D and !den[n-1]<den[n-1] establish the normalized short
            // kernel's high-window and guard bounds without a leading zero.
            let prepared = PreparedDivisor::new(den);
            let _ =
                prepared.divide_quotient::<false, false, false>(&mut scratch.u_norm, den, lower);
        }
        quotient
    }
}
