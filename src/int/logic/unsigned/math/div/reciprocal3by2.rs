//! The Möller–Granlund 3-by-2 division primitive.
//!
//! [`Division::invert_pi1`] prepares the normalized leading divisor pair.
//! [`Division::udiv_qr_3by2`] uses that reciprocal to compute each quotient
//! digit and two-limb remainder with multiplication and bounded correction.
//!
//! References:
//! - N. Möller and T. Granlund, "Improved Division by Invariant Integers", IEEE Transactions
//!   on Computers, Vol. 60, No. 2, pp. 165–175, Feb. 2011, Algorithms 5 and 6. DOI: 10.1109/TC.2010.143.

#![expect(
    unsafe_code,
    reason = "normalized leading limbs bound hardware quotients and callers provide initialized division windows"
)]

use super::{ArchKernels, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb};

impl Division {
    /// Divides a canonical numerator by a canonical two-limb divisor.
    ///
    /// The numerator has at least two limbs. Normalization is evaluated during
    /// input reads; only the two-limb remainder is retained between digits.
    /// A provably zero leading digit is excluded before allocating the quotient.
    pub fn div_rem_2_unnormalized<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool>(
        numerator: &[Limb],
        divisor_low: Limb,
        divisor_high: Limb,
        quotient: &mut InternalMpUint,
        remainder: &mut InternalMpUint,
    ) {
        let length = numerator.len();
        debug_assert!(
            length >= 2 && divisor_high != 0,
            "the dispatcher proves both operand widths"
        );
        let shift = divisor_high.leading_zeros();
        // SAFETY: divisor_high != 0 gives shift < Limb::BITS. The full-width
        // complement for shift=0 is never used as a shift count.
        let complement = unsafe { Limb::BITS.unchecked_sub(shift) };
        // SAFETY: length >= 2 bounds the two initialized high input limbs
        // and both subtractions. A canonical numerator has a nonzero top.
        let (top, next) = unsafe {
            (
                *numerator.get_unchecked(length.unchecked_sub(1)),
                *numerator.get_unchecked(length.unchecked_sub(2)),
            )
        };
        // U=P*B^(length-2)+L with 0<=L<B^(length-2). The leading
        // quotient digit is zero exactly when P=top*B+next<D. Comparing
        // the complete pair excludes that digit before normalization.
        let leading_zero = length > 2 && (top, next) < (divisor_high, divisor_low);
        let (upper, lower) = if leading_zero { (top, next) } else { (0, top) };
        let (d1, d0, mut r1, mut r0) = if shift == 0 {
            (divisor_high, divisor_low, upper, lower)
        } else {
            (
                (divisor_high << shift) | (divisor_low >> complement),
                divisor_low << shift,
                (upper << shift) | (lower >> complement),
                lower << shift,
            )
        };
        // SAFETY: length >= 2; removing a zero digit requires length > 2.
        // At least one quotient slot remains, and both subtractions fit usize.
        let digits = unsafe {
            length
                .unchecked_sub(1)
                .unchecked_sub(usize::from(leading_zero))
        };
        let inverse = Self::invert_pi1(d1, d0);
        (r1, r0) = if WRITE_QUOTIENT {
            let mut output = quotient.prepare_limb_write(digits);
            // SAFETY: the guard reserves digits aligned writable limbs,
            // disjoint from numerator. The raw loop initializes every slot;
            // its initial high pair is below D, preserved by each exact step.
            let result = unsafe {
                div_rem_2_raw::<true>(
                    numerator,
                    digits,
                    shift,
                    d1,
                    d0,
                    inverse,
                    r1,
                    r0,
                    output.as_mut_ptr(),
                )
            };
            // SAFETY: the descending loop initialized all digits slots.
            let _ = unsafe { output.commit() };
            quotient.normalize();
            result
        } else {
            // SAFETY: no output pointer is accessed in this specialization.
            // Input bounds and the initial remainder are identical to the
            // quotient-producing branch.
            unsafe {
                div_rem_2_raw::<false>(
                    numerator,
                    digits,
                    shift,
                    d1,
                    d0,
                    inverse,
                    r1,
                    r0,
                    [].as_mut_ptr(),
                )
            }
        };
        if WRITE_REMAINDER {
            let denormalized = if shift == 0 {
                [r0, r1]
            } else {
                [(r0 >> shift) | (r1 << complement), r1 >> shift]
            };
            remainder.clone_from_slice(&denormalized);
        }
    }

    /// Computes the 3-by-2 reciprocal of the normalized two-limb divisor top
    /// `(d1, d0)`, where `d1`'s most-significant bit is set.
    ///
    /// The result is `floor((B^3 - 1) / (d1 * B + d0)) - B` (with `B` the limb
    /// radix), the value [`Division::udiv_qr_3by2`] consumes.
    #[expect(
        clippy::inline_always,
        reason = "Computed once per division on the hot path; inlining keeps the divisor top in registers."
    )]
    #[expect(
        clippy::as_conversions,
        reason = "Widening Limb is exact; truncation extracts the low radix-B digit and shifting by LIMB_BITS extracts the high digit"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "Both casts extract one native limb from a product below B^2"
        )
    )]
    #[inline(always)]
    pub fn invert_pi1(d1: Limb, d0: Limb) -> Limb {
        // 2/1 inverse of d1: floor((B^2 - 1) / d1) - B. Because d1 is normalized
        // the quotient lies in [B, 2B). Its high digit is therefore one. After
        // consuming that digit, the remaining high numerator `B - 1 - d1` is
        // strictly below normalized `d1`, satisfying the hardware 2/1 kernel.
        // SAFETY: normalization gives d1 >= B/2, hence
        // `Limb::MAX - d1 < d1`; d1 is also nonzero.
        let (mut v, _) = unsafe { ArchKernels::divrem_1_unchecked(Limb::MAX, !d1, d1) };

        let mut p = d1.wrapping_mul(v).wrapping_add(d0);
        if p < d0 {
            // SAFETY: d1 <= B-1 makes the initial reciprocal at least one.
            v = unsafe { v.unchecked_sub(1) };
            let mask = if p >= d1 { Limb::MAX } else { 0 };
            p = p.wrapping_sub(d1);
            // The mask is zero or the encoding of -1 modulo B.
            v = v.wrapping_add(mask);
            p = p.wrapping_sub(mask & d1);
        }

        // SAFETY: two native limbs have a product below B^2, fitting DoubleLimb.
        let prod = unsafe { (d0 as DoubleLimb).unchecked_mul(v as DoubleLimb) };
        let t1 = (prod >> LIMB_BITS) as Limb;
        let t0 = prod as Limb;
        p = p.wrapping_add(t1);
        if p < t1 {
            // SAFETY: p<t1 gives t1>0, hence d0*v >= B. Since d0<B,
            // v >= 2, so both possible decrements have positive inputs.
            v = unsafe { v.unchecked_sub(1) };
            if p >= d1 && (p > d1 || t0 >= d0) {
                // SAFETY: the outer condition proved v >= 2 before the first decrement.
                v = unsafe { v.unchecked_sub(1) };
            }
        }
        v
    }

    /// Divides a normalized Algorithm D numerator by a two-limb divisor.
    ///
    /// `num` contains `m + 3` limbs with its top pair below `(d1, d0)`.
    /// `quo` is empty or has exactly `m + 1` limbs for the complete quotient.
    /// Only the final two-limb remainder is written back into `num`.
    pub fn div_rem_2(num: &mut [Limb], quo: &mut [Limb], m: usize, d1: Limb, d0: Limb, dinv: Limb) {
        // SAFETY: the caller supplies m + 3 initialized numerator limbs.
        let (mut r1, mut r0, digits) = unsafe {
            let digits = m.unchecked_add(1);
            (
                *num.get_unchecked(m.unchecked_add(2)),
                *num.get_unchecked(digits),
                digits,
            )
        };
        debug_assert!(
            (r1, r0) < (d1, d0),
            "initial remainder must be below the two-limb divisor"
        );
        if quo.is_empty() {
            for i in (0..digits).rev() {
                // Each preceding remainder is below D, so the digit fits.
                // SAFETY: i<digits=m+1<num.len(); remainder stores follow
                // all numerator reads and no quotient pointer is accessed.
                let low = unsafe { *num.get_unchecked(i) };
                (_, r1, r0) = Self::udiv_qr_3by2(r1, r0, low, d1, d0, dinv);
            }
        } else {
            debug_assert_eq!(quo.len(), digits, "the complete quotient span is required");
            for (i, digit) in quo.iter_mut().enumerate().rev() {
                // SAFETY: quo.len()=digits=m+1 bounds i within num. The
                // preceding remainder is below D; disjoint quotient stores
                // preserve every unread numerator limb.
                let low = unsafe { *num.get_unchecked(i) };
                (*digit, r1, r0) = Self::udiv_qr_3by2(r1, r0, low, d1, d0, dinv);
            }
        }
        // SAFETY: m + 3 >= 3; all numerator reads are complete. These distinct
        // initialized slots hold the exact remainder consumed by the caller.
        unsafe {
            *num.get_unchecked_mut(0) = r0;
            *num.get_unchecked_mut(1) = r1;
        }
    }
}

/// Consumes normalized input limbs with a retained two-limb remainder.
///
/// # Safety
/// The canonical input has at least two limbs and `digits` is its length
/// minus one or two. The initial pair is below the normalized divisor;
/// `inverse` is its 3-by-2 reciprocal and `shift < Limb::BITS`.
/// Both `d0` and the initial `r0` are divisible by `2^shift`.
/// With `WRITE_QUOTIENT`, `output` covers `digits` aligned writable limbs
/// disjoint from the input. Otherwise `output` is never accessed.
#[expect(
    clippy::too_many_arguments,
    reason = "the scalar divisor, inverse, shift and initial remainder stay invariant across the direct input loop"
)]
unsafe fn div_rem_2_raw<const WRITE_QUOTIENT: bool>(
    numerator: &[Limb],
    digits: usize,
    shift: u32,
    d1: Limb,
    d0: Limb,
    inverse: Limb,
    mut r1: Limb,
    mut r0: Limb,
    output: *mut Limb,
) -> (Limb, Limb) {
    if shift == 0 {
        for index in (0..digits).rev() {
            // SAFETY: index < digits < numerator.len(). The initial
            // pair and each preceding exact remainder are below D.
            let limb = unsafe { *numerator.get_unchecked(index) };
            let (digit, high, low) = Division::udiv_qr_3by2(r1, r0, limb, d1, d0, inverse);
            if WRITE_QUOTIENT {
                // SAFETY: the caller reserves digits output limbs and
                // index < digits; each slot is initialized exactly once.
                unsafe {
                    output.add(index).write(digit);
                }
            }
            r1 = high;
            r0 = low;
        }
    } else {
        // SAFETY: the normalization caller and this branch give
        // 0 < shift < Limb::BITS, so the complement is also sub-limb.
        let complement = unsafe { Limb::BITS.unchecked_sub(shift) };
        for index in (0..digits).rev() {
            // SAFETY: index < digits < numerator.len(); the input owner
            // initialized this limb and the output belongs to another owner.
            let limb = unsafe { *numerator.get_unchecked(index) };
            // D and the retained remainder are multiples of 2^shift.
            // Their positive gap is at least 2^shift, so inserting the
            // next high bits (<2^shift) preserves R < D. The next 3/2
            // numerator and its exact remainder again have zero low bits.
            r0 |= limb >> complement;
            let (digit, high, low) = Division::udiv_qr_3by2(r1, r0, limb << shift, d1, d0, inverse);
            if WRITE_QUOTIENT {
                // SAFETY: index < digits bounds the exclusive output
                // span; each descending iteration initializes one slot.
                unsafe {
                    output.add(index).write(digit);
                }
            }
            r1 = high;
            r0 = low;
        }
    }
    (r1, r0)
}

impl Division {
    /// Divides three numerator limbs by two normalized divisor limbs.
    ///
    /// Returns the quotient digit and remainder pair. The reciprocal is
    /// `dinv = floor((B^3-1)/(d1*B+d0))-B`, with `B = 2^LIMB_BITS`.
    /// Preconditions are `(n2,n1) < (d1,d0)` and `d1 >= B/2`.
    #[expect(
        clippy::inline_always,
        reason = "Called once per quotient digit on the division hot path; inlining removes the call and exposes the multiplies to the surrounding loop."
    )]
    #[expect(
        clippy::as_conversions,
        reason = "Widening Limb is exact; truncation extracts the low radix-B digit and shifting by LIMB_BITS extracts the high digit"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "Each cast extracts one native limb from a product below B^2"
        )
    )]
    #[inline(always)]
    pub fn udiv_qr_3by2(
        n2: Limb,
        n1: Limb,
        n0: Limb,
        d1: Limb,
        d0: Limb,
        dinv: Limb,
    ) -> (Limb, Limb, Limb) {
        // q1:q0 = n2 * dinv + (n2:n1), giving the initial quotient estimate.
        // SAFETY: two native limbs have a product below B^2, fitting DoubleLimb.
        let prod = unsafe { (n2 as DoubleLimb).unchecked_mul(dinv as DoubleLimb) };
        let mut q1 = (prod >> LIMB_BITS) as Limb;
        let (q0, carry_q) = (prod as Limb).overflowing_add(n1);
        // SAFETY: K=B+dinv >= B and K*(d1*B+d0) <= B^3-1 imply
        // K*d1+d0 < B^2. The strict leading-pair bound therefore gives
        // K*n2+n1 < B^2, whose high limb is this exact sum.
        q1 = unsafe { q1.unchecked_add(n2).unchecked_add(Limb::from(carry_q)) };
        // (r1:r0) = (n1:n0) - (q1+1)*(d1:d0), modulo B².
        let mut r1 = n1.wrapping_sub(q1.wrapping_mul(d1));
        let (mut r0, borrow_d) = n0.overflowing_sub(d0);
        r1 = r1.wrapping_sub(d1).wrapping_sub(Limb::from(borrow_d));
        // SAFETY: both factors are native limbs, so the full product fits DoubleLimb.
        let prod_d0 = unsafe { (q1 as DoubleLimb).unchecked_mul(d0 as DoubleLimb) };
        let t1 = (prod_d0 >> LIMB_BITS) as Limb;
        let t0 = prod_d0 as Limb;
        let (r0b, borrow_t) = r0.overflowing_sub(t0);
        r0 = r0b;
        r1 = r1.wrapping_sub(t1).wrapping_sub(Limb::from(borrow_t));
        // The provisional digit may equal B, represented as zero modulo B.
        q1 = q1.wrapping_add(1);
        // r1 >= q0 detects a negative residue. Undo one quotient unit
        // and add D; the mask encodes either zero or minus one modulo B.
        let mask = if r1 >= q0 { Limb::MAX } else { 0 };
        q1 = q1.wrapping_add(mask);
        let (r0c, carry_r) = r0.overflowing_add(mask & d0);
        r0 = r0c;
        r1 = r1.wrapping_add(mask & d1).wrapping_add(Limb::from(carry_r));
        // At most one upward quotient correction leaves the residue below D.
        if r1 >= d1 && (r1 > d1 || r0 >= d0) {
            // SAFETY: the remaining residue proves q1 is below the exact
            // quotient, which the strict leading-pair bound keeps below B.
            q1 = unsafe { q1.unchecked_add(1) };
            let (r0d, borrow_f) = r0.overflowing_sub(d0);
            r0 = r0d;
            // SAFETY: R >= D. At r1==d1, the branch comparison excludes a
            // low borrow; r1>d1 otherwise leaves room for that binary borrow.
            r1 = unsafe { r1.unchecked_sub(d1).unchecked_sub(Limb::from(borrow_f)) };
        }
        (q1, r1, r0)
    }
}
