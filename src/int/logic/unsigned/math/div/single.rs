//! Single-limb division with a reciprocal shared by descending quotient digits.
//!
//! The leading quotient digit for a normalized divisor is zero or one. Removing it
//! before reciprocal construction avoids a complete 2/1 step. A single
//! short remaining span uses direct division without preparing an inverse.

#![expect(
    unsafe_code,
    reason = "nonzero divisors and initialized pointer spans establish quotient-fit, overlap, and output bounds"
)]

use core::ptr::{copy, null_mut};

use super::{
    ArchKernels, DIVISION_SINGLE_NORMALIZED_PREINVERSE, DIVISION_SINGLE_UNNORMALIZED_PREINVERSE,
    Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb,
};

impl Division {
    /// Divides initialized numerator limbs by a nonzero limb.
    ///
    /// Returns the remainder and, with `WRITE_QUOTIENT`, writes the canonical
    /// quotient. A reciprocal is prepared when the remaining quotient span
    /// reaches the configured cutoff for the divisor's normalization state.
    pub fn div_rem_1<const WRITE_QUOTIENT: bool>(
        num_limbs: &[Limb],
        den_v: Limb,
        quotient_out: &mut InternalMpUint,
    ) -> Limb {
        debug_assert_ne!(den_v, 0, "single-limb division requires a nonzero divisor");
        if WRITE_QUOTIENT {
            let mut output = quotient_out.prepare_limb_write(num_limbs.len());
            // SAFETY: the source is initialized and the disjoint destination
            // reserves num_limbs.len() aligned limbs. The kernel initializes
            // every quotient limb before the guard commits its logical length.
            let rem = unsafe {
                div_rem_1_raw::<true>(
                    num_limbs.as_ptr(),
                    num_limbs.len(),
                    den_v,
                    output.as_mut_ptr(),
                )
            };
            // SAFETY: the raw kernel initialized the entire prepared span.
            let _ = unsafe { output.commit() };
            quotient_out.normalize();
            return rem;
        }
        // SAFETY: the input is initialized and den_v is nonzero. The false
        // specialization never dereferences its null output pointer.
        unsafe { div_rem_1_raw::<false>(num_limbs.as_ptr(), num_limbs.len(), den_v, null_mut()) }
    }

    /// Divides a magnitude in its existing storage by a nonzero limb.
    pub fn div_rem_1_assign(value: &mut InternalMpUint, divisor: Limb) -> Limb {
        debug_assert_ne!(
            divisor, 0,
            "single-limb assignment requires a nonzero divisor"
        );
        let limbs = value.limbs_mut();
        let pointer = limbs.as_mut_ptr();
        // SAFETY: the exclusive slice contains len initialized limbs. Input
        // and output coincide exactly; each source is consumed before writing
        // its output. Power-of-two shifts consume the next limb before changing
        // the lower one, preserving all later reads.
        let rem = unsafe { div_rem_1_raw::<true>(pointer, limbs.len(), divisor, pointer) };
        value.normalize();
        rem
    }
}

/// Divides initialized numerator limbs, optionally overwriting their storage.
///
/// # Safety
/// `num` covers `len` initialized aligned limbs; `divisor` is nonzero.
/// When writing, `quotient` covers `len` writable aligned limbs and either
/// equals `num` or is disjoint from it. Otherwise `quotient` is not accessed.
unsafe fn div_rem_1_raw<const WRITE_QUOTIENT: bool>(
    num: *const Limb,
    len: usize,
    divisor: Limb,
    quotient: *mut Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }
    if divisor.is_power_of_two() {
        let shift = divisor.trailing_zeros();
        // SAFETY: len > 0 proves the low limb exists. The validated
        // nonzero power of two has an exact predecessor for its mask.
        let remainder = unsafe { num.read() & divisor.unchecked_sub(1) };
        if WRITE_QUOTIENT {
            if shift == 0 {
                // SAFETY: both spans cover len limbs; ptr::copy permits
                // the exact overlap admitted by this raw kernel.
                unsafe {
                    copy(num, quotient, len);
                }
            } else {
                // SAFETY: len > 0 and the nonzero power of two gives
                // 0 < shift < Limb::BITS, bounding both differences.
                let (last, complement) =
                    unsafe { (len.unchecked_sub(1), Limb::BITS.unchecked_sub(shift)) };
                for index in 0..last {
                    // SAFETY: index+1 <= last < len. Ascending writes
                    // consume both source limbs before changing the lower
                    // one, preserving exact-overlap input for later reads.
                    unsafe {
                        let low = num.add(index).read();
                        let high = num.add(index.unchecked_add(1)).read();
                        quotient
                            .add(index)
                            .write((low >> shift) | (high << complement));
                    }
                }
                // SAFETY: last < len and earlier writes stop below last.
                unsafe {
                    quotient.add(last).write(num.add(last).read() >> shift);
                }
            }
        }
        return remainder;
    }

    // SAFETY: len > 0 bounds the high input read and subtraction.
    let (last, top) = unsafe {
        let last = len.unchecked_sub(1);
        (last, num.add(last).read())
    };
    // SAFETY: leading_zeros <= 63 fits usize on every supported pointer width.
    let shift = unsafe { usize::try_from(divisor.leading_zeros()).unwrap_unchecked() };
    let mut remaining = len;
    let mut remainder = 0;
    if shift == 0 || top < divisor {
        // D >= B/2 implies top/D <= 1. For an unnormalized divisor,
        // top < D instead proves that its leading quotient digit is zero.
        let digit = Limb::from(top >= divisor);
        // SAFETY: digit is one exactly when top >= divisor; the
        // subtrahend is otherwise zero, so subtraction cannot underflow.
        remainder = unsafe { top.unchecked_sub(if digit == 0 { 0 } else { divisor }) };
        remaining = last;
        if WRITE_QUOTIENT {
            // SAFETY: last < len; the high input was consumed above.
            unsafe {
                quotient.add(last).write(digit);
            }
        }
    }
    if remaining == 0 {
        return remainder;
    }
    let preinverse_cutoff = if shift == 0 {
        DIVISION_SINGLE_NORMALIZED_PREINVERSE
    } else {
        DIVISION_SINGLE_UNNORMALIZED_PREINVERSE
    };
    if remaining < preinverse_cutoff {
        for index in (0..remaining).rev() {
            // SAFETY: index < remaining <= len. The preceding division
            // leaves remainder < divisor, so the quotient fits one limb.
            // Each input is consumed before its exactly overlapping output.
            unsafe {
                let (digit, rem) =
                    ArchKernels::divrem_1_unchecked(num.add(index).read(), remainder, divisor);
                remainder = rem;
                if WRITE_QUOTIENT {
                    quotient.add(index).write(digit);
                }
            }
        }
        return remainder;
    }
    let normalized = divisor << shift;
    // SAFETY: a nonzero divisor has shift < LIMB_BITS. The complement
    // equals LIMB_BITS only for shift=0, where the loop never shifts by it.
    let complement = unsafe { LIMB_BITS.unchecked_sub(shift) };
    // SAFETY: normalized >= B/2 implies !normalized < normalized.
    let (reciprocal, _) =
        unsafe { ArchKernels::divrem_1_unchecked(Limb::MAX, !normalized, normalized) };
    // R < D implies R<<shift < B. Its low shift bits are zero; adding
    // the next input's high bits by OR preserves the normalized bound.
    remainder <<= shift;
    for i in (0..remaining).rev() {
        // SAFETY: i < remaining <= len. Exact overlap is valid because
        // the input is consumed before the corresponding quotient write.
        let limb = unsafe { num.add(i).read() };
        let (high, low) = if shift == 0 {
            (remainder, limb)
        } else {
            (remainder | (limb >> complement), limb << shift)
        };
        let (digit, next) = Division::divrem_2by1_reciprocal(high, low, normalized, reciprocal);
        remainder = next;
        if WRITE_QUOTIENT {
            // SAFETY: i < len and the destination covers len writable limbs.
            unsafe {
                quotient.add(i).write(digit);
            }
        }
    }
    remainder >> shift
}

impl Division {
    /// Divides `(u1*B + u0)` by a normalized limb using
    /// `v = floor((B²-1)/d)-B`, with `B = 2^LIMB_BITS` and `u1 < d`.
    ///
    /// Adding one to the reciprocal estimate places it within one of the exact
    /// quotient. The low reciprocal product distinguishes a negative residue
    /// from its limb-radix wrap; one final upward correction leaves `0 <= r < d`.
    ///
    /// Reference: Möller–Granlund, "Improved Division by Invariant Integers",
    /// Section II, Equation (3), DOI: 10.1109/TC.2010.143.
    #[expect(
        clippy::as_conversions,
        clippy::inline_always,
        reason = "DoubleLimb embeds native limbs exactly; extracting the reciprocal product and inlining preserve the scalar division recurrence"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "extracting the low and high halves of a double-limb product is reduction modulo the native limb radix"
        )
    )]
    #[inline(always)]
    pub fn divrem_2by1_reciprocal(u1: Limb, u0: Limb, d: Limb, v: Limb) -> (Limb, Limb) {
        debug_assert!(u1 < d, "the quotient must fit a limb");
        debug_assert!(d >> (Limb::BITS - 1) != 0, "normalized divisor");
        // e:l = (B+v)*u1+u0, q_hat=e+1. Algorithm 4 bounds q_hat-q to
        // {-1,0,1} and proves r>l detects the downward correction from
        // r=(u0-q_hat*d) mod B. q_hat may equal B: its modular encoding zero
        // becomes B-1 on the required decrement. One final correction suffices.
        // SAFETY: two native limbs multiply to less than B^2, which fits
        // DoubleLimb on all supported pointer widths.
        let product = unsafe { (v as DoubleLimb).unchecked_mul(u1 as DoubleLimb) };
        let high = (product >> LIMB_BITS) as Limb;
        let low = product as Limb;
        let (bound, carry) = low.overflowing_add(u0);
        // SAFETY: K=B+v=floor((B^2-1)/d) >= B and u1<d give
        // u1*K+u0 <= d*K-K+B-1 < B^2. Its high limb therefore fits.
        let estimate = unsafe { u1.unchecked_add(high).unchecked_add(Limb::from(carry)) };
        // The estimate plus one may equal B, encoded as zero modulo B.
        let mut quotient = estimate.wrapping_add(1);
        let mut remainder = u0.wrapping_sub(quotient.wrapping_mul(d));
        if remainder > bound {
            quotient = quotient.wrapping_sub(1);
            remainder = remainder.wrapping_add(d);
        }
        if remainder >= d {
            // SAFETY: the corrected estimate is at most floor(U/d). This
            // residue proves it is smaller; u1<d bounds the exact quotient
            // below B. The branch also proves remainder >= d.
            unsafe {
                quotient = quotient.unchecked_add(1);
                remainder = remainder.unchecked_sub(d);
            }
        }
        (quotient, remainder)
    }
}
