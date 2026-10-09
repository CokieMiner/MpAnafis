//! Division shortcuts for powers of two and equal-width scalar quotients.

#![expect(
    unsafe_code,
    reason = "canonical equal-width operands bound leading loads and admitted scalar corrections bound multiply-subtract spans"
)]

use core::{cmp::Ordering, num::NonZeroUsize, ops::Div, ptr::copy_nonoverlapping};

use super::{Addition, ArchKernels, DIVISION_SMALL_QUOTIENT_MAX, Division, InternalMpUint, Limb};

impl Division {
    /// Resolves division by a power of two, leaving both outputs untouched
    /// when the divisor has another set bit.
    ///
    /// The divisor is canonical and nonzero; the numerator is at least as
    /// wide. For `D = 2^s * B^k`, the quotient reads only limbs `k..` and
    /// the remainder reads only the first k limbs and s low bits. Keeping k
    /// and s separate avoids overflowing a bit count on 16-bit targets.
    pub fn power_of_two<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool>(
        numerator: &InternalMpUint,
        divisor: &InternalMpUint,
        quotient: &mut InternalMpUint,
        remainder: &mut InternalMpUint,
    ) -> bool {
        let den = divisor.limbs();
        // SAFETY: the caller validates a canonical nonzero divisor.
        let (&top, lower) = unsafe { den.split_last().unwrap_unchecked() };
        if !top.is_power_of_two() || lower.iter().any(|&limb| limb != 0) {
            return false;
        }
        let num = numerator.limbs();
        let whole = lower.len();
        let shift = top.trailing_zeros();
        if WRITE_QUOTIENT {
            // SAFETY: whole < den.len() <= num.len(), so this canonical
            // suffix exists and still ends at the numerator's nonzero top.
            let source = unsafe { num.get_unchecked(whole..) };
            if shift == 0 {
                quotient.clone_from_slice(source);
            } else {
                let mut output = quotient.prepare_limb_write(source.len());
                // SAFETY: 0<shift<Limb::BITS. The canonical suffix is
                // nonempty and initialized; the disjoint output guard reserves
                // source.len() writable limbs. The shift initializes that
                // complete span directly before its logical length is committed.
                unsafe {
                    let _ = ArchKernels::rshift_into_unchecked(
                        output.as_mut_ptr(),
                        source.as_ptr(),
                        source.len(),
                        shift,
                    );
                    let _ = output.commit();
                }
                quotient.normalize();
            }
        }
        if WRITE_REMAINDER {
            // SAFETY: whole+1 == den.len() <= num.len(); the optional
            // partial limb therefore fits both the input and usize.
            let width = unsafe { whole.unchecked_add(usize::from(shift != 0)) };
            let mut output = remainder.prepare_limb_write(width);
            // SAFETY: the initialized numerator and exclusive write guard
            // cover width aligned limbs and have disjoint owners. The copy
            // initializes whole low limbs; the masked source limb initializes
            // the optional final slot without copying or rereading that slot.
            // A nonzero shift is below Limb::BITS, so 1<<shift is positive.
            unsafe {
                let destination = output.as_mut_ptr();
                copy_nonoverlapping(num.as_ptr(), destination, whole);
                if shift != 0 {
                    destination
                        .add(whole)
                        .write(*num.get_unchecked(whole) & (1_usize << shift).unchecked_sub(1));
                }
                let _ = output.commit();
            }
            remainder.normalize();
        }
        true
    }

    /// Computes an equal-width division whose quotient fits a small scalar.
    ///
    /// The caller supplies canonical nonzero operands of equal limb count.
    /// Let e be the leading-limb ratio and q the exact quotient. Equal-width
    /// canonical operands give `q <= e` and `num/den > e*d_top/(d_top+1)`,
    /// hence `q >= floor(e/2)`. Admission bounds the downward corrections by
    /// `ceil(DIVISION_SMALL_QUOTIENT_MAX/2)` before either output is mutated.
    /// Estimates zero and one use the leading limbs directly. Equal leading
    /// limbs require only the trivial zero-or-one division.
    pub fn small_quotient_div_rem(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        let u_limbs = num_a.limbs();
        let v_limbs = den_b.limbs();
        let u_len = u_limbs.len();
        debug_assert!(
            u_len != 0 && u_len == v_limbs.len(),
            "the dispatcher establishes nonempty equal-width operands"
        );
        // SAFETY: both canonical slices have the same strictly positive length.
        let (numerator_top, denominator_top) = unsafe {
            (
                *u_limbs.get_unchecked(u_len.unchecked_sub(1)),
                *v_limbs.get_unchecked(u_len.unchecked_sub(1)),
            )
        };
        if numerator_top == denominator_top {
            let completed = Self::trivial::<true, true>(num_a, den_b, quotient_out, rem_out);
            debug_assert!(completed, "equal leading limbs bound the quotient by one");
            return true;
        }
        // SAFETY: a canonical nonzero denominator has a nonzero leading limb.
        let mut quotient =
            unsafe { Div::div(numerator_top, NonZeroUsize::new_unchecked(denominator_top)) };
        if quotient == 0 {
            quotient_out.clear();
            rem_out.clone_from(num_a);
            return true;
        }
        if quotient == 1 {
            quotient_out.set_limb(1);
            let underflowed = rem_out.assign_difference(num_a, den_b);
            debug_assert!(!underflowed, "unequal leading limbs prove num_a > den_b");
            return true;
        }
        if quotient > DIVISION_SMALL_QUOTIENT_MAX {
            return false;
        }
        // With e=floor(u_top/d_top), e*D-U < e*B^(n-1) < B^n.
        // The signed residue fits (-B^n,B^n); a scalar borrow encodes its sign.
        rem_out.clone_from(num_a);
        let sub_mul = ArchKernels::selected_sub_mul_limbs_unchecked();
        // SAFETY: the cloned exclusive remainder and immutable divisor cover
        // u_len initialized, aligned, disjoint limbs; CPU selection is complete.
        let (product_carry, low_borrow) = unsafe {
            sub_mul(
                rem_out.limbs_mut().as_mut_ptr(),
                v_limbs.as_ptr(),
                u_len,
                quotient,
            )
        };
        // SAFETY: the product carry is below quotient and low_borrow <= 1,
        // so their sum is at most quotient <= Limb::MAX.
        let mut borrow = unsafe { product_carry.unchecked_add(low_borrow) };
        debug_assert!(borrow <= 1, "the signed residue exceeds -B^n");
        let mut corrections = 0_usize;
        while borrow != 0 {
            debug_assert!(
                corrections < DIVISION_SMALL_QUOTIENT_MAX.div_ceil(2) && quotient >= 2,
                "the admitted leading-limb estimate bounds downward corrections"
            );
            // Adding D decrements the quotient estimate. The carry cancels
            // the negative guard; the first nonnegative residue is below D.
            let carry = Addition::add_slice_in_place(rem_out.limbs_mut(), v_limbs);
            // SAFETY: the loop has borrow=1 and carry <= 1. A negative
            // residue means quotient exceeds the exact quotient, which is
            // at least one for these admitted unequal leading limbs.
            unsafe {
                borrow = borrow.unchecked_sub(carry);
                quotient = quotient.unchecked_sub(1);
            }
            // SAFETY: the admitted estimate bounds corrections by
            // ceil(DIVISION_SMALL_QUOTIENT_MAX/2), strictly below usize::MAX.
            corrections = unsafe { corrections.unchecked_add(1) };
        }
        rem_out.normalize();
        debug_assert!(
            *rem_out < *den_b,
            "corrected remainder is below the divisor"
        );
        quotient_out.set_limb(quotient);
        true
    }

    /// Compares nonempty equal-width canonical values with `num < 2*den`.
    ///
    /// A set high divisor bit makes 2*den wider than num. Otherwise the carry
    /// into each doubled limb comes only from its immediate lower neighbor,
    /// permitting a descending comparison without materializing the product.
    pub fn less_than_double(num: &[Limb], den: &[Limb]) -> bool {
        debug_assert!(
            !den.is_empty() && num.len() == den.len(),
            "doubling comparison requires nonempty matching lengths"
        );
        // SAFETY: the sole trivial-division caller supplies nonempty equal
        // widths; every high limb is initialized and canonical.
        let (&top, lower) = unsafe { den.split_last().unwrap_unchecked() };
        // Supported limbs have 16, 32, or 64 bits, so the high-bit index exists.
        if top >> (Limb::BITS - 1) != 0 {
            return true;
        }
        for index in (1..den.len()).rev() {
            // SAFETY: 1<=index<den.len()=num.len() bounds both current limbs
            // and index-1 within the initialized lower divisor prefix.
            let (num_limb, den_limb, previous) = unsafe {
                (
                    *num.get_unchecked(index),
                    *den.get_unchecked(index),
                    *lower.get_unchecked(index.unchecked_sub(1)),
                )
            };
            let carry = previous >> (Limb::BITS - 1);
            let doubled = (den_limb << 1) | carry;
            match num_limb.cmp(&doubled) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }
        // SAFETY: both nonempty inputs contain their low limb. Its incoming
        // carry is zero; overflow was compared at index 1 or excluded by the
        // high-bit test for a one-limb divisor.
        unsafe { *num.get_unchecked(0) < (*den.get_unchecked(0) << 1) }
    }
}
