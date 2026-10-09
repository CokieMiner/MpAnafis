//! Three-way Toom-Cook operand evaluation.

#![expect(
    unsafe_code,
    reason = "Three-part split bounds, initialized endpoints, and retained evaluation guards bound the paired arithmetic spans"
)]

use core::cmp::Ordering;

use super::{AddMulKernel, Addition, ArchKernels, Limb, SharedEval, Toom3};

/// Middle evaluated products and the sign of `W(-1)`.
pub struct MiddleValues<'buffer> {
    pub one: &'buffer mut [Limb],
    pub neg_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub neg_one_negative: bool,
}

impl Toom3 {
    /// Transform an existing `A(1)` evaluation into `A(2)`.
    pub fn evaluate_two_from_one(
        dst: &mut [Limb],
        part1: &[Limb],
        part2: &[Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        debug_assert!(
            part1.len() <= dst.len(),
            "linear part exceeds the evaluation destination"
        );
        // A(2)-A(1) = a1+3*a2. Both additions consume the existing guard limb;
        // the final value remains below 7*B^m, so no carry can escape `dst`.
        // SAFETY: the guarded Toom-3 layout proves the source fits at offset zero.
        unsafe {
            SharedEval::add_part(dst, part1);
        }
        SharedEval::add_mul_word_with_kernel_in_place(dst, part2, 3, add_mul_kernel);
    }

    /// Evaluate one three-part operand at `x = 1` and `x = -1` together.
    ///
    /// The negative evaluation is returned as an absolute magnitude; the return
    /// value records whether its mathematical value was negative.
    pub fn evaluate_one_and_negative_one(
        positive: &mut [Limb],
        negative_magnitude: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
    ) -> bool {
        debug_assert_eq!(
            positive.len(),
            negative_magnitude.len(),
            "paired evaluations require equal widths"
        );
        Self::evaluate_even_at_one(negative_magnitude, part0, part2);
        let ordering = SharedEval::compare_with_zero_extension(negative_magnitude, part1);
        match ordering {
            Ordering::Less => {
                Self::form_sum_and_difference::<true>(positive, negative_magnitude, part1);
                true
            }
            Ordering::Equal | Ordering::Greater => {
                Self::form_sum_and_difference::<false>(positive, negative_magnitude, part1);
                false
            }
        }
    }

    fn form_sum_and_difference<const ODD_MINUS_EVEN: bool>(
        sum: &mut [Limb],
        even: &mut [Limb],
        odd: &[Limb],
    ) {
        debug_assert_eq!(sum.len(), even.len(), "paired outputs require equal widths");
        debug_assert!(odd.len() <= even.len(), "odd part exceeds evaluation width");
        // SAFETY: split_three gives odd.len()<=m; both disjoint evaluation
        // buffers span m+1 initialized limbs and retain that width throughout.
        let ((sum_shared, sum_extension), (even_shared, even_extension)) = unsafe {
            (
                sum.split_at_mut_unchecked(odd.len()),
                even.split_at_mut_unchecked(odd.len()),
            )
        };
        let mut sum_carry = 0;
        let mut difference_borrow = false;
        for ((sum_limb, even_limb), odd_limb) in sum_shared.iter_mut().zip(even_shared).zip(odd) {
            let even_value = *even_limb;
            let (partial_sum, overflow_a) = even_value.overflowing_add(*odd_limb);
            let (complete_sum, overflow_b) = partial_sum.overflowing_add(sum_carry);
            *sum_limb = complete_sum;
            sum_carry = Limb::from(overflow_a | overflow_b);

            let (difference, borrow_a) = if ODD_MINUS_EVEN {
                odd_limb.overflowing_sub(even_value)
            } else {
                even_value.overflowing_sub(*odd_limb)
            };
            let (complete_difference, borrow_b) =
                difference.overflowing_sub(Limb::from(difference_borrow));
            *even_limb = complete_difference;
            difference_borrow = borrow_a | borrow_b;
        }
        for (sum_limb, even_limb) in sum_extension.iter_mut().zip(even_extension) {
            let even_value = *even_limb;
            let (complete_sum, overflow) = even_value.overflowing_add(sum_carry);
            *sum_limb = complete_sum;
            sum_carry = Limb::from(overflow);

            let (difference, borrow_a) = if ODD_MINUS_EVEN {
                Limb::MIN.overflowing_sub(even_value)
            } else {
                (even_value, false)
            };
            let (complete_difference, borrow_b) =
                difference.overflowing_sub(Limb::from(difference_borrow));
            *even_limb = complete_difference;
            difference_borrow = borrow_a | borrow_b;
        }
        debug_assert_eq!(sum_carry, 0, "positive evaluation exceeded its guard limb");
        debug_assert!(!difference_borrow, "absolute difference underflowed");
    }

    fn evaluate_even_at_one(dst: &mut [Limb], part0: &[Limb], part2: &[Limb]) {
        debug_assert!(!dst.is_empty(), "Toom-3 evaluations retain a guard limb");
        // SAFETY: split_scratch supplies split_len + 1 limbs to every evaluation,
        // including zero-valued polynomial parts, so the last limb exists.
        let (guard, body) = unsafe { dst.split_last_mut().unwrap_unchecked() };
        debug_assert!(
            part2.len() <= part0.len(),
            "quadratic part exceeds constant part"
        );
        debug_assert!(
            part0.len() <= body.len(),
            "constant part exceeds evaluation body"
        );

        // SAFETY: split_three yields |part2|<=|part0|<=m=body.len(), also for
        // short zero-extended operands. The body is disjoint from the sources.
        let ((part0_shared, part0_extension), (body_shared, body_after_shared)) = unsafe {
            (
                part0.split_at_unchecked(part2.len()),
                body.split_at_mut_unchecked(part2.len()),
            )
        };
        let mut carry = if part2.is_empty() {
            0
        } else {
            // SAFETY: all three shared slices have exactly `part2.len()` limbs and
            // occupy disjoint buffers. The kernel returns the carry into part0's
            // zero-extended high portion.
            unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    body_shared.as_mut_ptr(),
                    part0_shared.as_ptr(),
                    part2.as_ptr(),
                    part2.len(),
                )
            }
        };
        // SAFETY: part0_extension.len()=|part0|-|part2|<=m-|part2|.
        let (body_extension, body_padding) =
            unsafe { body_after_shared.split_at_mut_unchecked(part0_extension.len()) };
        for (dst_limb, part0_limb) in body_extension.iter_mut().zip(part0_extension) {
            let (sum, overflow) = part0_limb.overflowing_add(carry);
            *dst_limb = sum;
            carry = Limb::from(overflow);
        }
        body_padding.fill(0);
        // a0+a2 < 2*B^m, hence the complete carry is zero or one.
        *guard = carry;
    }

    /// Interpolate in place using endpoint products already stored in `dst`.
    pub fn interpolate_endpoints(
        dst: &[Limb],
        low_product_len: usize,
        high_offset: usize,
        high_product_len: usize,
        middle: MiddleValues<'_>,
    ) {
        let MiddleValues {
            one,
            neg_one,
            two,
            neg_one_negative,
        } = middle;
        debug_assert!(
            low_product_len <= dst.len(),
            "constant product exceeds destination"
        );
        // SAFETY: the driver writes the initialized low endpoint within dst.
        let zero = unsafe { dst.split_at_unchecked(low_product_len).0 };
        let infinity = if high_product_len == 0 {
            &[][..]
        } else {
            debug_assert!(low_product_len <= high_offset, "endpoint products overlap");
            debug_assert!(
                high_offset.saturating_add(high_product_len) <= dst.len(),
                "infinity product exceeds destination"
            );
            // SAFETY: nonempty high parts place the initialized endpoint at 4m,
            // and the driver proves 4m+high_product_len<=dst.len().
            unsafe { dst.get_unchecked(high_offset..high_offset.unchecked_add(high_product_len)) }
        };

        // For W(x)=c0+...+c4*x^4, (W(2)-W(-1))/3=c1+c2+3c3+5c4
        // and (W(1)-W(-1))/2=c1+c3. The sign flag consumes the stored
        // magnitude directly; no two's-complement conversion is required.
        if neg_one_negative {
            let _ = Addition::add_slice_in_place(two, neg_one);
            SharedEval::exact_half_sum_in_place(neg_one, one);
        } else {
            SharedEval::sub_full_slices_in_place(two, neg_one);
            SharedEval::exact_half_reverse_difference_in_place(neg_one, one);
        }
        SharedEval::exact_div_radix_minus_one_in_place::<3>(two);
        SharedEval::sub_full_slices_in_place(one, zero);
        // (two-one)/2=c3+2c4; remove endpoints to recover c1,c2,c3.
        SharedEval::sub_full_slices_in_place(two, one);
        SharedEval::exact_div2_in_place(two);
        SharedEval::sub_two_full_slices_in_place(one, neg_one, infinity);
        SharedEval::sub_mul_word_in_place(two, infinity, 2);
        SharedEval::sub_full_slices_in_place(neg_one, two);
    }

    /// Add the three interpolated middle coefficients to their radix positions.
    pub fn reconstruct_middle(
        dst: &mut [Limb],
        split_len: usize,
        first: &[Limb],
        second: &[Limb],
        third: &[Limb],
    ) {
        // c0 and c4 already occupy their final disjoint endpoint ranges in dst.
        // Adding c1, c2, and c3 directly completes the polynomial reconstruction.
        SharedEval::add_coefficient_in_place(dst, first, split_len);
        // SAFETY: m=ceil(maximum actual operand length/3); 3m is at most
        // that slice length plus two, which fits usize from its byte bound.
        let (second_offset, third_offset) =
            unsafe { (split_len.unchecked_mul(2), split_len.unchecked_mul(3)) };
        SharedEval::add_coefficient_in_place(dst, second, second_offset);
        SharedEval::add_coefficient_in_place(dst, third, third_offset);
    }
}
