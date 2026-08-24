//! Four-way Toom-Cook operand evaluation.

use core::cmp::Ordering;

use super::{AddMulKernel, Addition, ArchKernels, DoubleLimb, LIMB_BITS, Limb, SharedEval, Toom4};

impl Toom4 {
    /// Evaluate a four-part operand at `x = 1` or `x = 2`.
    #[allow(
        unsafe_code,
        reason = "Each Toom-4 operand part is bounded by the guarded evaluation buffer"
    )]
    pub fn evaluate_positive(
        dst: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
        point_shift: u32,
    ) {
        debug_assert!(point_shift <= 1, "Toom-4 evaluates only at 1 or 2");
        debug_assert!(
            [part0, part1, part2, part3]
                .into_iter()
                .all(|part| part.len() < dst.len()),
            "each Toom-4 part must fit below the evaluation guard"
        );
        let evaluated = if point_shift == 0 {
            evaluate_positive_balanced::<false>(dst, part0, part1, part2, part3)
        } else if point_shift == 1 {
            evaluate_positive_balanced::<true>(dst, part0, part1, part2, part3)
        } else {
            false
        };
        if evaluated {
            return;
        }
        // Horner evaluation at k=2^point_shift uses only shifts and additions:
        // a0+k(a1+k(a2+k*a3)). At k=2 the result is below 15*B^m.
        SharedEval::copy_part(dst, part3);
        shift_left(dst, point_shift);
        // SAFETY: every part has at most `m` limbs while `dst` retains the
        // `(m + 1)`-limb evaluation guard.
        unsafe {
            SharedEval::add_part(dst, part2);
        }
        shift_left(dst, point_shift);
        // SAFETY: the same guarded evaluation-width invariant bounds `part1`.
        unsafe {
            SharedEval::add_part(dst, part1);
        }
        shift_left(dst, point_shift);
        // SAFETY: the same guarded evaluation-width invariant bounds `part0`.
        unsafe {
            SharedEval::add_part(dst, part0);
        }
    }

    /// Evaluate the denominator-scaled value `8*A(1/2)`.
    #[allow(
        unsafe_code,
        reason = "Each Toom-4 operand part is bounded by the guarded evaluation buffer"
    )]
    pub fn evaluate_half_scaled(
        dst: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
    ) {
        debug_assert!(
            [part0, part1, part2, part3]
                .into_iter()
                .all(|part| part.len() < dst.len()),
            "each Toom-4 part must fit below the evaluation guard"
        );
        if evaluate_half_balanced(dst, part0, part1, part2, part3) {
            return;
        }
        SharedEval::copy_part(dst, part0);
        shift_left(dst, 1);
        // SAFETY: each operand part has at most the `m` data limbs below
        // `dst`'s evaluation guard.
        unsafe {
            SharedEval::add_part(dst, part1);
        }
        shift_left(dst, 1);
        // SAFETY: the same guarded evaluation-width invariant bounds `part2`.
        unsafe {
            SharedEval::add_part(dst, part2);
        }
        shift_left(dst, 1);
        // SAFETY: the same guarded evaluation-width invariant bounds `part3`.
        unsafe {
            SharedEval::add_part(dst, part3);
        }
    }

    /// Evaluate `8*A(1/2)` through the selected scalar add-multiply backend.
    pub fn evaluate_half_scaled_with_kernel(
        dst: &mut [Limb],
        blocks: [&[Limb]; 4],
        kernel: AddMulKernel,
    ) -> bool {
        let [part0, part1, part2, part3] = blocks;
        let Some((guard, body)) = dst.split_last_mut() else {
            return false;
        };
        if part0.len() != body.len()
            || part1.len() != body.len()
            || part2.len() != body.len()
            || part3.len() > body.len()
        {
            return false;
        }
        let (prefix, suffix) = body.split_at_mut(part3.len());
        prefix.copy_from_slice(part3);
        suffix.fill(0);
        *guard = 0;
        SharedEval::add_mul_word_with_kernel_in_place(dst, part2, 2, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(dst, part1, 4, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(dst, part0, 8, kernel);
        true
    }

    /// Evaluate at `x = -1` or `x = -2`, returning an absolute magnitude.
    ///
    /// The return value records whether the mathematical evaluation was negative.
    #[allow(
        unsafe_code,
        reason = "The even and odd Toom-4 accumulators each retain a guard above every operand part"
    )]
    pub fn evaluate_negative_magnitude(
        dst: &mut [Limb],
        odd: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
        point_shift: u32,
    ) -> bool {
        debug_assert!(point_shift <= 1, "Toom-4 evaluates only at -1 or -2");
        debug_assert_eq!(dst.len(), odd.len(), "evaluation widths must match");
        debug_assert!(
            [part0, part1, part2, part3]
                .into_iter()
                .all(|part| part.len() < dst.len()),
            "each Toom-4 part must fit below the evaluation guard"
        );
        let evaluated = if point_shift == 0 {
            evaluate_negative_one_balanced(dst, odd, part0, part1, part2, part3)
        } else if point_shift == 1 {
            evaluate_negative_two_balanced(dst, odd, part0, part1, part2, part3)
        } else {
            false
        };
        if !evaluated {
            let squared_shift = point_shift.wrapping_mul(2);
            SharedEval::copy_part(dst, part2);
            shift_left(dst, squared_shift);
            // SAFETY: `dst` has `m + 1` limbs and `part0` has at most `m`.
            unsafe {
                SharedEval::add_part(dst, part0);
            }
            SharedEval::copy_part(odd, part3);
            shift_left(odd, squared_shift);
            // SAFETY: `odd` has `m + 1` limbs and `part1` has at most `m`.
            unsafe {
                SharedEval::add_part(odd, part1);
            }
            shift_left(odd, point_shift);
        }

        match dst.iter().rev().cmp(odd.iter().rev()) {
            Ordering::Less => {
                // dst = odd - dst in one pass; the comparison proves the result
                // is nonnegative, so the discarded borrow is provably zero.
                SharedEval::reverse_difference_in_place(dst, odd);
                true
            }
            Ordering::Equal => {
                dst.fill(0);
                false
            }
            Ordering::Greater => {
                let borrow = Addition::sub_slice_in_place(dst, odd);
                debug_assert_eq!(borrow, 0, "even magnitude must exceed odd magnitude");
                false
            }
        }
    }
}

fn evaluate_negative_two_balanced(
    even: &mut [Limb],
    odd: &mut [Limb],
    part0: &[Limb],
    part1: &[Limb],
    part2: &[Limb],
    part3: &[Limb],
) -> bool {
    let Some((even_guard, even_body)) = even.split_last_mut() else {
        return false;
    };
    let Some((odd_guard, odd_body)) = odd.split_last_mut() else {
        return false;
    };
    if even_body.is_empty()
        || part0.len() != even_body.len()
        || part1.len() != even_body.len()
        || part2.len() != even_body.len()
        || part3.len() > even_body.len()
        || odd_body.len() != even_body.len()
    {
        return false;
    }

    let prefix_len = part3.len();
    let (even_prefix, even_suffix) = even_body.split_at_mut(prefix_len);
    let (odd_prefix, odd_suffix) = odd_body.split_at_mut(prefix_len);
    let (part0_prefix, part0_suffix) = part0.split_at(prefix_len);
    let (part1_prefix, part1_suffix) = part1.split_at(prefix_len);
    let (part2_prefix, part2_suffix) = part2.split_at(prefix_len);
    let mut even_carry = 0;
    let mut odd_carry = 0;
    for (((((even_limb, odd_limb), part0_limb), part1_limb), part2_limb), part3_limb) in even_prefix
        .iter_mut()
        .zip(odd_prefix)
        .zip(part0_prefix)
        .zip(part1_prefix)
        .zip(part2_prefix)
        .zip(part3)
    {
        *even_limb = evaluate_at_two_limb(
            *part0_limb,
            Limb::MIN,
            *part2_limb,
            Limb::MIN,
            &mut even_carry,
        );
        *odd_limb = evaluate_at_two_limb(
            Limb::MIN,
            *part1_limb,
            Limb::MIN,
            *part3_limb,
            &mut odd_carry,
        );
    }
    for ((((even_limb, odd_limb), part0_limb), part1_limb), part2_limb) in even_suffix
        .iter_mut()
        .zip(odd_suffix)
        .zip(part0_suffix)
        .zip(part1_suffix)
        .zip(part2_suffix)
    {
        *even_limb = evaluate_at_two_limb(
            *part0_limb,
            Limb::MIN,
            *part2_limb,
            Limb::MIN,
            &mut even_carry,
        );
        *odd_limb =
            evaluate_at_two_limb(Limb::MIN, *part1_limb, Limb::MIN, Limb::MIN, &mut odd_carry);
    }
    // E=a0+4*a2 < 5*B^m and O=2*a1+8*a3 < 10*B^m, so each
    // complete carry fits in the existing evaluation guard limb.
    *even_guard = even_carry;
    *odd_guard = odd_carry;
    true
}

fn evaluate_negative_one_balanced(
    even: &mut [Limb],
    odd: &mut [Limb],
    part0: &[Limb],
    part1: &[Limb],
    part2: &[Limb],
    part3: &[Limb],
) -> bool {
    let Some((even_guard, even_body)) = even.split_last_mut() else {
        return false;
    };
    let Some((odd_guard, odd_body)) = odd.split_last_mut() else {
        return false;
    };
    if even_body.is_empty()
        || part0.len() != even_body.len()
        || part1.len() != even_body.len()
        || part2.len() != even_body.len()
        || part3.len() != even_body.len()
        || odd_body.len() != even_body.len()
    {
        return false;
    }

    // SAFETY: all source and destination bodies have the same nonzero split
    // length, as proven above, and the two output buffers are disjoint.
    *even_guard = unsafe {
        ArchKernels::add_limbs_3_unchecked(
            even_body.as_mut_ptr(),
            part0.as_ptr(),
            part2.as_ptr(),
            even_body.len(),
        )
    };
    // SAFETY: the same equal-length proof covers the odd-part addition.
    *odd_guard = unsafe {
        ArchKernels::add_limbs_3_unchecked(
            odd_body.as_mut_ptr(),
            part1.as_ptr(),
            part3.as_ptr(),
            odd_body.len(),
        )
    };
    true
}

impl Toom4 {
    /// Split a limb slice into four radix blocks of at most `split_len` limbs.
    pub const fn split_four(
        values: &[Limb],
        split_len: usize,
    ) -> (&[Limb], &[Limb], &[Limb], &[Limb]) {
        if values.len() > split_len.wrapping_mul(3) {
            let (part0, after_part0) = values.split_at(split_len);
            let (part1, after_part1) = after_part0.split_at(split_len);
            let (part2, part3) = after_part1.split_at(split_len);
            (part0, part1, part2, part3)
        } else if values.len() > split_len.wrapping_mul(2) {
            let (part0, rest) = values.split_at(split_len);
            let (part1, part2) = rest.split_at(split_len);
            (part0, part1, part2, &[])
        } else if values.len() > split_len {
            let (part0, part1) = values.split_at(split_len);
            (part0, part1, &[], &[])
        } else {
            (values, &[], &[], &[])
        }
    }
}

fn evaluate_positive_balanced<const AT_TWO: bool>(
    dst: &mut [Limb],
    part0: &[Limb],
    part1: &[Limb],
    part2: &[Limb],
    part3: &[Limb],
) -> bool {
    let Some((guard, body)) = dst.split_last_mut() else {
        return false;
    };
    if part0.len() != body.len()
        || part1.len() != body.len()
        || part2.len() != body.len()
        || part3.len() > body.len()
    {
        return false;
    }

    let prefix_len = part3.len();
    let (body_prefix, body_suffix) = body.split_at_mut(prefix_len);
    let (part0_prefix, part0_suffix) = part0.split_at(prefix_len);
    let (part1_prefix, part1_suffix) = part1.split_at(prefix_len);
    let (part2_prefix, part2_suffix) = part2.split_at(prefix_len);
    let mut carry = 0;
    for ((((dst_limb, part0_limb), part1_limb), part2_limb), part3_limb) in body_prefix
        .iter_mut()
        .zip(part0_prefix)
        .zip(part1_prefix)
        .zip(part2_prefix)
        .zip(part3)
    {
        *dst_limb = if AT_TWO {
            evaluate_at_two_limb(
                *part0_limb,
                *part1_limb,
                *part2_limb,
                *part3_limb,
                &mut carry,
            )
        } else {
            evaluate_at_one_limb(
                *part0_limb,
                *part1_limb,
                *part2_limb,
                *part3_limb,
                &mut carry,
            )
        };
    }
    for (((dst_limb, part0_limb), part1_limb), part2_limb) in body_suffix
        .iter_mut()
        .zip(part0_suffix)
        .zip(part1_suffix)
        .zip(part2_suffix)
    {
        *dst_limb = if AT_TWO {
            evaluate_at_two_limb(*part0_limb, *part1_limb, *part2_limb, 0, &mut carry)
        } else {
            evaluate_at_one_limb(*part0_limb, *part1_limb, *part2_limb, 0, &mut carry)
        };
    }
    // The polynomial bounds are 4*B^m at x=1 and 15*B^m at x=2, so one
    // guard limb holds the complete carry (at most three or fourteen).
    *guard = carry;
    true
}

fn evaluate_half_balanced(
    dst: &mut [Limb],
    part0: &[Limb],
    part1: &[Limb],
    part2: &[Limb],
    part3: &[Limb],
) -> bool {
    let Some((guard, body)) = dst.split_last_mut() else {
        return false;
    };
    if part0.len() != body.len()
        || part1.len() != body.len()
        || part2.len() != body.len()
        || part3.len() > body.len()
    {
        return false;
    }

    let prefix_len = part3.len();
    let (body_prefix, body_suffix) = body.split_at_mut(prefix_len);
    let (part0_prefix, part0_suffix) = part0.split_at(prefix_len);
    let (part1_prefix, part1_suffix) = part1.split_at(prefix_len);
    let (part2_prefix, part2_suffix) = part2.split_at(prefix_len);
    let mut carry = 0;
    for ((((dst_limb, part0_limb), part1_limb), part2_limb), part3_limb) in body_prefix
        .iter_mut()
        .zip(part0_prefix)
        .zip(part1_prefix)
        .zip(part2_prefix)
        .zip(part3)
    {
        *dst_limb = evaluate_at_two_limb(
            *part3_limb,
            *part2_limb,
            *part1_limb,
            *part0_limb,
            &mut carry,
        );
    }
    for (((dst_limb, part0_limb), part1_limb), part2_limb) in body_suffix
        .iter_mut()
        .zip(part0_suffix)
        .zip(part1_suffix)
        .zip(part2_suffix)
    {
        *dst_limb = evaluate_at_two_limb(0, *part2_limb, *part1_limb, *part0_limb, &mut carry);
    }
    // 8*a0+4*a1+2*a2+a3 has the same <15*B^m bound as A(2).
    *guard = carry;
    true
}

fn evaluate_at_one_limb(
    part0: Limb,
    part1: Limb,
    part2: Limb,
    part3: Limb,
    carry: &mut Limb,
) -> Limb {
    let (with_part1, overflow_part1) = part0.overflowing_add(part1);
    let (with_part2, overflow_part2) = with_part1.overflowing_add(part2);
    let (with_part3, overflow_part3) = with_part2.overflowing_add(part3);
    let (result, overflow_carry) = with_part3.overflowing_add(*carry);
    *carry = Limb::from(overflow_part1)
        .wrapping_add(Limb::from(overflow_part2))
        .wrapping_add(Limb::from(overflow_part3))
        .wrapping_add(Limb::from(overflow_carry));
    result
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The cast extracts the exact low and high limbs of the wide accumulator"
)]
const fn evaluate_at_two_limb(
    part0: Limb,
    part1: Limb,
    part2: Limb,
    part3: Limb,
    carry: &mut Limb,
) -> Limb {
    let sum = (part0 as DoubleLimb)
        .wrapping_add((part1 as DoubleLimb) << 1)
        .wrapping_add((part2 as DoubleLimb) << 2)
        .wrapping_add((part3 as DoubleLimb) << 3)
        .wrapping_add(*carry as DoubleLimb);
    *carry = (sum >> LIMB_BITS) as Limb;
    sum as Limb
}

fn shift_left(value: &mut [Limb], shift: u32) {
    if shift == 0 || value.is_empty() {
        return;
    }
    // SAFETY: `value` is valid for its complete length and callers pass only
    // shifts 1 or 2, below LIMB_BITS on every target. The 15*B^m bound proves
    // the evaluation guard absorbs every high bit.
    let carry = unsafe { ArchKernels::lshift_unchecked(value.as_mut_ptr(), value.len(), shift) };
    debug_assert_eq!(carry, 0, "Toom-4 evaluation exceeded its guard limb");
}
