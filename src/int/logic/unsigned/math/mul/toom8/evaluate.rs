//! Power-of-two paired evaluation for Toom-8 and Toom-8.5.

use super::{AddMulKernel, ArchKernels, Limb, ProductPair, SharedEval, Toom8};

pub struct MulEvaluationBuffers<'buffer> {
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub odd_a: &'buffer mut [Limb],
    pub odd_b: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub fast_paired_add_sub: bool,
    pub add_mul_kernel: AddMulKernel,
}

pub struct SqrEvaluationBuffers<'buffer> {
    pub eval: &'buffer mut [Limb],
    pub odd: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub fast_paired_add_sub: bool,
    pub add_mul_kernel: AddMulKernel,
}

#[derive(Clone, Copy, Debug)]
pub enum EvaluationDirection {
    Direct,
    Reciprocal,
}

#[derive(Clone, Copy, Debug)]
pub struct EvaluationPoint {
    pub direction: EvaluationDirection,
    pub shift: usize,
}

impl Toom8 {
    pub fn evaluate_mul_pair(
        packed: &mut [Limb],
        negative: &mut [Limb],
        buffers: &mut MulEvaluationBuffers<'_>,
        a: &[Limb],
        b: &[Limb],
        split_len: usize,
        point: EvaluationPoint,
    ) -> bool {
        let value_len = negative.len();
        let (_, positive_and_guard) = packed.split_at_mut(split_len);
        let (positive, _) = positive_and_guard.split_at_mut(value_len);
        let pair = ProductPair { positive, negative };

        let negative_a = evaluate_even_odd(
            buffers.eval_a,
            buffers.odd_a,
            a,
            split_len,
            point,
            buffers.add_mul_kernel,
        );
        let negative_b = evaluate_even_odd(
            buffers.eval_b,
            buffers.odd_b,
            b,
            split_len,
            point,
            buffers.add_mul_kernel,
        );
        let fast_paired = buffers.fast_paired_add_sub;
        if fast_paired {
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval_a,
                buffers.odd_a,
                negative_a,
            );
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval_b,
                buffers.odd_b,
                negative_b,
            );
        } else {
            // SAFETY: each Toom-8 even/odd pair is allocated with identical
            // guarded widths by the evaluation scratch layout.
            unsafe {
                SharedEval::add_part(buffers.eval_a, buffers.odd_a);
            }
            // SAFETY: the second operand pair has the same exact-width layout.
            unsafe {
                SharedEval::add_part(buffers.eval_b, buffers.odd_b);
            }
        }
        Self::multiply_active(
            pair.positive,
            buffers.eval_a,
            buffers.eval_b,
            buffers.scratch,
            split_len,
            buffers.add_mul_kernel,
        );

        if fast_paired {
            Self::multiply_active(
                pair.negative,
                buffers.odd_a,
                buffers.odd_b,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        } else {
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval_a,
                buffers.odd_a,
                negative_a,
            );
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval_b,
                buffers.odd_b,
                negative_b,
            );
            Self::multiply_active(
                pair.negative,
                buffers.eval_a,
                buffers.eval_b,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        }
        negative_a ^ negative_b
    }

    pub fn evaluate_sqr_pair(
        packed: &mut [Limb],
        negative: &mut [Limb],
        buffers: &mut SqrEvaluationBuffers<'_>,
        operand: &[Limb],
        split_len: usize,
        point: EvaluationPoint,
    ) {
        let value_len = negative.len();
        let (_, positive_and_guard) = packed.split_at_mut(split_len);
        let (positive, _) = positive_and_guard.split_at_mut(value_len);
        let pair = ProductPair { positive, negative };

        let negative_evaluation = evaluate_even_odd(
            buffers.eval,
            buffers.odd,
            operand,
            split_len,
            point,
            buffers.add_mul_kernel,
        );
        let fast_paired = buffers.fast_paired_add_sub;
        if fast_paired {
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval,
                buffers.odd,
                negative_evaluation,
            );
        } else {
            // SAFETY: `eval` and `odd` are the equal-width guarded buffers of
            // one Toom-8 conjugate evaluation pair.
            unsafe {
                SharedEval::add_part(buffers.eval, buffers.odd);
            }
        }
        Self::square_active(
            pair.positive,
            buffers.eval,
            buffers.scratch,
            split_len,
            buffers.add_mul_kernel,
        );
        if fast_paired {
            Self::square_active(
                pair.negative,
                buffers.odd,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        } else {
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval,
                buffers.odd,
                negative_evaluation,
            );
            Self::square_active(
                pair.negative,
                buffers.eval,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        }
    }
}

fn evaluate_even_odd(
    even: &mut [Limb],
    odd: &mut [Limb],
    operand: &[Limb],
    split_len: usize,
    point: EvaluationPoint,
    kernel: AddMulKernel,
) -> bool {
    assert_eq!(even.len(), odd.len(), "evaluation widths must match");
    assert!(split_len > 0, "Toom-8 parts must be nonempty");
    assert!(
        split_len < even.len(),
        "each Toom-8 part must fit below the evaluation guard"
    );
    assert!(point.shift <= 3, "Toom-8 point shift exceeds eight");
    assert!(
        operand.len().div_ceil(split_len) <= 9,
        "Toom-8.5 operand has more than nine parts"
    );
    match point.direction {
        EvaluationDirection::Direct => {
            evaluate_direct(even, odd, operand, split_len, point.shift, kernel);
        }
        EvaluationDirection::Reciprocal => {
            evaluate_reciprocal(even, odd, operand, split_len, point.shift, kernel);
        }
    }
    even.iter().rev().cmp(odd.iter().rev()).is_lt()
}

fn evaluate_direct(
    even: &mut [Limb],
    odd: &mut [Limb],
    operand: &[Limb],
    split_len: usize,
    point_shift: usize,
    kernel: AddMulKernel,
) {
    if Limb::BITS >= 32 {
        evaluate_direct_word_weights(even, odd, operand, split_len, point_shift, kernel);
        return;
    }
    even.fill(0);
    odd.fill(0);
    let squared_shift = point_shift.wrapping_mul(2);
    let mut even_started = false;
    let mut odd_started = false;
    for (part_index, part) in operand.chunks(split_len).enumerate().rev() {
        let (target, started) = if part_index.is_multiple_of(2) {
            (&mut *even, &mut even_started)
        } else {
            (&mut *odd, &mut odd_started)
        };
        if *started {
            shift_left(target, squared_shift);
        }
        // SAFETY: every chunk has at most `split_len` limbs, while each even or
        // odd accumulator retains the tier's evaluation guard above that width.
        unsafe {
            SharedEval::add_part(target, part);
        }
        *started = true;
    }
    shift_left(odd, point_shift);
}

fn evaluate_reciprocal(
    even: &mut [Limb],
    odd: &mut [Limb],
    operand: &[Limb],
    split_len: usize,
    denominator_shift: usize,
    kernel: AddMulKernel,
) {
    if Limb::BITS >= 32 {
        evaluate_reciprocal_word_weights(even, odd, operand, split_len, denominator_shift, kernel);
        return;
    }
    even.fill(0);
    odd.fill(0);
    let part_count = operand.len().div_ceil(split_len);
    let degree = part_count.wrapping_sub(1);
    let squared_shift = denominator_shift.wrapping_mul(2);
    let mut even_started = false;
    let mut odd_started = false;
    for (part_index, part) in operand.chunks(split_len).enumerate() {
        let (target, started) = if part_index.is_multiple_of(2) {
            (&mut *even, &mut even_started)
        } else {
            (&mut *odd, &mut odd_started)
        };
        if *started {
            shift_left(target, squared_shift);
        }
        // SAFETY: every chunk has at most `split_len` limbs, while each even or
        // odd accumulator retains the tier's evaluation guard above that width.
        unsafe {
            SharedEval::add_part(target, part);
        }
        *started = true;
    }
    if degree.is_multiple_of(2) {
        shift_left(odd, denominator_shift);
    } else {
        shift_left(even, denominator_shift);
    }
}

fn evaluate_direct_word_weights(
    even: &mut [Limb],
    odd: &mut [Limb],
    operand: &[Limb],
    split_len: usize,
    point_shift: usize,
    kernel: AddMulKernel,
) {
    if point_shift == 0 {
        evaluate_direct_one(even, odd, operand, split_len);
        return;
    }

    let mut parts = operand.chunks(split_len);
    let Some(constant) = parts.next() else {
        even.fill(0);
        odd.fill(0);
        return;
    };
    SharedEval::copy_part(even, constant);
    let Some(linear) = parts.next() else {
        odd.fill(0);
        return;
    };
    SharedEval::copy_part(odd, linear);

    let squared_shift = point_shift.wrapping_mul(2);
    // SAFETY: squared_shift is at most 2 * point_shift ≤ 2 * 3 = 6, always < 2^32.
    let squared_shift_u32 = unsafe { u32::try_from(squared_shift).unwrap_unchecked() };
    let squared_scalar = 1_usize.wrapping_shl(squared_shift_u32);
    let mut scalar = squared_scalar;
    while let Some(even_part) = parts.next() {
        SharedEval::add_mul_word_with_kernel_in_place(even, even_part, scalar, kernel);
        let Some(odd_part) = parts.next() else {
            break;
        };
        SharedEval::add_mul_word_with_kernel_in_place(odd, odd_part, scalar, kernel);
        scalar = scalar.wrapping_mul(squared_scalar);
    }
    shift_left(odd, point_shift);
}

fn evaluate_direct_one(even: &mut [Limb], odd: &mut [Limb], operand: &[Limb], split_len: usize) {
    let mut parts = operand.chunks(split_len);
    let Some(constant) = parts.next() else {
        even.fill(0);
        odd.fill(0);
        return;
    };
    let Some(linear) = parts.next() else {
        SharedEval::copy_part(even, constant);
        odd.fill(0);
        return;
    };
    let Some(quadratic) = parts.next() else {
        SharedEval::copy_part(even, constant);
        SharedEval::copy_part(odd, linear);
        return;
    };
    let Some(cubic) = parts.next() else {
        sum_initial_parts(even, constant, quadratic);
        SharedEval::copy_part(odd, linear);
        return;
    };

    sum_initial_parts(even, constant, quadratic);
    sum_initial_parts(odd, linear, cubic);

    // At x=1 every even or odd part has unit weight. Plain carry additions
    // avoid routing six unit scalars through the add-multiply backend.
    while let Some(even_part) = parts.next() {
        // SAFETY: each chunk has at most `split_len` limbs and `even` retains
        // its evaluation guard above that part width.
        unsafe {
            SharedEval::add_part(even, even_part);
        }
        let Some(odd_part) = parts.next() else {
            break;
        };
        // SAFETY: `odd` has the same guarded width and the chunk has at most
        // `split_len` limbs.
        unsafe {
            SharedEval::add_part(odd, odd_part);
        }
    }
}

fn sum_initial_parts(dst: &mut [Limb], left: &[Limb], right: &[Limb]) {
    if !left.is_empty() && left.len() == right.len() && dst.len() > left.len() {
        let (body, guard) = dst.split_at_mut(left.len());
        guard.fill(0);
        // SAFETY: the equality above proves both sources and the destination
        // body cover the same nonzero width, and all three slices are disjoint.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                body.as_mut_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                body.len(),
            )
        };
        if let Some(first_guard) = guard.first_mut() {
            *first_guard = carry;
        }
        return;
    }
    SharedEval::copy_part(dst, left);
    // SAFETY: every caller supplies chunks formed after `evaluate_even_odd`
    // validated that the chunk width is strictly below `dst.len()`. The fast
    // branch above covers the exact equal-width case but is not needed here.
    unsafe {
        SharedEval::add_part(dst, right);
    }
}

fn evaluate_reciprocal_word_weights(
    even: &mut [Limb],
    odd: &mut [Limb],
    operand: &[Limb],
    split_len: usize,
    denominator_shift: usize,
    kernel: AddMulKernel,
) {
    let part_count = operand.len().div_ceil(split_len);
    let degree = part_count.wrapping_sub(1);
    let mut parts = operand.chunks(split_len).rev();
    let (leading, trailing) = if degree.is_multiple_of(2) {
        (&mut *even, &mut *odd)
    } else {
        (&mut *odd, &mut *even)
    };
    let Some(highest) = parts.next() else {
        leading.fill(0);
        trailing.fill(0);
        return;
    };
    SharedEval::copy_part(leading, highest);
    let Some(next) = parts.next() else {
        trailing.fill(0);
        return;
    };
    SharedEval::copy_part(trailing, next);

    let squared_shift = denominator_shift.wrapping_mul(2);
    // SAFETY: squared_shift ≤ 6, always < 2^32.
    let squared_shift_u32 = unsafe { u32::try_from(squared_shift).unwrap_unchecked() };
    let squared_scalar = 1_usize.wrapping_shl(squared_shift_u32);
    let mut scalar = squared_scalar;
    while let Some(leading_part) = parts.next() {
        SharedEval::add_mul_word_with_kernel_in_place(leading, leading_part, scalar, kernel);
        let Some(trailing_part) = parts.next() else {
            break;
        };
        SharedEval::add_mul_word_with_kernel_in_place(trailing, trailing_part, scalar, kernel);
        scalar = scalar.wrapping_mul(squared_scalar);
    }
    shift_left(trailing, denominator_shift);
}

fn shift_left(value: &mut [Limb], shift: usize) {
    if shift == 0 {
        return;
    }
    // SAFETY: entry validation bounds a point shift by three, so this helper's
    // direct or squared shift is at most six and therefore fits in `u32`.
    let shift_u32 = unsafe { u32::try_from(shift).unwrap_unchecked() };
    // SAFETY: evaluation owns `value.len()` writable limbs and uses shifts at
    // most six, below every supported limb width. The 2^25 evaluation bound
    // proves the retained guard absorbs the top carry.
    let carry =
        unsafe { ArchKernels::lshift_unchecked(value.as_mut_ptr(), value.len(), shift_u32) };
    debug_assert_eq!(carry, 0, "Toom-8 evaluation exceeded its guard");
}
