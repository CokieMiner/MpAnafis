//! The six- and seven-part operand view, and its evaluation at a single point.
//!
//! A Toom-6 operand is a degree-five polynomial in `B^split_len` (degree six for
//! the 6.5 split). Every point this tier uses is a power of two, so evaluation is
//! a scaled accumulation and never a general multiplication.
//!
//! Each point is evaluated as two accumulators rather than one: the even-degree
//! terms and the odd-degree terms. `A(k) = E + O` and `A(-k) = E - O`, so one
//! evaluation pass serves both of a conjugate pair.

use core::cmp::min;

use super::{AddMulKernel, ArchKernels, Limb, SharedEval, Toom6};

/// A six- or seven-way split of an operand, one polynomial coefficient per field.
///
/// `sextic` is empty for a plain six-way split and populated only for the
/// degree-six operand of the 6.5 split.
#[derive(Clone, Copy)]
pub struct Parts<'value> {
    pub constant: &'value [Limb],
    pub linear: &'value [Limb],
    pub quadratic: &'value [Limb],
    pub cubic: &'value [Limb],
    pub quartic: &'value [Limb],
    pub quintic: &'value [Limb],
    pub sextic: &'value [Limb],
}

/// Whether a point is evaluated as `A(k)` or as the scaled `d^5 * A(1/d)`.
#[derive(Clone, Copy)]
pub enum EvaluationDirection {
    Direct,
    Reciprocal,
}

impl Toom6 {
    /// Splits `values` into six parts of `split_len` limbs, the last possibly short.
    pub fn split_six(values: &[Limb], split_len: usize) -> Parts<'_> {
        let constant_len = min(values.len(), split_len);
        let (constant, after_constant) = values.split_at(constant_len);
        let linear_len = min(after_constant.len(), split_len);
        let (linear, after_linear) = after_constant.split_at(linear_len);
        let quadratic_len = min(after_linear.len(), split_len);
        let (quadratic, after_quadratic) = after_linear.split_at(quadratic_len);
        let cubic_len = min(after_quadratic.len(), split_len);
        let (cubic, after_cubic) = after_quadratic.split_at(cubic_len);
        let quartic_len = min(after_cubic.len(), split_len);
        let (quartic, quintic) = after_cubic.split_at(quartic_len);
        debug_assert!(
            quintic.len() <= split_len,
            "six-way split left an oversized high part"
        );
        Parts {
            constant,
            linear,
            quadratic,
            cubic,
            quartic,
            quintic,
            sextic: &[],
        }
    }

    /// Splits `values` into seven parts, for the degree-six operand of a 6.5 split.
    pub fn split_seven(values: &[Limb], split_len: usize) -> Parts<'_> {
        let sextic_start = split_len.wrapping_mul(6).min(values.len());
        let (lower, sextic) = values.split_at(sextic_start);
        let mut parts = Self::split_six(lower, split_len);
        debug_assert!(
            sextic.len() <= split_len,
            "seven-way split left an oversized high part"
        );
        parts.sextic = sextic;
        parts
    }
}

impl Toom6 {
    /// Evaluates `parts` at one point into an even and an odd accumulator.
    ///
    /// Returns whether the odd accumulator exceeds the even one, which is exactly
    /// whether the conjugate point `A(-k)` is negative.
    pub fn evaluate_even_odd(
        even: &mut [Limb],
        odd: &mut [Limb],
        parts: Parts<'_>,
        direction: EvaluationDirection,
        shift: u32,
        kernel: AddMulKernel,
    ) -> bool {
        assert_eq!(even.len(), odd.len(), "evaluation widths must match");
        assert!(shift <= 2, "Toom-6 point shift exceeds four");
        assert!(
            [
                parts.constant,
                parts.linear,
                parts.quadratic,
                parts.cubic,
                parts.quartic,
                parts.quintic,
                parts.sextic,
            ]
            .into_iter()
            .all(|part| part.len() < even.len()),
            "each Toom-6 part must fit below the evaluation guard"
        );
        match direction {
            EvaluationDirection::Direct => evaluate_direct(even, odd, parts, shift, kernel),
            EvaluationDirection::Reciprocal => evaluate_reciprocal(even, odd, parts, shift, kernel),
        }
        even.iter().rev().cmp(odd.iter().rev()).is_lt()
    }
}

#[allow(
    unsafe_code,
    reason = "Every Toom-6 polynomial part fits below the retained evaluation guard"
)]
fn evaluate_direct(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point_shift: u32,
    kernel: AddMulKernel,
) {
    if point_shift != 0 {
        evaluate_direct_weighted(even, odd, parts, point_shift, kernel);
        return;
    }

    // At k=1, plain additions avoid scalar multiplication. The possible
    // sextic part extends E by a6 for the 6.5-way degree-six operand.
    SharedEval::copy_part(even, parts.quartic);
    // SAFETY: every `Parts` field is at most `split_len` limbs, while `even`
    // retains the evaluation's additional guard limbs.
    unsafe {
        SharedEval::add_part(even, parts.sextic);
    }
    // SAFETY: the same split-width bound applies to the quadratic part.
    unsafe {
        SharedEval::add_part(even, parts.quadratic);
    }
    // SAFETY: the same split-width bound applies to the constant part.
    unsafe {
        SharedEval::add_part(even, parts.constant);
    }

    SharedEval::copy_part(odd, parts.quintic);
    // SAFETY: `odd` has the same guarded width and both remaining odd parts
    // have at most `split_len` limbs.
    unsafe {
        SharedEval::add_part(odd, parts.cubic);
    }
    // SAFETY: the same guarded split-width bound applies to the linear part.
    unsafe {
        SharedEval::add_part(odd, parts.linear);
    }
}

fn evaluate_direct_weighted(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point_shift: u32,
    kernel: AddMulKernel,
) {
    let squared_shift = point_shift.wrapping_mul(2);
    let squared_scalar = 1_usize.wrapping_shl(squared_shift);
    let fourth_scalar = squared_scalar.wrapping_mul(squared_scalar);
    let sixth_scalar = fourth_scalar.wrapping_mul(squared_scalar);

    SharedEval::copy_part(even, parts.constant);
    SharedEval::add_mul_word_with_kernel_in_place(even, parts.quadratic, squared_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(even, parts.quartic, fourth_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(even, parts.sextic, sixth_scalar, kernel);

    SharedEval::copy_part(odd, parts.linear);
    SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(odd, parts.quintic, fourth_scalar, kernel);
    shift_left(odd, point_shift);
}

fn evaluate_reciprocal(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    denominator_shift: u32,
    kernel: AddMulKernel,
) {
    let squared_shift = denominator_shift.wrapping_mul(2);
    // For d=2^denominator_shift, these are the even and odd terms of the
    // integral scaled value d^5*A(1/d):
    //   E=d(a4+d^2(a2+d^2*a0)), O=a5+d^2(a3+d^2*a1).
    // Thus E+O=d^5*A(1/d) and E-O=d^5*A(-1/d), with no fractional limbs.
    let squared_scalar = 1_usize.wrapping_shl(squared_shift);
    let fourth_scalar = squared_scalar.wrapping_mul(squared_scalar);

    if parts.sextic.is_empty() {
        SharedEval::copy_part(even, parts.quartic);
        SharedEval::add_mul_word_with_kernel_in_place(
            even,
            parts.quadratic,
            squared_scalar,
            kernel,
        );
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.constant, fourth_scalar, kernel);
        shift_left(even, denominator_shift);

        SharedEval::copy_part(odd, parts.quintic);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.linear, fourth_scalar, kernel);
    } else {
        let sixth_scalar = fourth_scalar.wrapping_mul(squared_scalar);
        SharedEval::copy_part(even, parts.sextic);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.quartic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.quadratic, fourth_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.constant, sixth_scalar, kernel);

        SharedEval::copy_part(odd, parts.quintic);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.linear, fourth_scalar, kernel);
        shift_left(odd, denominator_shift);
    }
}

fn shift_left(value: &mut [Limb], shift: u32) {
    if shift == 0 || value.is_empty() {
        return;
    }
    // SAFETY: `value` supplies exactly `value.len()` writable limbs, and the
    // entry validation bounds point shifts by two, so this helper receives at
    // most the squared shift four, below the minimum `LIMB_BITS = 16`.
    // The 1365*B^m evaluation bound proves the returned top carry is zero.
    let carry = unsafe { ArchKernels::lshift_unchecked(value.as_mut_ptr(), value.len(), shift) };
    debug_assert_eq!(carry, 0, "evaluation exceeded its guard limb");
}
