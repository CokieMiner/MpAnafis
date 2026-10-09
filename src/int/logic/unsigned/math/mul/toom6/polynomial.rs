//! The six- and seven-part operand view, and its evaluation at a single point.
//!
//! A Toom-6 operand is a degree-five polynomial in `B^split_len` (degree six for
//! the 6.5 split). Every point this tier uses is a power of two, so evaluation is
//! a scaled accumulation and never a general multiplication.
//!
//! Each point is evaluated as two accumulators rather than one: the even-degree
//! terms and the odd-degree terms. `A(k) = E + O` and `A(-k) = E - O`, so one
//! evaluation pass serves both of a conjugate pair.

#![expect(
    unsafe_code,
    reason = "Admitted operand parts fit guarded evaluations; the point schedule bounds every weight and shift on 16-, 32-, and 64-bit limbs"
)]

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

/// The complete Toom-6 point schedule: `2^s` or its reciprocal, with `s<=2`.
/// The discriminant range carries the shift bound through every evaluation.
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum PointShift {
    Zero = 0,
    One = 1,
    Two = 2,
}

/// Positive point exponents after the driver handles the unit point once.
#[derive(Clone, Copy)]
#[repr(u32)]
enum EvaluationShift {
    One = 1,
    Two = 2,
}

impl Toom6 {
    /// Splits `values` into six parts of `split_len` limbs, the last possibly short.
    pub fn split_six(values: &[Limb], split_len: usize) -> Parts<'_> {
        let constant_len = min(values.len(), split_len);
        // SAFETY: constant_len is bounded by values.len() by construction.
        let (constant, after_constant) = unsafe { values.split_at_unchecked(constant_len) };
        let linear_len = min(after_constant.len(), split_len);
        // SAFETY: linear_len is the minimum of the remaining length and m.
        let (linear, after_linear) = unsafe { after_constant.split_at_unchecked(linear_len) };
        let quadratic_len = min(after_linear.len(), split_len);
        // SAFETY: quadratic_len is bounded by after_linear.len().
        let (quadratic, after_quadratic) =
            unsafe { after_linear.split_at_unchecked(quadratic_len) };
        let cubic_len = min(after_quadratic.len(), split_len);
        // SAFETY: cubic_len is bounded by after_quadratic.len().
        let (cubic, after_cubic) = unsafe { after_quadratic.split_at_unchecked(cubic_len) };
        let quartic_len = min(after_cubic.len(), split_len);
        // SAFETY: quartic_len is bounded by after_cubic.len().
        let (quartic, quintic) = unsafe { after_cubic.split_at_unchecked(quartic_len) };
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

    /// Split an admitted degree-six operand with six full low blocks.
    pub fn split_seven(values: &[Limb], split_len: usize) -> Parts<'_> {
        // SAFETY: this splitter is used by an admitted seven-part operand,
        // whose nonempty high block establishes 6m<values.len().
        let sextic_start = unsafe { split_len.unchecked_mul(6) };
        // SAFETY: seven-by-six admission proves the six full low blocks exist.
        let (lower, sextic) = unsafe { values.split_at_unchecked(sextic_start) };
        let mut parts = Self::split_six(lower, split_len);
        debug_assert!(
            sextic.len() <= split_len,
            "seven-way split left an oversized high part"
        );
        parts.sextic = sextic;
        parts
    }
    /// Evaluates `parts` at one point into an even and an odd accumulator.
    ///
    /// Returns whether the odd accumulator exceeds the even one, which is exactly
    /// whether the conjugate point `A(-k)` is negative.
    pub fn evaluate_even_odd(
        even: &mut [Limb],
        odd: &mut [Limb],
        parts: Parts<'_>,
        direction: EvaluationDirection,
        shift: PointShift,
        kernel: AddMulKernel,
    ) -> bool {
        debug_assert_eq!(even.len(), odd.len(), "evaluation widths must match");
        debug_assert!(
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
        let evaluation_shift = match shift {
            PointShift::Zero => {
                // Both integral point forms coincide at denominator one.
                evaluate_direct_one(even, odd, parts);
                return even.iter().rev().cmp(odd.iter().rev()).is_lt();
            }
            PointShift::One => EvaluationShift::One,
            PointShift::Two => EvaluationShift::Two,
        };
        match direction {
            EvaluationDirection::Direct => {
                evaluate_direct_weighted(even, odd, parts, evaluation_shift, kernel);
            }
            EvaluationDirection::Reciprocal => {
                evaluate_reciprocal(even, odd, parts, evaluation_shift, kernel);
            }
        }
        even.iter().rev().cmp(odd.iter().rev()).is_lt()
    }
}

fn evaluate_direct_one(even: &mut [Limb], odd: &mut [Limb], parts: Parts<'_>) {
    // At k=1, plain additions avoid scalar multiplication. The possible
    // sextic part extends E by a6 for the 6.5-way degree-six operand.
    // The constant and linear blocks are full width in every admitted shape;
    // initialize from them so short high blocks require no body zero extension.
    SharedEval::copy_part(even, parts.constant);
    // SAFETY: every `Parts` field is at most `split_len` limbs, while `even`
    // retains the evaluation's additional guard limbs.
    unsafe {
        SharedEval::add_part(even, parts.sextic);
    }
    // SAFETY: the same split-width bound applies to the quadratic part.
    unsafe {
        SharedEval::add_part(even, parts.quadratic);
    }
    // SAFETY: the same split-width bound applies to the quartic part.
    unsafe {
        SharedEval::add_part(even, parts.quartic);
    }

    SharedEval::copy_part(odd, parts.linear);
    // SAFETY: `odd` has the same guarded width and both remaining odd parts
    // have at most `split_len` limbs.
    unsafe {
        SharedEval::add_part(odd, parts.cubic);
    }
    // SAFETY: the same guarded split-width bound applies to the quintic part.
    unsafe {
        SharedEval::add_part(odd, parts.quintic);
    }
}

fn evaluate_direct_weighted(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
    kernel: AddMulKernel,
) {
    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift's u32 discriminant is one or two on every pointer width"
    )]
    let point_shift = point as u32;
    // SAFETY: EvaluationShift gives 1<=point_shift<=2, squared_shift<=4 and squared_scalar<=16.
    // Its square is <=256, below usize::MAX even on a 16-bit target.
    let (squared_scalar, fourth_scalar) = unsafe {
        let squared_scalar = 1_usize.unchecked_shl(point_shift.unchecked_mul(2));
        (squared_scalar, squared_scalar.unchecked_mul(squared_scalar))
    };

    SharedEval::copy_part(even, parts.constant);
    SharedEval::add_mul_word_with_kernel_in_place(even, parts.quadratic, squared_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(even, parts.quartic, fourth_scalar, kernel);
    if !parts.sextic.is_empty() {
        // SAFETY: squared_scalar<=16 and fourth_scalar<=256 imply sixth<=4096,
        // within a 16-bit limb; plain six-way operands need no sextic weight.
        let sixth_scalar = unsafe { fourth_scalar.unchecked_mul(squared_scalar) };
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.sextic, sixth_scalar, kernel);
    }

    SharedEval::copy_part(odd, parts.linear);
    SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(odd, parts.quintic, fourth_scalar, kernel);
    // SAFETY: odd is an initialized guarded evaluation; 1<=point_shift<=2<Limb::BITS.
    // The degree-six bound 5461*B^m is below B^(m+1) even on 16-bit limbs.
    let carry = unsafe { ArchKernels::lshift_unchecked(odd.as_mut_ptr(), odd.len(), point_shift) };
    debug_assert_eq!(carry, 0, "odd evaluation exceeds its guard");
}

fn evaluate_reciprocal(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
    kernel: AddMulKernel,
) {
    // For d=2^denominator_shift, these are the even and odd terms of the
    // integral scaled value d^5*A(1/d):
    //   E=d(a4+d^2(a2+d^2*a0)), O=a5+d^2(a3+d^2*a1).
    // Thus E+O=d^5*A(1/d) and E-O=d^5*A(-1/d), with no fractional limbs.
    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift's u32 discriminant is one or two on 16-, 32-, and 64-bit targets"
    )]
    let denominator_shift = point as u32;
    // SAFETY: EvaluationShift bounds denominator_shift in 1..=2, giving a shift<=4, scalar<=16, and its
    // square<=256. All operations fit the minimum supported 16-bit limb.
    let (squared_scalar, fourth_scalar) = unsafe {
        let squared_scalar = 1_usize.unchecked_shl(denominator_shift.unchecked_mul(2));
        (squared_scalar, squared_scalar.unchecked_mul(squared_scalar))
    };

    if parts.sextic.is_empty() {
        SharedEval::copy_part(even, parts.quartic);
        SharedEval::add_mul_word_with_kernel_in_place(
            even,
            parts.quadratic,
            squared_scalar,
            kernel,
        );
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.constant, fourth_scalar, kernel);
        // SAFETY: even retains its initialized guarded width; the positive
        // denominator shift is <=2<Limb::BITS and its scaled evaluation fits.
        let carry = unsafe {
            ArchKernels::lshift_unchecked(even.as_mut_ptr(), even.len(), denominator_shift)
        };
        debug_assert_eq!(carry, 0, "even reciprocal evaluation exceeds its guard");

        SharedEval::copy_part(odd, parts.quintic);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.linear, fourth_scalar, kernel);
    } else {
        // SAFETY: 256*16=4096 fits even the minimum 16-bit limb width.
        let sixth_scalar = unsafe { fourth_scalar.unchecked_mul(squared_scalar) };
        SharedEval::copy_part(even, parts.sextic);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.quartic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.quadratic, fourth_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.constant, sixth_scalar, kernel);

        SharedEval::copy_part(odd, parts.quintic);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.cubic, squared_scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(odd, parts.linear, fourth_scalar, kernel);
        // SAFETY: odd retains its initialized guarded width; the same positive
        // shift and degree-six bound prevent escaping carry on every limb width.
        let carry = unsafe {
            ArchKernels::lshift_unchecked(odd.as_mut_ptr(), odd.len(), denominator_shift)
        };
        debug_assert_eq!(carry, 0, "odd reciprocal evaluation exceeds its guard");
    }
}
