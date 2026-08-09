//! Four-by-three operand evaluation and the six-point interpolation.

use super::{AddMulKernel, Limb, SharedEval, Toom43};

impl Toom43 {
    /// Evaluate a four-part operand at `+x` and `-x` for `x` in `{1, 2}`.
    ///
    /// `A(x) = a0 + a1*x + a2*x^2 + a3*x^3` splits into an even half `a0 + a2*x^2`
    /// and an odd half `x*(a1 + a3*x^2)`; the two points differ only in the sign of
    /// the odd half, so one pass over the parts produces both. Returns whether
    /// `A(-x)` was negative, its magnitude having been stored.
    pub fn evaluate_four_parts<const AT_TWO: bool>(
        positive: &mut [Limb],
        negative_magnitude: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
        add_mul_kernel: AddMulKernel,
    ) -> bool {
        let scale: Limb = if AT_TWO { 4 } else { 1 };
        SharedEval::copy_part(positive, part0);
        SharedEval::add_mul_word_with_kernel_in_place(positive, part2, scale, add_mul_kernel);
        SharedEval::copy_part(negative_magnitude, part1);
        SharedEval::add_mul_word_with_kernel_in_place(
            negative_magnitude,
            part3,
            scale,
            add_mul_kernel,
        );
        if AT_TWO {
            // The odd half carries one factor of x. At x = 2 that is a doubling,
            // and a1 + 4*a3 < 5*B^m leaves 10*B^m, which the guard limb holds.
            SharedEval::double_evaluation_in_place(negative_magnitude);
        }
        SharedEval::sum_and_absolute_difference(positive, negative_magnitude)
    }

    /// Evaluate a three-part operand at `+x` and `-x` for `x` in `{1, 2}`.
    ///
    /// `B(x) = b0 + b1*x + b2*x^2` has even half `b0 + b2*x^2` and odd half `x*b1`.
    pub fn evaluate_three_parts<const AT_TWO: bool>(
        positive: &mut [Limb],
        negative_magnitude: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        add_mul_kernel: AddMulKernel,
    ) -> bool {
        let scale: Limb = if AT_TWO { 4 } else { 1 };
        SharedEval::copy_part(positive, part0);
        SharedEval::add_mul_word_with_kernel_in_place(positive, part2, scale, add_mul_kernel);
        SharedEval::copy_part(negative_magnitude, part1);
        if AT_TWO {
            SharedEval::double_evaluation_in_place(negative_magnitude);
        }
        SharedEval::sum_and_absolute_difference(positive, negative_magnitude)
    }
}

/// The four evaluated products that need interpolating, with their signs.
pub struct MiddleProducts<'buffer> {
    pub one: &'buffer mut [Limb],
    pub negative_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub negative_two: &'buffer mut [Limb],
    pub one_is_negative: bool,
    pub two_is_negative: bool,
}

/// The four interpolated middle coefficients, lowest first.
pub struct MiddleCoefficients<'buffer> {
    pub linear: &'buffer mut [Limb],
    pub quadratic: &'buffer mut [Limb],
    pub cubic: &'buffer mut [Limb],
    pub quartic: &'buffer mut [Limb],
}

impl Toom43 {
    /// Recover `c1..c4` from the four middle products and the two exact endpoints.
    ///
    /// For `W(x) = c0 + c1*x + ... + c5*x^5`, the endpoints are already exact —
    /// `c0 = W(0)` and `c5 = W(inf)` sit in the destination — and the two signed
    /// pairs separate the rest into even and odd systems that never mix:
    ///
    /// ```text
    /// S1 = (W(1) + W(-1)) / 2 = c0 +   c2 +    c4
    /// S2 = (W(2) + W(-2)) / 2 = c0 + 4*c2 + 16*c4
    /// D1 = (W(1) - W(-1)) / 2 = c1 +   c3 +    c5
    /// D2 = (W(2) - W(-2)) / 4 = c1 + 4*c3 + 16*c5
    /// ```
    ///
    /// Removing the endpoints leaves two two-by-two systems whose eliminations are
    /// `S2 - 4*S1 = 12*c4` and `D2 - D1 = 3*c3`. Every division is exact by
    /// construction, and none is by a power of two alone, which is why this tier
    /// carries a division by three where the three-by-two split carries none.
    pub fn interpolate_middle<'buffer>(
        products: MiddleProducts<'buffer>,
        zero: &[Limb],
        infinity: &[Limb],
    ) -> MiddleCoefficients<'buffer> {
        let MiddleProducts {
            one,
            negative_one,
            two,
            negative_two,
            one_is_negative,
            two_is_negative,
        } = products;

        let (sum_one, difference_one) = signed_pair(one, negative_one, one_is_negative);
        let (sum_two, difference_two) = signed_pair(two, negative_two, two_is_negative);

        // S1 and S2; the halving is exact because each sum is twice an even system.
        SharedEval::exact_div2_in_place(sum_one);
        SharedEval::exact_div2_in_place(sum_two);
        // D1 likewise. D2 divides by four: W(2) - W(-2) is 4*c1 + 16*c3 + 64*c5.
        SharedEval::exact_div2_in_place(difference_one);
        SharedEval::exact_div4_in_place(difference_two);

        // Drop the endpoints, leaving c2 + c4 / 4*c2 + 16*c4 and c1 + c3 / c1 + 4*c3.
        SharedEval::sub_full_slices_in_place(sum_one, zero);
        SharedEval::sub_full_slices_in_place(sum_two, zero);
        SharedEval::sub_full_slices_in_place(difference_one, infinity);
        SharedEval::sub_mul_word_in_place(difference_two, infinity, 16);

        // Evens: S2 - 4*S1 = 12*c4.
        SharedEval::sub_mul_word_in_place(sum_two, sum_one, 4);
        SharedEval::exact_div4_in_place(sum_two);
        SharedEval::exact_div_radix_minus_one_in_place::<3>(sum_two);
        // S1 - c4 = c2.
        SharedEval::sub_full_slices_in_place(sum_one, sum_two);

        // Odds: D2 - D1 = 3*c3, then D1 - c3 = c1.
        SharedEval::sub_full_slices_in_place(difference_two, difference_one);
        SharedEval::exact_div_radix_minus_one_in_place::<3>(difference_two);
        SharedEval::sub_full_slices_in_place(difference_one, difference_two);

        MiddleCoefficients {
            linear: difference_one,
            quadratic: sum_one,
            cubic: difference_two,
            quartic: sum_two,
        }
    }
}

/// Turn a product and a stored magnitude into their mathematical sum and
/// difference.
///
/// `W(x) >= |W(-x)|` because every polynomial part is a nonnegative magnitude,
/// so `A(x) >= |A(-x)|` and `B(x) >= |B(-x)|`. The butterfly is therefore always
/// oriented, and only the *naming* of its two outputs depends on the sign.
fn signed_pair<'buffer>(
    positive: &'buffer mut [Limb],
    negative_magnitude: &'buffer mut [Limb],
    is_negative: bool,
) -> (&'buffer mut [Limb], &'buffer mut [Limb]) {
    let flipped = SharedEval::sum_and_absolute_difference(positive, negative_magnitude);
    debug_assert!(
        !flipped,
        "the positive evaluated product must dominate the negative one"
    );
    if is_negative {
        (negative_magnitude, positive)
    } else {
        (positive, negative_magnitude)
    }
}
