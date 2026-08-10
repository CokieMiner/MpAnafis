//! Two-part operand evaluation and the four-point Toom-3-by-2 interpolation.

use core::cmp::Ordering;

use super::{ArchKernels, Limb, SharedEval, Toom32};

impl Toom32 {
    /// Evaluate one two-part operand at `x = 1` and `x = -1` together.
    ///
    /// The negative evaluation is returned as an absolute magnitude; the return
    /// value records whether its mathematical value was negative. Fusing the two
    /// keeps one pass over the parts, and the sign is decided before the pass so the
    /// subtraction orientation is a const parameter rather than a per-limb branch.
    pub fn evaluate_one_and_negative_one(
        positive: &mut [Limb],
        negative_magnitude: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
    ) -> bool {
        debug_assert_eq!(
            positive.len(),
            negative_magnitude.len(),
            "paired evaluations require equal widths"
        );
        debug_assert!(
            part1.len() <= part0.len(),
            "linear part exceeds the constant part"
        );
        debug_assert!(
            part0.len() < positive.len(),
            "evaluation must leave at least one guard limb"
        );

        if SharedEval::compare_with_zero_extension(part0, part1) == Ordering::Less {
            form_sum_and_difference::<true>(positive, negative_magnitude, part0, part1);
            true
        } else {
            form_sum_and_difference::<false>(positive, negative_magnitude, part0, part1);
            false
        }
    }
}

/// Write `part0 + part1` and `|part0 - part1|` in one pass over the parts.
///
/// `LINEAR_MINUS_CONSTANT` selects the orientation whose difference is
/// nonnegative, which the caller established by comparison.
fn form_sum_and_difference<const LINEAR_MINUS_CONSTANT: bool>(
    sum: &mut [Limb],
    difference: &mut [Limb],
    part0: &[Limb],
    part1: &[Limb],
) {
    let (sum_body, sum_guard) = sum.split_at_mut(part0.len());
    let (difference_body, difference_guard) = difference.split_at_mut(part0.len());
    let (part0_shared, part0_extension) = part0.split_at(part1.len());
    let (sum_shared, sum_extension) = sum_body.split_at_mut(part1.len());
    let (difference_shared, difference_extension) = difference_body.split_at_mut(part1.len());

    let mut carry = false;
    let mut borrow = false;
    for (((sum_limb, difference_limb), part0_limb), part1_limb) in sum_shared
        .iter_mut()
        .zip(difference_shared)
        .zip(part0_shared)
        .zip(part1)
    {
        let (partial_sum, overflow_a) = part0_limb.overflowing_add(*part1_limb);
        let (complete_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        *sum_limb = complete_sum;
        carry = overflow_a | overflow_b;

        let (minuend, subtrahend) = if LINEAR_MINUS_CONSTANT {
            (*part1_limb, *part0_limb)
        } else {
            (*part0_limb, *part1_limb)
        };
        let (partial_difference, underflow_a) = minuend.overflowing_sub(subtrahend);
        let (complete_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        *difference_limb = complete_difference;
        borrow = underflow_a | underflow_b;
    }

    // Above `part1`'s width the linear part is zero, so the sum only propagates
    // its carry and the difference is either the constant part or its negation.
    for ((sum_limb, difference_limb), part0_limb) in sum_extension
        .iter_mut()
        .zip(difference_extension)
        .zip(part0_extension)
    {
        let (complete_sum, overflow) = part0_limb.overflowing_add(Limb::from(carry));
        *sum_limb = complete_sum;
        carry = overflow;

        let (minuend, subtrahend) = if LINEAR_MINUS_CONSTANT {
            (Limb::MIN, *part0_limb)
        } else {
            (*part0_limb, Limb::MIN)
        };
        let (partial_difference, underflow_a) = minuend.overflowing_sub(subtrahend);
        let (complete_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        *difference_limb = complete_difference;
        borrow = underflow_a | underflow_b;
    }

    // b0 + b1 < 2*B^m, so the sum's carry is zero or one and lands in the guard.
    sum_guard.fill(Limb::from(carry));
    difference_guard.fill(0);
    debug_assert!(!borrow, "absolute difference underflowed");
}

// ── Four-point interpolation ─────────────────────────────────────────────────

impl Toom32 {
    /// Recover the two middle coefficients from the two middle products.
    ///
    /// For `W(x) = c0 + c1*x + c2*x^2 + c3*x^3` the endpoints are already exact:
    /// `c0 = W(0)` and `c3 = W(inf)`, both written straight into the destination.
    /// The remaining pair follows from one butterfly and one halving each,
    ///
    /// ```text
    /// W(1) + W(-1) = 2*(c0 + c2)
    /// W(1) - W(-1) = 2*(c1 + c3)
    /// ```
    ///
    /// which is the whole reason this tier is cheaper than running the balanced
    /// five-point solve on a zero-extended operand: no division by three, no
    /// division by four, and no evaluation at two.
    ///
    /// Returns `(c1, c2)`. Which buffer holds which depends on the sign of `W(-1)`,
    /// because only its magnitude is stored — a negative `W(-1)` swaps the roles of
    /// the butterfly's sum and difference outputs.
    pub fn interpolate_middle<'buffer>(
        one: &'buffer mut [Limb],
        negative_one: &'buffer mut [Limb],
        zero: &[Limb],
        infinity: &[Limb],
        negative_one_is_negative: bool,
    ) -> (&'buffer mut [Limb], &'buffer mut [Limb]) {
        debug_assert_eq!(
            one.len(),
            negative_one.len(),
            "the two middle products must share one fixed width"
        );
        let width = one.len();
        // SAFETY: the two products occupy disjoint scratch spans of equal width.
        // The difference cannot underflow: every polynomial part is a nonnegative
        // magnitude, so A(1) >= |A(-1)| and B(1) >= |B(-1)|, hence W(1) >= |W(-1)|.
        let (carry, borrow) = unsafe {
            ArchKernels::add_sub_limbs_unchecked(one.as_mut_ptr(), negative_one.as_mut_ptr(), width)
        };
        debug_assert_eq!(carry, 0, "the middle sum exceeded its fixed width");
        debug_assert_eq!(borrow, 0, "the middle difference underflowed");

        // `one` now holds W(1) + |W(-1)| and `negative_one` holds W(1) - |W(-1)|.
        // Those are the mathematical sum and difference only when W(-1) was
        // nonnegative; otherwise each is the other.
        let (doubled_even, doubled_odd) = if negative_one_is_negative {
            (negative_one, one)
        } else {
            (one, negative_one)
        };
        SharedEval::exact_div2_in_place(doubled_even);
        SharedEval::exact_div2_in_place(doubled_odd);

        // c0 + c2 and c1 + c3 remain; the endpoints remove themselves.
        SharedEval::sub_full_slices_in_place(doubled_even, zero);
        SharedEval::sub_full_slices_in_place(doubled_odd, infinity);
        (doubled_odd, doubled_even)
    }
}
