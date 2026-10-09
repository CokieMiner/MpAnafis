//! Tier selection: turning a pair of widths into one plan.

use super::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, MulPlan, Multiplication,
    SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD, SQR_TOOM_COOK_6_THRESHOLD,
    SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, SquarePlan, TOOM_COOK_4_THRESHOLD,
    TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD, TierCeiling, Widths,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, SQR_SSA_THRESHOLD, SSA_THRESHOLD, Ssa, SsaPlan, Toom8};

impl Multiplication {
    /// Counts the selected square tower's recursive products and linear visits.
    /// Schoolbook squares evaluate the triangular product matrix; higher tiers
    /// preserve that square recurrence at every evaluation point.
    #[cfg(not(target_pointer_width = "16"))]
    pub fn structural_square_work(len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let parts: usize = match Self::select_square_plan(len, TierCeiling::Full) {
            SquarePlan::Schoolbook => {
                // One diagonal and one product for each unordered off-diagonal
                // pair, plus a full-width doubling/carry pass.
                return len
                    .saturating_mul(len.saturating_add(1))
                    .div_euclid(2)
                    .saturating_add(len.saturating_mul(2));
            }
            SquarePlan::Karatsuba => 2,
            SquarePlan::Toom3 => 3,
            SquarePlan::Toom4 => 4,
            SquarePlan::Toom6 => 6,
            SquarePlan::Toom8 => Toom8::BALANCED_PARTS,
            #[cfg(not(target_pointer_width = "16"))]
            SquarePlan::Large(LargePlan::Ssa) => {
                return SsaPlan::square_work(len).unwrap_or(usize::MAX);
            }
        };
        // Squaring has equal polynomial degrees: 2*parts-1 coefficients.
        let points = parts.saturating_mul(2).saturating_sub(1);
        let child = len.div_ceil(parts).saturating_add(1);
        // Guard padding must not prevent model termination at tiny cutoffs.
        let child_work = if child >= len {
            child
                .saturating_mul(child)
                .saturating_add(child.saturating_mul(2))
        } else {
            Self::structural_square_work(child)
        };
        child_work
            .saturating_mul(points)
            .saturating_add(len.saturating_mul(points))
    }

    /// Returns dispatched product work from operand capacities alone.
    ///
    /// Transform products retain the CRT rounding and recursively planned
    /// Fermat geometries. Every conventional tier follows its own split
    /// recurrence and point count, while lopsided multiplication follows its
    /// block decomposition. The units count structural limb work and are used
    /// only to compare shapes of the same enclosing operation.
    #[cfg(not(target_pointer_width = "16"))]
    #[must_use]
    pub fn structural_product_work(len_a: usize, len_b: usize) -> usize {
        let widths = Widths::new(len_a, len_b);
        if widths.smaller == 0 {
            return 0;
        }
        let plan = Self::select_plan(len_a, len_b, TierCeiling::Full);
        #[cfg(not(target_pointer_width = "16"))]
        if plan == MulPlan::Large(LargePlan::Ssa) {
            return SsaPlan::product_work(len_a, len_b).unwrap_or(usize::MAX);
        }
        conventional_product_work(plan, widths)
    }

    /// Selects a product plan within the conventional tier ceiling.
    ///
    /// Each tier must satisfy its crossover and operand-shape constraints.
    /// The full tower considers transforms before polynomial splits.
    #[inline]
    pub fn select_plan(len_a: usize, len_b: usize, ceiling: TierCeiling) -> MulPlan {
        let widths = Widths::new(len_a, len_b);
        if widths.smaller < KARATSUBA_THRESHOLD {
            return MulPlan::Schoolbook;
        }

        #[cfg(not(target_pointer_width = "16"))]
        if ceiling == TierCeiling::Full {
            // The aspect-ratio bound limits zero-padding at the transform crossover.
            if widths.transform_padding_is_affordable()
                && Self::crossover_admits(SSA_THRESHOLD, widths)
                && Ssa::admits_mul(len_a, len_b)
            {
                return MulPlan::Large(LargePlan::Ssa);
            }
        }

        // The equal-width root crossover applies only to the full tower.
        if ceiling == TierCeiling::Full
            && widths.smaller == widths.larger
            && Self::crossover_admits(BALANCED_TOOM8_THRESHOLD, widths)
            && widths.toom8_balanced()
        {
            return MulPlan::Toom8;
        }

        // Fractional splits precede blocked products in their admitted ratio band.
        if matches!(ceiling, TierCeiling::Toom6 | TierCeiling::Full)
            && Self::crossover_admits(TOOM_COOK_THRESHOLD, widths)
            && widths.prefers_fractional_split()
        {
            // Three-by-two uses four point products; four-by-three uses six.
            if widths.toom32_suitable() {
                return MulPlan::Toom32;
            }
            if widths.toom43_suitable() {
                return MulPlan::Toom43;
            }
        }
        if matches!(ceiling, TierCeiling::Toom6 | TierCeiling::Full)
            && widths.prefers_blocked_product()
        {
            return MulPlan::Lopsided;
        }
        // Collapsed splits under the Toom-4 ceiling use the basecase kernel.
        if ceiling == TierCeiling::Toom4 && widths.degenerate_child_split() {
            return MulPlan::Schoolbook;
        }

        // The conventional ladder follows directly once transform, fractional,
        // blocked, and degenerate-child plans have declined this shape.
        if !Self::crossover_admits(TOOM_COOK_THRESHOLD, widths) {
            return MulPlan::Karatsuba;
        }

        let selects_toom4 =
            Self::crossover_admits(TOOM_COOK_4_THRESHOLD, widths) && widths.toom4_balanced();
        if ceiling == TierCeiling::Toom3 || !selects_toom4 {
            // Toom-3 requires both operands to reach its crossover. Naming the
            // lower tier here avoids repeating this decision during execution.
            return if widths.smaller >= TOOM_COOK_THRESHOLD.max(3) {
                MulPlan::Toom3
            } else {
                MulPlan::Karatsuba
            };
        }

        let selects_toom6 = Self::crossover_admits(TOOM_COOK_6_THRESHOLD, widths)
            && (widths.toom6_balanced() || widths.toom6_half_suitable());
        if ceiling == TierCeiling::Toom4 || !selects_toom6 {
            return MulPlan::Toom4;
        }

        let selects_toom8 = Self::crossover_admits(TOOM_COOK_85_THRESHOLD, widths)
            && (widths.toom8_balanced() || widths.toom8_half_suitable());
        if ceiling == TierCeiling::Toom6 || !selects_toom8 {
            return MulPlan::Toom6;
        }
        MulPlan::Toom8
    }

    /// Select a squaring strategy whose conventional tier cannot exceed `ceiling`.
    #[inline]
    pub fn select_square_plan(len: usize, ceiling: TierCeiling) -> SquarePlan {
        if len < SQR_KARATSUBA_THRESHOLD {
            return SquarePlan::Schoolbook;
        }
        if len < SQR_TOOM_COOK_THRESHOLD {
            return SquarePlan::Karatsuba;
        }
        let widths = Widths::new(len, len);

        #[cfg(not(target_pointer_width = "16"))]
        if ceiling == TierCeiling::Full
            && Self::crossover_admits(SQR_SSA_THRESHOLD, widths)
            && Ssa::admits_sqr(len)
        {
            return SquarePlan::Large(LargePlan::Ssa);
        }

        if ceiling == TierCeiling::Toom3
            || !Self::crossover_admits(SQR_TOOM_COOK_4_THRESHOLD, widths)
        {
            return SquarePlan::Toom3;
        }
        if ceiling == TierCeiling::Toom4
            || !Self::crossover_admits(SQR_TOOM_COOK_6_THRESHOLD, widths)
        {
            return SquarePlan::Toom4;
        }
        if ceiling == TierCeiling::Toom6
            || !Self::crossover_admits(SQR_TOOM_COOK_85_THRESHOLD, widths)
            || !Self::operand_has_eight_parts(len)
        {
            return SquarePlan::Toom6;
        }
        SquarePlan::Toom8
    }
}

/// Structural recurrence of the selected conventional multiplication plan.
#[cfg(not(target_pointer_width = "16"))]
fn conventional_product_work(plan: MulPlan, widths: Widths) -> usize {
    match plan {
        MulPlan::Schoolbook => widths.smaller.saturating_mul(widths.larger),
        MulPlan::Lopsided => {
            let full_blocks = widths.larger.div_euclid(widths.smaller);
            let remainder = widths.larger.rem_euclid(widths.smaller);
            Multiplication::structural_product_work(widths.smaller, widths.smaller)
                .saturating_mul(full_blocks)
                .saturating_add(Multiplication::structural_product_work(
                    widths.smaller,
                    remainder,
                ))
        }
        MulPlan::Karatsuba => split_recurrence_work(widths, 2, 2),
        MulPlan::Toom32 => split_recurrence_work(widths, 3, 2),
        MulPlan::Toom3 => split_recurrence_work(widths, 3, 3),
        MulPlan::Toom43 => split_recurrence_work(widths, 4, 3),
        MulPlan::Toom4 => split_recurrence_work(widths, 4, 4),
        // Balanced k-by-k interpolation has 2*k-1 coefficients; adjacent
        // (k+1)-by-k interpolation has 2*k. Counts and chunk bounds follow
        // the same shape admission used by the arithmetic drivers.
        MulPlan::Toom6 => {
            split_recurrence_work(widths, if widths.toom6_balanced() { 6 } else { 7 }, 6)
        }
        MulPlan::Toom8 => split_recurrence_work(
            widths,
            if widths.toom8_balanced() {
                Toom8::BALANCED_PARTS
            } else {
                Toom8::HALF_LARGE_PARTS
            },
            Toom8::HALF_SMALL_PARTS,
        ),
        #[cfg(not(target_pointer_width = "16"))]
        MulPlan::Large(LargePlan::Ssa) => usize::MAX,
    }
}

/// Upper-bounds one Toom/Karatsuba node by equal-width evaluation products.
///
/// Evaluation values carry one guard limb. Endpoint products are narrower, so
/// charging every point at `split + 1` is conservative while preserving the
/// tier's recursive exponent. One input-width visit per point represents the
/// linear split, evaluation, interpolation, and reconstruction passes.
#[cfg(not(target_pointer_width = "16"))]
fn split_recurrence_work(widths: Widths, larger_parts: usize, smaller_parts: usize) -> usize {
    let points = larger_parts.saturating_add(smaller_parts).saturating_sub(1);
    let child_width = widths
        .larger
        .div_ceil(larger_parts)
        .max(widths.smaller.div_ceil(smaller_parts))
        .saturating_add(1);
    // A guard can make a modeled two-/three-limb child as wide as its parent
    // under very low cutoffs. Price that bounded leaf directly; only strictly
    // shrinking estimates recurse. This preserves planner termination.
    let child_work = if child_width >= widths.larger {
        child_width.saturating_mul(child_width)
    } else {
        Multiplication::structural_product_work(child_width, child_width)
    };
    let products = child_work.saturating_mul(points);
    let linear = widths
        .larger
        .saturating_add(widths.smaller)
        .saturating_mul(points);
    products.saturating_add(linear)
}
