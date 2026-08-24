//! Tier selection: turning a pair of widths into one plan.

use super::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, MulPlan, Multiplication,
    SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD, SQR_TOOM_COOK_6_THRESHOLD,
    SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, SquarePlan, TOOM_COOK_4_THRESHOLD,
    TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD, TierCeiling, Widths,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, SQR_SSA_THRESHOLD, SSA_THRESHOLD, Ssa};

impl Multiplication {
    /// Select a multiplication strategy whose conventional tier cannot exceed
    /// `ceiling`.
    ///
    /// Tiers are offered highest first, and each is asked two independent
    /// questions: does its crossover admit these widths, and can it compute this
    /// shape at all. Splitting the two is what lets a transform be named: it has no
    /// split of any arity, so it answers the shape question unconditionally, while
    /// every Toom tier answers it from its own split geometry.
    ///
    /// The transforms are offered before the Toom tiers rather than after them.
    /// Ordering them last denied the only quasi-linear tier to every shape that
    /// failed a Toom split test, however large the operands: a ratio worse than 4:3
    /// fails the four-way split and fell to Toom-3, and one that then failed both
    /// six-way tests fell to Toom-4. Measured against GMP that cost 3.8x at
    /// two-to-one and 7.8x at four-to-one, with the transform reproducing the
    /// identical product when invoked directly on the same operands.
    #[inline]
    pub fn select_plan(len_a: usize, len_b: usize, ceiling: TierCeiling) -> MulPlan {
        let widths = Widths::new(len_a, len_b);
        if widths.smaller < KARATSUBA_THRESHOLD {
            return MulPlan::Schoolbook;
        }

        #[cfg(not(target_pointer_width = "16"))]
        if ceiling == TierCeiling::Full {
            // Keying the crossover on the longer operand lets extreme ratios clear
            // it very early, where the padding is most of the ring: at 4091 by 255
            // limbs the transform pads a 255-limb operand into a 2173-limb ring and
            // measured 1.43x behind the reference against the blocked path's 0.91x.
            if widths.transform_padding_is_affordable()
                && Self::crossover_admits(SSA_THRESHOLD, widths)
                && Ssa::admits_mul(len_a, len_b)
            {
                return MulPlan::Large(LargePlan::Ssa);
            }
        }

        // Balanced Toom-8 has a substantially earlier top-level crossover on
        // the measured x86-64 profile than the ordinary ordered Toom ladder.
        // Keep this gate at the full ceiling: lowering the Toom-6/8 thresholds
        // themselves would also retune every recursive child and lose the
        // performance this direct root is intended to preserve.
        if ceiling == TierCeiling::Full
            && widths.smaller == widths.larger
            && Self::crossover_admits(BALANCED_TOOM8_THRESHOLD, widths)
            && widths.toom8_balanced()
        {
            return MulPlan::Toom8;
        }

        // Offered above the blocked path because it competes with it, not with the
        // balanced Toom ladder: `prefers_blocked_product` fires from four-to-three
        // upward, so every shape this tier admits would otherwise be blocked. It
        // takes only the shapes where blocking's residue is genuinely bad, because
        // where the residue is near zero blocking measured strictly faster. The
        // width gate is Toom-3's, because this is a Toom-3-class split — four point
        // products at the same width, one fewer than the balanced tier spends on the
        // same operands after zero-extending the shorter one.
        if matches!(ceiling, TierCeiling::Toom6 | TierCeiling::Full)
            && Self::crossover_admits(TOOM_COOK_THRESHOLD, widths)
            && widths.prefers_fractional_split()
        {
            // Both splits admit `[1.5, 2)`. The three-way one is offered first
            // there: it spends four recursive products where the four-way one
            // spends six, and its interpolation is a butterfly and a halving
            // against a solve carrying a division by three.
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
        // A Toom-6 child whose split has collapsed goes straight to basecase. A
        // Toom-4 child instead keeps the lower conventional tower even when
        // lopsided, because its evaluated products are admitted there directly.
        if ceiling == TierCeiling::Toom4 && widths.degenerate_child_split() {
            return MulPlan::Schoolbook;
        }

        select_conventional_mul_plan(widths, ceiling)
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

/// The Toom ladder, once the transforms and the blocked path have declined.
#[inline]
fn select_conventional_mul_plan(widths: Widths, ceiling: TierCeiling) -> MulPlan {
    if !Multiplication::crossover_admits(TOOM_COOK_THRESHOLD, widths) {
        return MulPlan::Karatsuba;
    }

    let selects_toom4 =
        Multiplication::crossover_admits(TOOM_COOK_4_THRESHOLD, widths) && widths.toom4_balanced();
    if ceiling == TierCeiling::Toom3 || !selects_toom4 {
        return MulPlan::Toom3;
    }

    let selects_toom6 = Multiplication::crossover_admits(TOOM_COOK_6_THRESHOLD, widths)
        && (widths.toom6_balanced() || widths.toom6_half_suitable());
    if ceiling == TierCeiling::Toom4 || !selects_toom6 {
        return MulPlan::Toom4;
    }

    let selects_toom8 = Multiplication::crossover_admits(TOOM_COOK_85_THRESHOLD, widths)
        && (widths.toom8_balanced() || widths.toom8_half_suitable());
    if ceiling == TierCeiling::Toom6 || !selects_toom8 {
        return MulPlan::Toom6;
    }
    MulPlan::Toom8
}
