//! Caller-owned workspace sizing for every multiplication tier.
//!
//! Two layers live here and they are one subject: the plan-to-size dispatch
//! that turns a selected [`MulPlan`]/[`SquarePlan`] into a limb count, and the
//! per-tier computations it dispatches to.
//!
//! Every tier function is self-contained: it checks the tier threshold, falls
//! back to the next-lower tier when the shape is unsuitable, and otherwise
//! computes the local layout plus the maximum recursive inner-space. The pair
//! of `*_scratch_len` / `*_forced_scratch_len` entry points differ only in
//! whether the configured crossover is consulted, so the guarded form is a
//! threshold check in front of the forced one.

use core::cmp::{max, min};

use crate::parallel::{DefaultExecutor, ParallelExecutor};

use super::{
    KARATSUBA_THRESHOLD, Karatsuba, Lopsided, MulPlan, MulShape, Multiplication,
    SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, SquarePlan, TOOM_COOK_THRESHOLD, TierCeiling,
    Toom3, Toom4, Toom6, Toom8, Toom32, Toom43, Widths,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa};

// ---------------------------------------------------------------------------
// Plan-to-scratch-size dispatch
// ---------------------------------------------------------------------------

/// Return the caller-owned workspace required by `plan`.
///
/// One variant, one algorithm, one layout. While a plan could name a transform
/// *and* a Toom fallback at once, this had to reserve the larger of two
/// unrelated layouts because nothing knew which would run.
impl Multiplication {
    #[inline]
    pub fn scratch_len(plan: MulPlan, len_a: usize, len_b: usize) -> usize {
        DefaultExecutor::with_resolved(|executor| {
            Self::scratch_len_for_parallelism(plan, len_a, len_b, executor.parallelism().get())
        })
    }

    /// Return the caller-owned workspace required by `plan` at one executor width.
    #[inline]
    pub fn scratch_len_for_parallelism(
        plan: MulPlan,
        len_a: usize,
        len_b: usize,
        parallelism: usize,
    ) -> usize {
        #[cfg(target_pointer_width = "16")]
        let _ = parallelism;
        match plan {
            MulPlan::Schoolbook => 0,
            MulPlan::Lopsided => {
                let widths = Widths::new(len_a, len_b);
                Lopsided::mul_forced_scratch_len(
                    len_a,
                    len_b,
                    Lopsided::block_len(widths.larger, widths.smaller),
                    parallelism,
                )
            }
            MulPlan::Karatsuba => Self::karatsuba_mul_scratch_len(len_a, len_b),
            MulPlan::Toom3 => Self::toom3_mul_scratch_len(len_a, len_b),
            MulPlan::Toom32 => Self::toom32_mul_scratch_len(len_a, len_b),
            MulPlan::Toom43 => Self::toom43_mul_scratch_len(len_a, len_b),
            MulPlan::Toom4 => Self::toom4_mul_scratch_len(len_a, len_b),
            MulPlan::Toom6 => Self::toom6_mul_scratch_len(len_a, len_b),
            MulPlan::Toom8 => Self::toom8_mul_scratch_len(len_a, len_b),
            #[cfg(not(target_pointer_width = "16"))]
            MulPlan::Large(LargePlan::Ssa) => {
                Ssa::mul_scratch_len_for_parallelism(len_a, len_b, parallelism)
            }
        }
    }

    /// Return the caller-owned workspace required by `plan`.
    #[inline]
    pub fn square_scratch_len(plan: SquarePlan, len: usize) -> usize {
        DefaultExecutor::with_resolved(|executor| {
            Self::square_scratch_len_for_parallelism(plan, len, executor.parallelism().get())
        })
    }

    /// Return the caller-owned square workspace required at one executor width.
    #[inline]
    pub fn square_scratch_len_for_parallelism(
        plan: SquarePlan,
        len: usize,
        parallelism: usize,
    ) -> usize {
        #[cfg(target_pointer_width = "16")]
        let _ = parallelism;
        match plan {
            SquarePlan::Schoolbook => 0,
            SquarePlan::Karatsuba => Self::karatsuba_sqr_scratch_len(len),
            SquarePlan::Toom3 => Self::toom3_sqr_scratch_len(len),
            SquarePlan::Toom4 => Self::toom4_sqr_scratch_len(len),
            SquarePlan::Toom6 => Self::toom6_sqr_scratch_len(len),
            SquarePlan::Toom8 => Self::toom8_sqr_scratch_len(len),
            #[cfg(not(target_pointer_width = "16"))]
            SquarePlan::Large(LargePlan::Ssa) => {
                Ssa::sqr_scratch_len_for_parallelism(len, parallelism)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lopsided
// ---------------------------------------------------------------------------

impl Multiplication {
    // ---------------------------------------------------------------------------
    // Karatsuba
    // ---------------------------------------------------------------------------

    pub fn karatsuba_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if min(len_a, len_b) < KARATSUBA_THRESHOLD {
            return 0;
        }
        Self::karatsuba_mul_forced_scratch_len(len_a, len_b)
    }

    pub fn karatsuba_mul_forced_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 2 || len_b < 2 || min(len_a, len_b) <= max(len_a, len_b).div_ceil(2) {
            return 0;
        }
        if len_a == len_b {
            match len_a {
                20 => return Karatsuba::BALANCED_20_SCRATCH_LIMBS,
                24 => return Karatsuba::BALANCED_24_SCRATCH_LIMBS,
                32 => return Karatsuba::BALANCED_32_SCRATCH_LIMBS,
                48 => return Karatsuba::BALANCED_48_SCRATCH_LIMBS,
                _ => {}
            }
            let split_len = Karatsuba::balanced_split_len(len_a);
            let high_len = len_a.wrapping_sub(split_len);
            let local_space = split_len.wrapping_mul(4).wrapping_add(1);
            let recursive_space = max(
                Self::karatsuba_mul_scratch_len(split_len, split_len),
                Self::karatsuba_mul_scratch_len(high_len, high_len),
            );
            return local_space.wrapping_add(recursive_space);
        }
        let split_len = max(len_a, len_b).div_ceil(2);
        let local_space = split_len.wrapping_add(1).wrapping_mul(4);
        let recursive_space = max(
            Self::karatsuba_mul_scratch_len(split_len.wrapping_add(1), split_len),
            max(
                Self::karatsuba_mul_scratch_len(split_len, split_len),
                Self::karatsuba_mul_scratch_len(
                    split_len.wrapping_add(1),
                    split_len.wrapping_add(1),
                ),
            ),
        );
        local_space.wrapping_add(recursive_space)
    }

    pub fn karatsuba_sqr_scratch_len(len: usize) -> usize {
        if len < SQR_KARATSUBA_THRESHOLD {
            return 0;
        }
        Self::karatsuba_sqr_forced_scratch_len(len)
    }

    pub fn karatsuba_sqr_forced_scratch_len(len: usize) -> usize {
        if len < 2 {
            return 0;
        }
        let split_len = len.div_ceil(2);
        let high_len = len.wrapping_sub(split_len);
        let local_space = split_len
            .wrapping_add(split_len.wrapping_mul(2))
            .wrapping_add(1);
        let recursive_space = max(
            Self::karatsuba_sqr_scratch_len(split_len),
            Self::karatsuba_sqr_scratch_len(high_len),
        );
        local_space.wrapping_add(recursive_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-3
    // ---------------------------------------------------------------------------

    pub fn toom3_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if min(len_a, len_b) < TOOM_COOK_THRESHOLD {
            return Self::karatsuba_mul_scratch_len(len_a, len_b);
        }
        Self::toom3_mul_forced_scratch_len(len_a, len_b)
    }

    pub fn toom3_mul_forced_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 3 || len_b < 3 {
            return Self::karatsuba_mul_scratch_len(len_a, len_b);
        }
        let split_len = max(len_a, len_b).div_ceil(3);
        let low_len_a = min(len_a, split_len);
        let low_len_b = min(len_b, split_len);
        let high_len_a = len_a.saturating_sub(split_len.wrapping_mul(2));
        let high_len_b = len_b.saturating_sub(split_len.wrapping_mul(2));
        let eval_len = split_len.wrapping_add(1);
        let evaluation_inner = Self::toom3_mul_scratch_len(eval_len, eval_len);
        let low_inner = Self::toom3_mul_scratch_len(low_len_a, low_len_b);
        let high_inner = Self::toom3_mul_scratch_len(high_len_a, high_len_b);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        Toom3::local_scratch_len(split_len, inner_space)
    }

    pub fn toom3_sqr_scratch_len(len: usize) -> usize {
        if len < SQR_TOOM_COOK_THRESHOLD {
            return Self::karatsuba_sqr_scratch_len(len);
        }
        Self::toom3_sqr_forced_scratch_len(len)
    }

    pub fn toom3_sqr_forced_scratch_len(len: usize) -> usize {
        if len < 3 {
            return Self::karatsuba_sqr_scratch_len(len);
        }
        let split_len = len.div_ceil(3);
        let low_len = min(len, split_len);
        let high_len = len.saturating_sub(split_len.wrapping_mul(2));
        let eval_len = split_len.wrapping_add(1);
        let evaluation_inner = Self::toom3_sqr_scratch_len(eval_len);
        let low_inner = Self::toom3_sqr_scratch_len(low_len);
        let high_inner = Self::toom3_sqr_scratch_len(high_len);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        Toom3::local_scratch_len(split_len, inner_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-4
    // ---------------------------------------------------------------------------

    pub fn toom4_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 4 || len_b < 4 {
            return Self::toom3_mul_scratch_len(len_a, len_b);
        }
        let split_len = max(len_a, len_b).div_ceil(4);
        if !Self::operand_has_four_parts(len_a, split_len)
            || !Self::operand_has_four_parts(len_b, split_len)
        {
            return Self::toom3_mul_scratch_len(len_a, len_b);
        }
        let eval_len = split_len.wrapping_add(1);
        let low_len_a = min(len_a, split_len);
        let low_len_b = min(len_b, split_len);
        let high_len_a = len_a.saturating_sub(split_len.wrapping_mul(3));
        let high_len_b = len_b.saturating_sub(split_len.wrapping_mul(3));
        let plan_eval = Self::select_plan(eval_len, eval_len, TierCeiling::Toom4);
        let evaluation_inner = Self::scratch_len(plan_eval, eval_len, eval_len);
        let low_inner = Self::toom3_mul_scratch_len(low_len_a, low_len_b);
        let high_inner = Self::toom3_mul_scratch_len(high_len_a, high_len_b);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        Toom4::local_scratch_len(split_len, inner_space)
    }

    pub fn toom4_sqr_scratch_len(len: usize) -> usize {
        if len < 4 {
            return Self::toom3_sqr_scratch_len(len);
        }
        let split_len = len.div_ceil(4);
        if !Self::operand_has_four_parts(len, split_len) {
            return Self::toom3_sqr_scratch_len(len);
        }
        let eval_len = split_len.wrapping_add(1);
        let low_len = min(len, split_len);
        let high_len = len.saturating_sub(split_len.wrapping_mul(3));
        let plan_eval = Self::select_square_plan(eval_len, TierCeiling::Toom3);
        let evaluation_inner = Self::square_scratch_len(plan_eval, eval_len);
        let low_inner = Self::toom3_sqr_scratch_len(low_len);
        let high_inner = Self::toom3_sqr_scratch_len(high_len);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        Toom4::local_scratch_len(split_len, inner_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-3 by 2
    // ---------------------------------------------------------------------------

    /// Workspace for one three-by-two level.
    ///
    /// The driver has no internal fallback, so this has none to mirror: the selector
    /// names the tier only for a shape `toom32_suitable` admits.
    ///
    /// It does have two *shapes* of child, and being narrower is not the same as
    /// needing less workspace. The two guarded point products and the `W(0)`
    /// endpoint are all `split_len` by `split_len`; the `W(inf)` endpoint is the two
    /// high parts, which are each at most `split_len` but generally unequal and
    /// unequal to each other. A narrower unbalanced pair selects a different plan
    /// with a different layout, and can need strictly more than the wider balanced
    /// pair — so the two are maximised rather than one assumed to bound the other.
    pub fn toom32_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let widths = Widths::new(len_a, len_b);
        debug_assert!(
            widths.toom32_suitable(),
            "three-by-two scratch asked for a shape the tier cannot split"
        );
        let split_len = widths.larger.div_ceil(3);
        let high_len_a = widths.larger.saturating_sub(split_len.wrapping_mul(2));
        let high_len_b = widths.smaller.saturating_sub(split_len);

        let balanced_child = Self::select_plan(split_len, split_len, TierCeiling::Full);
        let balanced_inner = Self::scratch_len(balanced_child, split_len, split_len);
        let infinity_child = Self::select_plan(high_len_a, high_len_b, TierCeiling::Full);
        let infinity_inner = Self::scratch_len(infinity_child, high_len_a, high_len_b);
        let inner_space = max(balanced_inner, infinity_inner);
        Toom32::local_scratch_len(split_len, inner_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-4 by 3
    // ---------------------------------------------------------------------------

    /// Workspace for one four-by-three level.
    ///
    /// Same two child shapes as the three-by-two split, and the same reason for
    /// maximising rather than assuming: the four guarded point products and `W(0)`
    /// are `split_len` square, while `W(inf)` is the two high parts, which are
    /// narrower, unequal, and can select a plan with a larger layout.
    pub fn toom43_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let widths = Widths::new(len_a, len_b);
        debug_assert!(
            widths.toom43_suitable(),
            "four-by-three scratch asked for a shape the tier cannot split"
        );
        let split_len = widths.larger.div_ceil(4);
        let high_len_a = widths.larger.saturating_sub(split_len.wrapping_mul(3));
        let high_len_b = widths.smaller.saturating_sub(split_len.wrapping_mul(2));

        let balanced_child = Self::select_plan(split_len, split_len, TierCeiling::Full);
        let balanced_inner = Self::scratch_len(balanced_child, split_len, split_len);
        let infinity_child = Self::select_plan(high_len_a, high_len_b, TierCeiling::Full);
        let infinity_inner = Self::scratch_len(infinity_child, high_len_a, high_len_b);
        let inner_space = max(balanced_inner, infinity_inner);
        Toom43::local_scratch_len(split_len, inner_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-6
    // ---------------------------------------------------------------------------

    pub fn toom6_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 6 || len_b < 6 {
            return Self::toom4_mul_scratch_len(len_a, len_b);
        }
        let Some(shape) = Widths::new(len_a, len_b).toom6_shape() else {
            // Mirror `toom6::mul`, which hands an unsuitable shape to
            // `recursive_mul` under a Toom-4 ceiling. Sizing it at the full ceiling
            // instead would size for whatever tier the *top level* would pick,
            // which since the blocked crossover moved to four-to-three is a
            // different plan with a different layout for every lopsided shape.
            let plan = Self::select_plan(len_a, len_b, TierCeiling::Toom4);
            return Self::scratch_len(plan, len_a, len_b);
        };
        if matches!(shape, MulShape::Half) {
            return Toom6::half_scratch_len(len_a, len_b);
        }
        let split_len = max(len_a, len_b).div_ceil(6);
        let eval_len = split_len.wrapping_add(1);
        let low_len_a = min(len_a, split_len);
        let low_len_b = min(len_b, split_len);
        // A power-of-two split needs no evaluation guard limb, so the widest
        // evaluated product is the split itself rather than split+1.
        let evaluation_len = if split_len.is_power_of_two() {
            split_len
        } else {
            eval_len
        };
        let plan_eval = Self::select_plan(evaluation_len, evaluation_len, TierCeiling::Toom4);
        let evaluation_inner = Self::scratch_len(plan_eval, evaluation_len, evaluation_len);
        let plan_low = Self::select_plan(low_len_a, low_len_b, TierCeiling::Toom4);
        let low_inner = Self::scratch_len(plan_low, low_len_a, low_len_b);
        let inner_space = max(evaluation_inner, low_inner);
        Toom6::local_scratch_len(split_len, inner_space)
    }

    pub fn toom6_sqr_scratch_len(len: usize) -> usize {
        if len < 6 {
            return Self::toom4_sqr_scratch_len(len);
        }
        let split_len = len.div_ceil(6);
        let eval_len = split_len.wrapping_add(1);
        let low_len = min(len, split_len);
        let evaluation_len = if split_len.is_power_of_two() {
            split_len
        } else {
            eval_len
        };
        let plan_eval = Self::select_square_plan(evaluation_len, TierCeiling::Toom4);
        let evaluation_inner = Self::square_scratch_len(plan_eval, evaluation_len);
        let plan_low = Self::select_square_plan(low_len, TierCeiling::Toom4);
        let low_inner = Self::square_scratch_len(plan_low, low_len);
        let inner_space = max(evaluation_inner, low_inner);
        Toom6::local_scratch_len(split_len, inner_space)
    }

    // ---------------------------------------------------------------------------
    // Toom-8
    // ---------------------------------------------------------------------------

    pub fn toom8_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let Some(shape) = Widths::new(len_a, len_b).toom8_shape() else {
            // Mirror `toom8::mul`, which hands an unsuitable shape to
            // `toom6::recursive_mul` under a Toom-6 ceiling. Sizing it at the full
            // ceiling would instead re-select a Toom-8 plan and re-enter here.
            let plan = Self::select_plan(len_a, len_b, TierCeiling::Toom6);
            return Self::scratch_len(plan, len_a, len_b);
        };
        let split_len = Toom8::multiplication_split_len(shape, len_a, len_b);
        Toom8::local_mul_scratch_len(shape, split_len, len_a, len_b)
    }

    pub fn toom8_sqr_scratch_len(len: usize) -> usize {
        if !Self::operand_has_eight_parts(len) {
            // See `toom8_mul_scratch_len`: match `toom6::recursive_sqr`'s ceiling
            // rather than re-selecting a Toom-8 plan.
            let plan = Self::select_square_plan(len, TierCeiling::Toom6);
            return Self::square_scratch_len(plan, len);
        }
        Toom8::local_sqr_scratch_len(len)
    }
}
