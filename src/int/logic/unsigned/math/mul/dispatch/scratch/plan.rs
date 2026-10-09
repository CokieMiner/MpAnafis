//! Workspace dispatch for a selected product or square plan.

#![expect(
    unsafe_code,
    reason = "Ordered widths and a positive smaller operand establish a positive larger width"
)]

use core::num::NonZeroUsize;

use crate::parallel::{DefaultExecutor, ParallelExecutor};

#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa};
use super::{Lopsided, MulPlan, Multiplication, SquarePlan, Widths};

impl Multiplication {
    /// Returns the selected product plan's workspace in limbs.
    #[inline]
    pub fn scratch_len(plan: MulPlan, len_a: usize, len_b: usize) -> usize {
        if plan != MulPlan::Lopsided && !plan.is_transform() {
            return Self::scratch_len_for_parallelism(plan, len_a, len_b, 1);
        }
        DefaultExecutor::with_resolved(|executor| {
            Self::scratch_len_for_parallelism(plan, len_a, len_b, executor.parallelism().get())
        })
    }

    /// Returns product workspace for an explicit worker budget.
    #[inline]
    pub fn scratch_len_for_parallelism(
        plan: MulPlan,
        len_a: usize,
        len_b: usize,
        parallelism: usize,
    ) -> usize {
        match plan {
            MulPlan::Schoolbook => 0,
            MulPlan::Lopsided => {
                let widths = Widths::new(len_a, len_b);
                let Some(smaller_width) = NonZeroUsize::new(widths.smaller) else {
                    return 0;
                };
                // SAFETY: Widths orders larger >= smaller, and the boundary
                // establishes smaller > 0, including virtual sizing widths.
                let larger_width = unsafe { NonZeroUsize::new_unchecked(widths.larger) };
                Lopsided::mul_scratch_len(
                    len_a,
                    len_b,
                    Lopsided::block_len(larger_width, smaller_width),
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

    /// Returns the selected square plan's workspace in limbs.
    #[inline]
    pub fn square_scratch_len(plan: SquarePlan, len: usize) -> usize {
        if !plan.is_transform() {
            return Self::square_scratch_len_for_parallelism(plan, len, 1);
        }
        DefaultExecutor::with_resolved(|executor| {
            Self::square_scratch_len_for_parallelism(plan, len, executor.parallelism().get())
        })
    }

    /// Returns square workspace for an explicit worker budget.
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
