//! Execution: running exactly the algorithm a plan names.

#![expect(
    unsafe_code,
    reason = "Ordered nonempty operands establish positive blocked widths; valid limb-slice byte bounds admit square widths on every supported target"
)]

use core::num::NonZeroUsize;

use crate::parallel::{DefaultExecutor, ParallelExecutor, SequentialExecutor};

use super::{
    Karatsuba, Limb, LimbOutput, Lopsided, MulPlan, Multiplication, Schoolbook, SquarePlan, Toom3,
    Toom4, Toom6, Toom8, Toom32, Toom43,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa, TransformChoice};

impl Multiplication {
    /// Execute exactly the strategy described by `plan` with the default executor policy.
    ///
    /// The selector establishes shape validity before execution. Destinations
    /// and workspace must cover the selected plan's complete layout.
    #[inline]
    pub fn execute_plan(
        plan: MulPlan,
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        if plan != MulPlan::Lopsided && !plan.is_transform() {
            Self::execute_plan_with_executor(plan, dst, a, b, scratch, &SequentialExecutor);
            return;
        }
        DefaultExecutor::with_resolved(|executor| {
            Self::execute_plan_with_executor(plan, dst, a, b, scratch, executor);
        });
    }

    /// Execute a multiplication plan using a caller-selected executor.
    ///
    /// Lopsided and transform plans use the supplied executor. Conventional
    /// recursive products execute synchronously.
    ///
    /// Every conventional arm enters the named tier directly. Selection already
    /// establishes both operand crossovers for Karatsuba and Toom-3; recursive
    /// children retain their own dispatch and minimum-shape fallbacks.
    #[inline]
    pub fn execute_plan_with_executor<E: ParallelExecutor>(
        plan: MulPlan,
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        match plan {
            MulPlan::Schoolbook => Schoolbook::mul(dst, a, b),
            MulPlan::Lopsided => {
                let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                let Some(smaller_width) = NonZeroUsize::new(smaller.len()) else {
                    dst.fill(LimbOutput::from_limb(0));
                    return;
                };
                // SAFETY: ordering and smaller_width prove larger.len() >=
                // smaller.len() > 0 before block selection and execution.
                let larger_width = unsafe { NonZeroUsize::new_unchecked(larger.len()) };
                let block_len = Lopsided::block_len(larger_width, smaller_width);
                Lopsided::mul(dst, larger, smaller, scratch, block_len, executor);
            }
            MulPlan::Karatsuba => Karatsuba::mul(dst, a, b, scratch),
            MulPlan::Toom3 => Toom3::mul(dst, a, b, scratch),
            MulPlan::Toom32 => Toom32::mul(dst, a, b, scratch),
            MulPlan::Toom43 => Toom43::mul(dst, a, b, scratch),
            MulPlan::Toom4 => Toom4::mul(dst, a, b, scratch),
            MulPlan::Toom6 => Toom6::mul(dst, a, b, scratch),
            MulPlan::Toom8 => Toom8::mul(dst, a, b, scratch),
            #[cfg(not(target_pointer_width = "16"))]
            MulPlan::Large(LargePlan::Ssa) => {
                let computed = Ssa::try_mul_with_executor(
                    dst,
                    a,
                    b,
                    TransformChoice::PLANNED,
                    scratch,
                    executor,
                );
                debug_assert!(computed, "the validated SSA plan and scratch must execute");
            }
        }
    }

    /// Execute exactly the squaring strategy described by `plan` with the default executor policy.
    #[inline]
    pub fn execute_square_plan(
        plan: SquarePlan,
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
    ) {
        if !plan.is_transform() {
            Self::execute_square_plan_with_executor(plan, dst, a, scratch, &SequentialExecutor);
            return;
        }
        DefaultExecutor::with_resolved(|executor| {
            Self::execute_square_plan_with_executor(plan, dst, a, scratch, executor);
        });
    }

    /// Execute a squaring plan using a caller-selected executor.
    ///
    /// Karatsuba and Toom-3 enter their tier drivers because the selector has
    /// already applied their square crossovers. Recursive children retain
    /// guarded dispatch, and every tier retains its minimum-shape fallback.
    #[inline]
    pub fn execute_square_plan_with_executor<E: ParallelExecutor>(
        plan: SquarePlan,
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        #[cfg(target_pointer_width = "16")]
        let _ = executor;
        // Recursive Toom evaluators can provide fixed-width guard limbs above the
        // exact 2*n-limb square. Every tier overwrites the exact product; only the
        // disjoint guard suffix must be initialized here.
        // SAFETY: a valid limb slice occupies at most isize::MAX bytes, and
        // size_of::<Limb>() >= 2 on 16/32/64-bit targets. Thus 2*a.len()
        // <= isize::MAX < usize::MAX, independently of the selected tier.
        let square_len = unsafe { a.len().unchecked_mul(2) };
        if dst.len() > square_len {
            let (_, guard) = dst.split_at_mut(square_len);
            guard.fill(0);
        }

        match plan {
            SquarePlan::Schoolbook => Schoolbook::sqr(dst, a),
            SquarePlan::Karatsuba => Karatsuba::sqr(dst, a, scratch),
            SquarePlan::Toom3 => Toom3::sqr(dst, a, scratch),
            SquarePlan::Toom4 => Toom4::sqr(dst, a, scratch),
            SquarePlan::Toom6 => Toom6::sqr(dst, a, scratch),
            SquarePlan::Toom8 => Toom8::sqr(dst, a, scratch),
            #[cfg(not(target_pointer_width = "16"))]
            SquarePlan::Large(LargePlan::Ssa) => {
                let computed =
                    Ssa::try_sqr_with_executor(dst, a, TransformChoice::PLANNED, scratch, executor);
                debug_assert!(
                    computed,
                    "the validated SSA square plan and scratch must execute"
                );
            }
        }
    }
}
