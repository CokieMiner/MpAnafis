//! Execution: running exactly the algorithm a plan names.

use core::cmp::{max, min};

use crate::parallel::{DefaultExecutor, ParallelExecutor};

use super::{
    Karatsuba, Limb, Lopsided, MulPlan, Multiplication, Schoolbook, SquarePlan, Toom3, Toom4,
    Toom6, Toom8, Toom32, Toom43,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa, TransformChoice};

/// Execute exactly the strategy described by `plan` with the default executor policy.
///
/// Total: every variant names one algorithm and runs it. The transform arms
/// carry no conventional fallback because `select_mul_plan` only names them
/// after `ssa_admits_mul` established the construction exists
/// for these widths. `dispatch::tests` sweeps those predicates against the
/// entry points they guard.
impl Multiplication {
    #[inline]
    pub fn execute_plan(
        plan: MulPlan,
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        let executor = DefaultExecutor::default();
        Self::execute_plan_with_executor(plan, dst, a, b, scratch, &executor);
    }

    /// Execute a multiplication plan using a caller-selected executor.
    ///
    /// Non-transform tiers remain on their existing synchronous kernels. Large
    /// SSA tiers receive `executor` all the way through their transform
    /// orchestration, so applications can reuse an existing work-stealing pool.
    #[inline]
    pub fn execute_plan_with_executor<E: ParallelExecutor>(
        plan: MulPlan,
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        match plan {
            MulPlan::Schoolbook => Schoolbook::mul(dst, a, b),
            MulPlan::Lopsided => Lopsided::mul(dst, a, b, scratch, executor),
            MulPlan::Karatsuba => {
                // Rectangular Karatsuba reconstructs into fixed-width output and
                // therefore requires a zero base. Equal-width paths overwrite it.
                if a.len() != b.len() {
                    let larger = max(a.len(), b.len());
                    let smaller = min(a.len(), b.len());
                    let split = larger.div_ceil(2);
                    if smaller > split {
                        let high_written = a
                            .len()
                            .saturating_sub(split)
                            .wrapping_add(b.len().saturating_sub(split));
                        let zero_start = split.wrapping_mul(2).wrapping_add(high_written);
                        if zero_start < dst.len()
                            && let Some(fill_span) = dst.get_mut(zero_start..)
                        {
                            fill_span.fill(0);
                        }
                    }
                }
                Karatsuba::mul(dst, a, b, scratch);
            }
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
                    Some(scratch),
                    executor,
                );
                debug_assert!(computed, "the planner named SSA for operands it declines");
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
        let executor = DefaultExecutor::default();
        Self::execute_square_plan_with_executor(plan, dst, a, scratch, &executor);
    }

    /// Execute a squaring plan using a caller-selected executor.
    #[inline]
    pub fn execute_square_plan_with_executor<E: ParallelExecutor>(
        plan: SquarePlan,
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        // Recursive Toom evaluators can provide fixed-width guard limbs above the
        // exact 2*n-limb square. Every tier overwrites the exact product; only the
        // disjoint guard suffix must be initialized here.
        let square_len = a.len().wrapping_mul(2);
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
                let computed = Ssa::try_sqr_with_executor(
                    dst,
                    a,
                    TransformChoice::PLANNED,
                    Some(scratch),
                    executor,
                );
                debug_assert!(computed, "the planner named SSA for an operand it declines");
            }
        }
    }
}
