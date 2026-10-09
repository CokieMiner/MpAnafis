//! Coefficient strategies bound to their ring before convolution execution.

#![expect(
    unsafe_code,
    reason = "Positive aligned rings reserve nonzero basecase, factorized, or retained-transform coefficient workspaces"
)]

#[cfg(feature = "std")]
use core::cell::RefCell;
use core::num::NonZeroUsize;
#[cfg(feature = "std")]
use std::thread_local;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use alloc::sync::Arc;

#[cfg(feature = "std")]
use super::RetainedPlanCache;
use super::{
    FftPlan, MulPlan, MulTransformPlan, Multiplication, NegacyclicPlan, SSA_BASE_MODULUS_BITS,
    SquarePlan, SquareTransformPlan, SsaRing, TierCeiling,
};

#[cfg(feature = "std")]
type RetainedPlan<T> = Arc<T>;
#[cfg(not(feature = "std"))]
type RetainedPlan<T> = Box<T>;

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static MUL_PLANS: RefCell<RetainedPlanCache<RetainedPlan<MulTransformPlan>>> =
        const { RefCell::new(RetainedPlanCache::new()) };
    static SQUARE_PLANS: RefCell<RetainedPlanCache<RetainedPlan<SquareTransformPlan>>> =
        const { RefCell::new(RetainedPlanCache::new()) };
}

// OS-key TLS initializes the same retained coefficient trees lazily.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static MUL_PLANS: RefCell<RetainedPlanCache<RetainedPlan<MulTransformPlan>>> =
        RefCell::from(RetainedPlanCache::new());
    static SQUARE_PLANS: RefCell<RetainedPlanCache<RetainedPlan<SquareTransformPlan>>> =
        RefCell::from(RetainedPlanCache::new());
}

/// Ring-bound strategy accepted by both prepared and ordinary pointwise calls.
#[derive(Clone, Debug)]
pub struct PointwiseMulPlan {
    pub bits: usize,
    pub strategy: PointwiseMulStrategy,
    pub scratch_len: NonZeroUsize,
}

/// Mutually exclusive coefficient multiplication strategies.
#[derive(Clone, Debug)]
pub enum PointwiseMulStrategy {
    Basecase(MulPlan),
    Negacyclic(NegacyclicPlan),
    Transform(RetainedPlan<MulTransformPlan>),
}

/// Ring-bound square strategy.
#[derive(Clone, Debug)]
pub struct PointwiseSquarePlan {
    pub bits: usize,
    pub strategy: PointwiseSquareStrategy,
    pub scratch_len: NonZeroUsize,
}

/// Mutually exclusive coefficient square strategies.
#[derive(Clone, Debug)]
#[expect(
    variant_size_differences,
    reason = "the retained transform handle avoids constructing plans during coefficient execution"
)]
pub enum PointwiseSquareStrategy {
    Basecase(SquarePlan),
    Transform(RetainedPlan<SquareTransformPlan>),
}

impl From<usize> for PointwiseMulPlan {
    /// Selects a coefficient strategy when the caller supplies a bare ring width.
    fn from(bits: usize) -> Self {
        let strategy = if bits > SSA_BASE_MODULUS_BITS {
            PointwiseMulStrategy::Transform(mul_transform(bits))
        } else {
            let limbs = SsaRing::mod_limbs(bits);
            NegacyclicPlan::select_factor(limbs)
                .and_then(|factor| NegacyclicPlan::for_factor(limbs, factor))
                .map_or_else(
                    || {
                        PointwiseMulStrategy::Basecase(Multiplication::select_plan(
                            limbs,
                            limbs,
                            TierCeiling::Full,
                        ))
                    },
                    PointwiseMulStrategy::Negacyclic,
                )
        };
        let workspace_len = match &strategy {
            PointwiseMulStrategy::Basecase(plan) => {
                let limbs = SsaRing::mod_limbs(bits);
                limbs
                    .saturating_mul(2)
                    .saturating_add(Multiplication::scratch_len(*plan, limbs, limbs))
            }
            PointwiseMulStrategy::Negacyclic(plan) => SsaRing::coeff_limbs(bits)
                .get()
                .saturating_add(plan.scratch_len),
            PointwiseMulStrategy::Transform(plan) => plan.transform_mul_scratch(1),
        };
        // SAFETY: every caller supplies a positive limb-aligned ring. Basecase
        // scratch includes 2*mod_limbs>=2; negacyclic scratch includes the
        // positive coefficient width; a transform includes complete matrices.
        // Saturation produces usize::MAX, which is also positive.
        let scratch_len = unsafe { NonZeroUsize::new_unchecked(workspace_len) };
        Self {
            bits,
            strategy,
            scratch_len,
        }
    }
}

impl From<usize> for PointwiseSquarePlan {
    /// Selects a square strategy when the caller supplies a bare ring width.
    fn from(bits: usize) -> Self {
        let strategy = if bits > SSA_BASE_MODULUS_BITS {
            PointwiseSquareStrategy::Transform(square_transform(bits))
        } else {
            PointwiseSquareStrategy::Basecase(Multiplication::select_square_plan(
                SsaRing::mod_limbs(bits),
                TierCeiling::Full,
            ))
        };
        let workspace_len = match &strategy {
            PointwiseSquareStrategy::Basecase(plan) => {
                let limbs = SsaRing::mod_limbs(bits);
                limbs
                    .saturating_mul(2)
                    .saturating_add(Multiplication::square_scratch_len(*plan, limbs))
            }
            PointwiseSquareStrategy::Transform(plan) => plan.transform_sqr_scratch(1),
        };
        // SAFETY: the admitted ring has mod_limbs>=1. Basecase scratch reserves
        // 2*mod_limbs>=2; transform scratch reserves a nonempty complete matrix.
        // The overflow sentinel usize::MAX remains positive.
        let scratch_len = unsafe { NonZeroUsize::new_unchecked(workspace_len) };
        Self {
            bits,
            strategy,
            scratch_len,
        }
    }
}

/// Coefficient descendants always use the multiplication objective and one worker.
#[cfg_attr(
    not(feature = "std"),
    allow(
        clippy::unnecessary_box_returns,
        reason = "boxed ownership breaks the recursive mul-plan layout when no thread-local cache is available"
    )
)]
fn mul_transform(bits: usize) -> RetainedPlan<MulTransformPlan> {
    #[cfg(feature = "std")]
    if let Some(plan) = MUL_PLANS.with(|cache| cache.borrow().get(bits)) {
        return plan;
    }
    let plan = RetainedPlan::new(MulTransformPlan::new(FftPlan::new(bits)));
    #[cfg(feature = "std")]
    MUL_PLANS.with(|cache| cache.borrow_mut().insert(bits, RetainedPlan::clone(&plan)));
    plan
}

/// Square descendants retain their own geometry and immutable coefficient tree.
#[cfg_attr(
    not(feature = "std"),
    allow(
        clippy::unnecessary_box_returns,
        reason = "boxed ownership breaks the recursive square-plan layout when no thread-local cache is available"
    )
)]
fn square_transform(bits: usize) -> RetainedPlan<SquareTransformPlan> {
    #[cfg(feature = "std")]
    if let Some(plan) = SQUARE_PLANS.with(|cache| cache.borrow().get(bits)) {
        return plan;
    }
    let plan = RetainedPlan::new(SquareTransformPlan::new(FftPlan::new_for_square(bits)));
    #[cfg(feature = "std")]
    SQUARE_PLANS.with(|cache| cache.borrow_mut().insert(bits, RetainedPlan::clone(&plan)));
    plan
}
