//! Prepared Mersenne recursion with retained Fermat and lower-tower plans.

#[cfg(feature = "std")]
use core::cell::RefCell;
use core::ops::Deref;
#[cfg(feature = "std")]
use std::thread_local;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use alloc::sync::Arc;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use super::RetainedPlanCache;
use super::{
    FftPlan, LIMB_BITS, MulPlan, MulTransformPlan, Multiplication, SSA_BNM1_BASECASE_LIMBS,
    SquarePlan, SquareTransformPlan, TierCeiling,
};

#[cfg(feature = "std")]
type RetainedLevels<T> = Arc<Vec<T>>;
#[cfg(not(feature = "std"))]
type RetainedLevels<T> = Box<[T]>;

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static MUL_PLANS: RefCell<RetainedPlanCache<CrtMulPlan>> =
        const { RefCell::new(RetainedPlanCache::new()) };
    static SQUARE_PLANS: RefCell<RetainedPlanCache<CrtSquarePlan>> =
        const { RefCell::new(RetainedPlanCache::new()) };
}

// OS-key TLS initializes the same retained-plan cache lazily.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static MUL_PLANS: RefCell<RetainedPlanCache<CrtMulPlan>> =
        RefCell::from(RetainedPlanCache::new());
    static SQUARE_PLANS: RefCell<RetainedPlanCache<CrtSquarePlan>> =
        RefCell::from(RetainedPlanCache::new());
}

/// Exactly one executable stage of a Mersenne product.
#[derive(Clone, Debug)]
pub enum CrtMulLevel {
    Basecase { plan: MulPlan, work: usize },
    Split { ring: MulTransformPlan },
}

/// Exactly one executable stage of a Mersenne square.
#[derive(Clone, Debug)]
pub enum CrtSquareLevel {
    Basecase { plan: SquarePlan, work: usize },
    Split { ring: SquareTransformPlan },
}

/// Immutable product recursion with sequential lower-tower leaves.
#[derive(Clone, Debug)]
pub struct CrtMulPlan {
    levels: RetainedLevels<CrtMulLevel>,
}

impl Deref for CrtMulPlan {
    type Target = [CrtMulLevel];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.levels
    }
}

/// Immutable square recursion with sequential lower-tower leaves.
#[derive(Clone, Debug)]
pub struct CrtSquarePlan {
    levels: RetainedLevels<CrtSquareLevel>,
}

impl Deref for CrtSquarePlan {
    type Target = [CrtSquareLevel];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.levels
    }
}

impl CrtMulPlan {
    /// Checks the complete halving geometry and retains every descendant without dynamic recursion.
    pub fn new(n: usize) -> Option<Self> {
        #[cfg(feature = "std")]
        if let Some(plan) = MUL_PLANS.with(|cache| cache.borrow().get(n)) {
            return Some(plan);
        }
        let mut levels = Vec::with_capacity(level_count(n)?);
        let mut current_n = n;

        while current_n > SSA_BNM1_BASECASE_LIMBS {
            let half = current_n >> 1;
            let bits = half.checked_mul(LIMB_BITS)?;
            let ring = MulTransformPlan::new(FftPlan::new(bits));
            levels.push(CrtMulLevel::Split { ring });
            current_n = half;
        }

        let base_plan = Multiplication::select_plan(current_n, current_n, TierCeiling::Full);
        let base_work =
            Multiplication::scratch_len_for_parallelism(base_plan, current_n, current_n, 1);
        levels.push(CrtMulLevel::Basecase {
            plan: base_plan,
            work: base_work,
        });

        #[cfg(feature = "std")]
        let retained_levels = Arc::new(levels);
        #[cfg(not(feature = "std"))]
        let retained_levels = levels.into_boxed_slice();
        let plan = Self {
            levels: retained_levels,
        };
        #[cfg(feature = "std")]
        MUL_PLANS.with(|cache| cache.borrow_mut().insert(n, plan.clone()));
        Some(plan)
    }
}

impl CrtSquarePlan {
    /// Checks square recursion and retains square-specific descendants without dynamic recursion.
    pub fn new(n: usize) -> Option<Self> {
        #[cfg(feature = "std")]
        if let Some(plan) = SQUARE_PLANS.with(|cache| cache.borrow().get(n)) {
            return Some(plan);
        }
        let mut levels = Vec::with_capacity(level_count(n)?);
        let mut current_n = n;

        while current_n > SSA_BNM1_BASECASE_LIMBS {
            let half = current_n >> 1;
            let bits = half.checked_mul(LIMB_BITS)?;
            let ring = SquareTransformPlan::new(FftPlan::new_for_square(bits));
            levels.push(CrtSquareLevel::Split { ring });
            current_n = half;
        }

        let base_plan = Multiplication::select_square_plan(current_n, TierCeiling::Full);
        let base_work = Multiplication::square_scratch_len_for_parallelism(base_plan, current_n, 1);
        levels.push(CrtSquareLevel::Basecase {
            plan: base_plan,
            work: base_work,
        });

        #[cfg(feature = "std")]
        let retained_levels = Arc::new(levels);
        #[cfg(not(feature = "std"))]
        let retained_levels = levels.into_boxed_slice();
        let plan = Self {
            levels: retained_levels,
        };
        #[cfg(feature = "std")]
        SQUARE_PLANS.with(|cache| cache.borrow_mut().insert(n, plan.clone()));
        Some(plan)
    }
}

/// Validates all halvings before allocating the exact immutable level list.
fn level_count(mut width: usize) -> Option<usize> {
    if width == 0 {
        return None;
    }
    let mut count = 1_usize;
    while width > SSA_BNM1_BASECASE_LIMBS {
        if !width.is_multiple_of(2) {
            return None;
        }
        width >>= 1;
        count = count.checked_add(1)?;
    }
    Some(count)
}
