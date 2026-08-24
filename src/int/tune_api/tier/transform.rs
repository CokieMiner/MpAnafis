//! Prepared SSA benchmark state plus transform inspection entry points.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use crate::parallel::{
    DefaultExecutor, FixedParallelismExecutor, ParallelExecutor, SequentialExecutor,
};

use super::{
    BenchValidation, Limb, Ssa, SsaMultiplicationPlan, TransformBench, TransformChoice, Tuner,
};

/// Executor selected for one transform benchmark row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransformExecutor {
    /// Never schedules work on another thread.
    Sequential,
    /// Uses the crate's feature-selected executor.
    Default,
}

/// SSA geometry policy selected before timing begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SsaGeometryPolicy {
    /// Force the transform while retaining the planner's geometry.
    Forced,
    /// Use the production transform/basecase and geometry decision.
    Production,
    /// Force one exact top-level transform exponent.
    ForcedExponent(u32),
}

/// Workspace ownership policy selected for one SSA row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SsaScratchPolicy {
    /// Let SSA allocate its transform workspace inside every measured call.
    Allocating,
    /// Allocate exact workspace during benchmark construction and reuse it.
    Reusable,
}

/// Shape-validated SSA state with explicit geometry, executor, and scratch policy.
#[derive(Debug)]
#[non_exhaustive]
pub struct SsaMultiplicationRunner<'operands> {
    plan: SsaMultiplicationPlan<'operands>,
    executor: TransformExecutor,
    parallelism: NonZeroUsize,
    scratch: Option<Vec<Limb>>,
}

/// Borrowed SSA call whose configuration and buffer shape were validated.
#[derive(Debug)]
#[non_exhaustive]
pub struct PreparedSsaMultiplication<'runner, 'operands, 'buffers> {
    runner: &'runner mut SsaMultiplicationRunner<'operands>,
    dst: &'buffers mut [Limb],
}

impl TransformExecutor {
    /// Returns the scheduling capacity recorded with the benchmark row.
    #[must_use]
    pub fn parallelism(self) -> NonZeroUsize {
        match self {
            Self::Sequential => SequentialExecutor.parallelism(),
            Self::Default => DefaultExecutor::with_resolved(|executor| executor.parallelism()),
        }
    }
}

impl SsaGeometryPolicy {
    const fn choice(self) -> TransformChoice {
        match self {
            Self::Forced => TransformChoice::FORCED,
            Self::Production => TransformChoice::PLANNED,
            Self::ForcedExponent(exponent) => TransformChoice::forced_at(exponent),
        }
    }
}

impl PreparedSsaMultiplication<'_, '_, '_> {
    /// Runs SSA without repeating facade validation or allocating reusable scratch.
    #[inline]
    pub fn run(&mut self) {
        self.runner.run_kernel(self.dst);
    }
}

impl<'operands> SsaMultiplicationRunner<'operands> {
    fn new(
        geometry: SsaGeometryPolicy,
        executor: TransformExecutor,
        scratch_policy: SsaScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<Self> {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "SSA operands must be nonzero-width"
        );
        assert!(
            Ssa::admits_mul(a.len(), b.len()),
            "SSA cannot represent the benchmark product width"
        );
        let parallelism = executor.parallelism();
        let plan = SsaMultiplicationPlan::try_new(a, b, geometry.choice(), parallelism)?;
        let scratch = match scratch_policy {
            SsaScratchPolicy::Allocating => None,
            SsaScratchPolicy::Reusable => Some(vec![Limb::MIN; plan.scratch_len()]),
        };
        Some(Self {
            plan,
            executor,
            parallelism,
            scratch,
        })
    }

    /// Prepares an exact borrowed SSA call for repeated measurements.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
    ) -> PreparedSsaMultiplication<'runner, 'operands, 'buffers> {
        assert_eq!(
            dst.len(),
            self.plan.destination_len(),
            "SSA destination width changed"
        );
        PreparedSsaMultiplication { runner: self, dst }
    }

    fn run_kernel(&mut self, dst: &mut [Limb]) {
        match self.executor {
            TransformExecutor::Sequential => {
                self.run_with_executor(dst, &SequentialExecutor);
            }
            TransformExecutor::Default => {
                let parallelism = self.parallelism;
                DefaultExecutor::with_resolved(|executor| {
                    let fixed = FixedParallelismExecutor::new(executor, parallelism);
                    self.run_with_executor(dst, &fixed);
                });
            }
        }
    }

    fn run_with_executor<E: ParallelExecutor>(&mut self, dst: &mut [Limb], executor: &E) {
        if let Some(scratch) = self.scratch.as_deref_mut() {
            // SAFETY: construction allocated the exact plan scratch and
            // `prepare` validated the exact destination width. The executor is
            // the same policy whose parallelism was recorded in the plan.
            unsafe { self.plan.run_with_scratch(dst, scratch, executor) }
        } else {
            // SAFETY: `prepare` validated the destination width and this is the
            // same executor policy used during plan construction.
            unsafe { self.plan.run_allocating(dst, executor) }
        }
    }
}

impl Tuner {
    /// Execute one production out-of-place Fermat twiddle shift.
    ///
    /// # Panics
    ///
    /// Panics unless both coefficient spans have the same supported width.
    pub fn bench_ssa_shift_from(dst: &mut [Limb], src: &[Limb], shift: usize) {
        assert_eq!(dst.len(), src.len(), "Fermat shift widths differ");
        assert!(
            TransformBench::ssa_shift_from(dst, src, shift),
            "Fermat shift benchmark width is invalid"
        );
    }

    /// Creates reusable state for one explicitly configured SSA benchmark row.
    ///
    /// Returns `None` only when a forced exponent is invalid for the shape.
    #[must_use]
    pub fn bench_ssa_multiplication<'operands>(
        geometry: SsaGeometryPolicy,
        executor: TransformExecutor,
        scratch: SsaScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<SsaMultiplicationRunner<'operands>> {
        SsaMultiplicationRunner::new(geometry, executor, scratch, a, b)
    }

    /// Execute the full-width Fermat-ring product used inside recursive SSA.
    ///
    /// # Panics
    ///
    /// Panics unless the operands and destination describe a supported equal-width
    /// Fermat-ring product.
    pub fn bench_ssa_fermat_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        assert_eq!(a.len(), b.len(), "Fermat benchmark widths differ");
        let completed = TransformBench::ssa_fermat_mul(dst, a, b, None);
        assert!(completed, "benchmark dimensions exceed the SSA size bounds");
    }

    /// Execute a full-width Fermat product with a forced transform geometry.
    ///
    /// # Panics
    ///
    /// Panics unless the widths and forced geometry are valid.
    pub fn bench_ssa_fermat_mul_forced_plan(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        transform_exponent: u32,
    ) {
        assert_eq!(a.len(), b.len(), "Fermat benchmark widths differ");
        let completed = TransformBench::ssa_fermat_mul(dst, a, b, Some(transform_exponent));
        assert!(completed, "forced Fermat geometry exceeds its ring bounds");
    }

    /// Return exact scratch for a Mersenne-ring product.
    #[must_use]
    pub fn bench_ssa_mersenne_mul_scratch_len(len: usize) -> usize {
        TransformBench::ssa_mul_mod_bnm1_scratch_len(len)
    }

    /// Execute the recursive Mersenne-ring half of SSA with caller-owned scratch.
    ///
    /// # Panics
    ///
    /// Panics unless all data spans have equal width and scratch is sufficient.
    pub fn bench_ssa_mersenne_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        assert_eq!(a.len(), b.len(), "Mersenne benchmark widths differ");
        assert_eq!(dst.len(), a.len(), "Mersenne destination width differs");
        BenchValidation::scratch(
            scratch,
            TransformBench::ssa_mul_mod_bnm1_scratch_len(a.len()),
        );
        TransformBench::ssa_mul_mod_bnm1(dst, a, b, scratch);
    }
}
