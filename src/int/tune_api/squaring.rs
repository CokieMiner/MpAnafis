//! Reusable forced-tier squaring tuner.

use crate::parallel::DefaultExecutor;

use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Ssa, SsaSquaringPlan, Toom3, Toom4,
    Toom6, Toom8, TransformChoice,
};

/// Root squaring tier measured by [`Tuner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SquaringAlgorithm {
    /// Quadratic schoolbook squaring.
    Schoolbook,
    /// One forced Karatsuba square level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 square level with normal child dispatch.
    ToomCook3,
    /// One Toom-Cook 4 square level with normal child dispatch.
    ToomCook4,
    /// One Toom-Cook 6 square level with normal child dispatch.
    ToomCook6,
    /// One Toom-Cook 8/8.5 square level with normal child dispatch.
    ToomCook85,
    /// Schonhage-Strassen squaring with a forced transform geometry.
    #[cfg(not(target_pointer_width = "16"))]
    SsaForced,
    /// Schonhage-Strassen squaring using production planning.
    #[cfg(not(target_pointer_width = "16"))]
    SsaProduction,
}

/// Borrowed, shape-validated squaring call used inside timed loops.
#[derive(Debug)]
pub struct PreparedSquaring<'runner, 'buffers> {
    runner: &'runner mut SquaringRunner,
    dst: &'buffers mut [Limb],
    a: &'buffers [Limb],
    kernel: PreparedSquareKernel<'buffers>,
}

#[derive(Debug)]
enum PreparedSquareKernel<'buffers> {
    Schoolbook,
    Karatsuba,
    ToomCook3,
    ToomCook4,
    ToomCook6,
    ToomCook85,
    #[cfg(not(target_pointer_width = "16"))]
    Ssa(SsaSquaringPlan<'buffers>),
}

impl PreparedSquaring<'_, '_> {
    /// Runs the validated square without repeating shape checks.
    pub fn run(&mut self) {
        // SAFETY: `prepare` validated the exact spans, geometry, and scratch
        // ownership represented by `kernel`.
        unsafe {
            self.runner.run_kernel(self.dst, self.a, &self.kernel);
        }
    }
}

/// Allocation-free reusable state for one squaring crossover sample.
#[derive(Debug)]
pub struct SquaringRunner {
    algorithm: SquaringAlgorithm,
    len: usize,
    destination_len: usize,
    scratch: ScratchBuffer,
}

impl SquaringRunner {
    /// Pre-allocates the exact scratch required for `algorithm` at this width.
    ///
    /// # Panics
    ///
    /// Panics if the operand width is zero or SSA cannot represent the
    /// requested square width.
    #[must_use]
    pub(crate) fn new(algorithm: SquaringAlgorithm, len: usize) -> Self {
        assert!(len != 0, "squaring tuner operand must be nonzero-width");
        let destination_len = len
            .checked_mul(2)
            .expect("squaring tuner product width overflows usize");
        #[cfg(not(target_pointer_width = "16"))]
        if matches!(
            algorithm,
            SquaringAlgorithm::SsaForced | SquaringAlgorithm::SsaProduction
        ) {
            assert!(
                Ssa::admits_sqr(len),
                "SSA cannot represent the requested tuning width"
            );
        }
        let scratch_len = match algorithm {
            SquaringAlgorithm::Schoolbook => 0,
            SquaringAlgorithm::Karatsuba => Multiplication::karatsuba_sqr_forced_scratch_len(len),
            SquaringAlgorithm::ToomCook3 => Multiplication::toom3_sqr_forced_scratch_len(len),
            SquaringAlgorithm::ToomCook4 => Multiplication::toom4_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook6 => Multiplication::toom6_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook85 => Multiplication::toom8_sqr_scratch_len(len),
            #[cfg(not(target_pointer_width = "16"))]
            SquaringAlgorithm::SsaForced | SquaringAlgorithm::SsaProduction => {
                Ssa::sqr_scratch_len(len)
            }
        };
        let mut scratch = ScratchBuffer::acquire(scratch_len);
        // SAFETY: forced tiers write or clear every scratch region before
        // reading it, and their sizing functions include child workspaces.
        unsafe {
            scratch.set_len(scratch_len);
        }
        Self {
            algorithm,
            len,
            destination_len,
            scratch,
        }
    }

    /// Prepares an exact borrowed call for repeated allocation-free runs.
    ///
    /// # Panics
    ///
    /// Panics if the operand or destination width differs from construction.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
        a: &'buffers [Limb],
    ) -> PreparedSquaring<'runner, 'buffers> {
        assert_eq!(a.len(), self.len, "tuner operand width changed");
        assert_eq!(
            dst.len(),
            self.destination_len,
            "tuner destination width changed"
        );
        let kernel = match self.algorithm {
            SquaringAlgorithm::Schoolbook => PreparedSquareKernel::Schoolbook,
            SquaringAlgorithm::Karatsuba => PreparedSquareKernel::Karatsuba,
            SquaringAlgorithm::ToomCook3 => PreparedSquareKernel::ToomCook3,
            SquaringAlgorithm::ToomCook4 => PreparedSquareKernel::ToomCook4,
            SquaringAlgorithm::ToomCook6 => PreparedSquareKernel::ToomCook6,
            SquaringAlgorithm::ToomCook85 => PreparedSquareKernel::ToomCook85,
            #[cfg(not(target_pointer_width = "16"))]
            SquaringAlgorithm::SsaForced | SquaringAlgorithm::SsaProduction => {
                let choice = if matches!(self.algorithm, SquaringAlgorithm::SsaForced) {
                    TransformChoice::FORCED
                } else {
                    TransformChoice::PLANNED
                };
                let plan = SsaSquaringPlan::try_new(a, choice, 1)
                    .expect("validated SSA tuning shape must produce a square plan");
                assert_eq!(
                    plan.destination_len(),
                    dst.len(),
                    "SSA square plan destination differs from validated width"
                );
                assert!(
                    self.scratch.len() >= plan.scratch_len(),
                    "SSA tuning scratch is smaller than its operand-bound plan"
                );
                PreparedSquareKernel::Ssa(plan)
            }
        };
        PreparedSquaring {
            runner: self,
            dst,
            a,
            kernel,
        }
    }

    /// Squares with the configured root tier without caller-side allocation.
    ///
    /// Repeated measurements should retain the object returned by
    /// [`Self::prepare`] instead.
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb]) {
        let mut prepared = self.prepare(dst, a);
        prepared.run();
    }

    /// Executes the already validated root tier.
    unsafe fn run_kernel(
        &mut self,
        dst: &mut [Limb],
        a: &[Limb],
        kernel: &PreparedSquareKernel<'_>,
    ) {
        dst.fill(0);
        match kernel {
            PreparedSquareKernel::Schoolbook => Schoolbook::sqr(dst, a),
            PreparedSquareKernel::Karatsuba => Karatsuba::sqr_forced(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook3 => {
                Toom3::sqr_forced(dst, a, &mut self.scratch);
            }
            PreparedSquareKernel::ToomCook4 => Toom4::sqr(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook6 => Toom6::sqr(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook85 => Toom8::sqr(dst, a, &mut self.scratch),
            #[cfg(not(target_pointer_width = "16"))]
            PreparedSquareKernel::Ssa(plan) => {
                let executor = DefaultExecutor::default();
                // SAFETY: `prepare` validated the exact destination, plan, and
                // reusable scratch capacity; the executor width is fixed at one.
                unsafe {
                    plan.run_with_scratch(dst, &mut self.scratch, &executor);
                }
            }
        }
    }
}
