//! Forced squaring runners with shape-bound calls and reusable scratch.

#[cfg(not(target_pointer_width = "16"))]
use core::num::NonZeroUsize;

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use super::ParallelExecutor;
#[cfg(not(target_pointer_width = "16"))]
use super::{DefaultExecutor, Ssa, SsaSquaringPlan, TransformChoice};
use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Toom3, Toom4, Toom6, Toom8,
};

/// Root squaring tier measured by [`SquaringRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SquaringAlgorithm {
    /// Quadratic schoolbook squaring.
    Schoolbook,
    /// One forced Karatsuba square level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 square level with normal child dispatch.
    ToomCook3,
    /// One forced Toom-Cook 4 square level with normal child dispatch.
    ToomCook4,
    /// One forced Toom-Cook 6 square level with normal child dispatch.
    ToomCook6,
    /// One forced Toom-Cook 8/8.5 square level with normal child dispatch.
    ToomCook85,
    /// Schönhage-Strassen squaring with a forced transform geometry.
    #[cfg(not(target_pointer_width = "16"))]
    SsaForced,
    /// Single full-width Fermat square, measured independently of CRT squares.
    #[cfg(not(target_pointer_width = "16"))]
    SsaDirectFermat,
    /// Schönhage-Strassen squaring using production planning.
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

#[cfg(target_pointer_width = "16")]
type PreparedSquareKernel<'buffers> = SquaringAlgorithm;

#[cfg(not(target_pointer_width = "16"))]
#[derive(Debug)]
enum PreparedSquareKernel<'buffers> {
    Schoolbook,
    Karatsuba,
    ToomCook3,
    ToomCook4,
    ToomCook6,
    ToomCook85,
    Ssa(SsaSquaringPlan<'buffers>),
    SsaProduction(TransformChoice),
}

/// Retained algorithm, operand width, and scratch for squaring comparisons.
///
/// Forced tiers prepare outside execution. Production SSA includes planning in
/// each run and may grow scratch when its selected geometry requires it.
#[derive(Debug)]
pub struct SquaringRunner {
    algorithm: SquaringAlgorithm,
    len: usize,
    destination_len: usize,
    #[cfg(not(target_pointer_width = "16"))]
    executor_parallelism: NonZeroUsize,
    scratch: ScratchBuffer,
}

impl SquaringRunner {
    /// Allocates the selected algorithm's scratch for the operand width.
    ///
    /// SSA may grow strategy-specific scratch during preparation or execution.
    ///
    /// # Panics
    ///
    /// Panics if the operand width is zero, doubling it overflows `usize`, or
    /// SSA cannot represent the requested square width.
    #[must_use]
    pub fn new(algorithm: SquaringAlgorithm, len: usize) -> Self {
        assert!(len != 0, "squaring tuner operand must be nonzero-width");
        let destination_len = len
            .checked_mul(2)
            .expect("squaring tuner product width overflows usize");
        #[cfg(not(target_pointer_width = "16"))]
        if matches!(
            algorithm,
            SquaringAlgorithm::SsaForced
                | SquaringAlgorithm::SsaDirectFermat
                | SquaringAlgorithm::SsaProduction
        ) {
            assert!(
                Ssa::admits_sqr(len),
                "SSA cannot represent the requested tuning width"
            );
        }
        #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
        let executor_parallelism = match algorithm {
            SquaringAlgorithm::SsaForced
            | SquaringAlgorithm::SsaDirectFermat
            | SquaringAlgorithm::SsaProduction => {
                DefaultExecutor::with_resolved(|executor| executor.parallelism())
            }
            SquaringAlgorithm::Schoolbook
            | SquaringAlgorithm::Karatsuba
            | SquaringAlgorithm::ToomCook3
            | SquaringAlgorithm::ToomCook4
            | SquaringAlgorithm::ToomCook6
            | SquaringAlgorithm::ToomCook85 => NonZeroUsize::MIN,
        };
        #[cfg(all(not(feature = "rayon"), not(target_pointer_width = "16")))]
        let executor_parallelism = NonZeroUsize::MIN;
        let scratch_len = match algorithm {
            SquaringAlgorithm::Schoolbook => 0,
            SquaringAlgorithm::Karatsuba => Multiplication::karatsuba_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook3 => Multiplication::toom3_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook4 => Multiplication::toom4_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook6 => Multiplication::toom6_sqr_scratch_len(len),
            SquaringAlgorithm::ToomCook85 => Multiplication::toom8_sqr_scratch_len(len),
            #[cfg(not(target_pointer_width = "16"))]
            SquaringAlgorithm::SsaForced
            | SquaringAlgorithm::SsaDirectFermat
            | SquaringAlgorithm::SsaProduction => {
                Ssa::sqr_scratch_len_for_parallelism(len, executor_parallelism.get())
            }
        };
        let mut scratch = ScratchBuffer::acquire(scratch_len);
        // A limb slice requires initialized elements even when a kernel
        // overwrites them. Initialization occurs before repeated execution.
        scratch.resize(scratch_len, 0);
        Self {
            algorithm,
            len,
            destination_len,
            #[cfg(not(target_pointer_width = "16"))]
            executor_parallelism,
            scratch,
        }
    }

    /// Prepares an exact borrowed call for repeated runs with reusable scratch.
    ///
    /// # Panics
    ///
    /// Panics if an operand or destination width differs from construction, or
    /// an SSA plan cannot be prepared from the supplied operand.
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
            SquaringAlgorithm::SsaProduction => {
                PreparedSquareKernel::SsaProduction(TransformChoice::PLANNED)
            }
            #[cfg(not(target_pointer_width = "16"))]
            SquaringAlgorithm::SsaForced | SquaringAlgorithm::SsaDirectFermat => {
                let choice = if matches!(self.algorithm, SquaringAlgorithm::SsaDirectFermat) {
                    TransformChoice::FORCED_DIRECT_FERMAT
                } else {
                    TransformChoice::FORCED
                };
                let maybe_plan =
                    SsaSquaringPlan::try_new(a, choice, self.executor_parallelism.get());
                let plan =
                    maybe_plan.expect("validated SSA tuning shape must produce a square plan");
                debug_assert_eq!(
                    plan.result_len,
                    dst.len(),
                    "SSA square plan destination differs from validated width"
                );
                if self.scratch.len() < plan.scratch_len {
                    self.scratch.reset_with_capacity(plan.scratch_len);
                    self.scratch.resize(plan.scratch_len, 0);
                }
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

    /// Prepares and executes one squaring call.
    ///
    /// Repeated measurements should retain the object returned by
    /// [`Self::prepare`] instead.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::prepare`].
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb]) {
        let mut prepared = self.prepare(dst, a);
        prepared.run();
    }
}

impl PreparedSquaring<'_, '_> {
    /// Executes the prepared square with retained operands and scratch.
    #[expect(
        unsafe_code,
        reason = "Preparation binds an exact square width and sufficient scratch to immutable input and exclusive output borrows"
    )]
    #[inline]
    pub fn run(&mut self) {
        // SAFETY: prepare proves dst.len() = 2 * a.len() and sizes the selected
        // kernel's scratch. The shared input borrow and exclusive destination
        // and runner borrows preserve disjoint storage, widths, and plan
        // lifetimes throughout repeated execution.
        unsafe {
            self.runner.run_kernel(self.dst, self.a, &self.kernel);
        }
    }
}

impl SquaringRunner {
    /// Executes the already validated root tier.
    ///
    /// # Safety
    ///
    /// `a` must be nonempty, `dst.len()` must equal `2 * a.len()`, and `dst`
    /// must not overlap `a` or scratch. The kernel must belong to the borrowed
    /// call produced by `prepare`, with its validated scratch and worker budget.
    #[expect(
        unsafe_code,
        reason = "Prepared calls establish the buffer, scratch, and executor contracts required by SSA execution"
    )]
    unsafe fn run_kernel(
        &mut self,
        dst: &mut [Limb],
        a: &[Limb],
        kernel: &PreparedSquareKernel<'_>,
    ) {
        // The validated nonempty operand has an exact 2*n-limb destination.
        // Basecase writes every limb; recursive tiers initialize endpoints and
        // gaps before reconstruction; SSA writes its result and zero padding.
        match kernel {
            PreparedSquareKernel::Schoolbook => Schoolbook::sqr(dst, a),
            PreparedSquareKernel::Karatsuba => Karatsuba::sqr(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook3 => {
                Toom3::sqr(dst, a, &mut self.scratch);
            }
            PreparedSquareKernel::ToomCook4 => Toom4::sqr(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook6 => Toom6::sqr(dst, a, &mut self.scratch),
            PreparedSquareKernel::ToomCook85 => {
                Toom8::sqr(dst, a, &mut self.scratch);
            }
            #[cfg(not(target_pointer_width = "16"))]
            PreparedSquareKernel::Ssa(plan) => {
                let parallelism = self.executor_parallelism;
                DefaultExecutor::with_resolved_parallelism(parallelism, |executor| {
                    // SAFETY: `prepare` validated the exact destination, plan,
                    // and reusable scratch capacity, and `executor` reports the
                    // same worker budget the plan was built with.
                    unsafe {
                        plan.run_with_scratch(dst, &mut self.scratch, executor);
                    }
                });
            }
            #[cfg(not(target_pointer_width = "16"))]
            PreparedSquareKernel::SsaProduction(choice) => {
                let parallelism = self.executor_parallelism;
                let maybe_plan = SsaSquaringPlan::try_new(a, *choice, parallelism.get());
                let plan = maybe_plan.expect("validated SSA square shape must produce a plan");
                if self.scratch.len() < plan.scratch_len {
                    self.scratch.reset_with_capacity(plan.scratch_len);
                    self.scratch.resize(plan.scratch_len, 0);
                }
                DefaultExecutor::with_resolved_parallelism(parallelism, |executor| {
                    // SAFETY: the plan borrows the validated immutable operand;
                    // dst has its exact result width and cannot alias it or the
                    // runner's scratch. Scratch meets the plan's capacity, and
                    // executor reports the worker budget used during planning.
                    unsafe {
                        plan.run_with_scratch(dst, &mut self.scratch, executor);
                    }
                });
            }
        }
    }
}
