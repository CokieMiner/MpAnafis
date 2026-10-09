//! Forced multiplication runners with shape-bound calls and reusable scratch.

#[cfg(not(target_pointer_width = "16"))]
use core::num::NonZeroUsize;

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use super::ParallelExecutor;
#[cfg(not(target_pointer_width = "16"))]
use super::{DefaultExecutor, Ssa, SsaMultiplicationPlan, TransformChoice};
use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Toom3, Toom4, Toom6, Toom8,
};

/// Root multiplication tier measured by [`MultiplicationRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MultiplicationAlgorithm {
    /// Quadratic schoolbook multiplication.
    Schoolbook,
    /// One forced Karatsuba level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 level with normal child dispatch.
    ToomCook3,
    /// One forced Toom-Cook 4 level with normal child dispatch.
    ToomCook4,
    /// One forced Toom-Cook 6/6.5 level with normal child dispatch.
    ToomCook6,
    /// One forced Toom-Cook 8.5 level with normal child dispatch.
    ToomCook85,
    /// Schönhage-Strassen multiplication with a forced transform geometry.
    #[cfg(not(target_pointer_width = "16"))]
    SsaForced,
    /// Schönhage-Strassen multiplication using production planning.
    #[cfg(not(target_pointer_width = "16"))]
    SsaProduction,
    /// Schönhage-Strassen multiplication forced through the two-modulus CRT path.
    #[cfg(not(target_pointer_width = "16"))]
    SsaCrt,
    /// Schönhage-Strassen multiplication forced through one full-width Fermat ring.
    #[cfg(not(target_pointer_width = "16"))]
    SsaDirectFermat,
}

/// Borrowed, shape-validated multiplication call used inside timed loops.
#[derive(Debug)]
pub struct PreparedMultiplication<'runner, 'buffers> {
    runner: &'runner mut MultiplicationRunner,
    dst: &'buffers mut [Limb],
    a: &'buffers [Limb],
    b: &'buffers [Limb],
    kernel: PreparedMultiplicationKernel<'buffers>,
}

#[cfg(target_pointer_width = "16")]
type PreparedMultiplicationKernel<'buffers> = MultiplicationAlgorithm;

#[cfg(not(target_pointer_width = "16"))]
#[derive(Debug)]
#[cfg_attr(
    target_pointer_width = "64",
    expect(
        clippy::large_enum_variant,
        reason = "the prepared SSA plan is built once outside timing; boxing it would add an avoidable allocation"
    )
)]
enum PreparedMultiplicationKernel<'buffers> {
    Schoolbook,
    Karatsuba,
    ToomCook3,
    ToomCook4,
    ToomCook6,
    ToomCook85,
    Ssa(SsaMultiplicationPlan<'buffers>),
    SsaProduction(TransformChoice),
}

/// Retained algorithm, operand widths, and scratch for multiplication comparisons.
///
/// Forced tiers prepare outside execution. Production SSA includes planning in
/// each run and may grow scratch when its selected geometry requires it.
#[derive(Debug)]
pub struct MultiplicationRunner {
    algorithm: MultiplicationAlgorithm,
    len_a: usize,
    len_b: usize,
    destination_len: usize,
    #[cfg(not(target_pointer_width = "16"))]
    executor_parallelism: NonZeroUsize,
    scratch: ScratchBuffer,
}

impl MultiplicationRunner {
    /// Allocates the selected algorithm's scratch for the operand widths.
    ///
    /// A forced SSA strategy may finalize and grow its strategy-specific
    /// scratch in [`Self::prepare`] before the timed call is created.
    ///
    /// # Panics
    ///
    /// Panics if either operand width is zero, their sum overflows `usize`, or
    /// SSA cannot represent the requested product width.
    #[must_use]
    pub fn new(algorithm: MultiplicationAlgorithm, len_a: usize, len_b: usize) -> Self {
        assert!(
            len_a != 0 && len_b != 0,
            "multiplication tuner operands must be nonzero-width"
        );
        let destination_len = len_a
            .checked_add(len_b)
            .expect("multiplication tuner product width overflows usize");
        #[cfg(not(target_pointer_width = "16"))]
        if matches!(
            algorithm,
            MultiplicationAlgorithm::SsaForced
                | MultiplicationAlgorithm::SsaProduction
                | MultiplicationAlgorithm::SsaCrt
                | MultiplicationAlgorithm::SsaDirectFermat
        ) {
            assert!(
                Ssa::admits_mul(len_a, len_b),
                "SSA cannot represent the requested tuning shape"
            );
        }
        #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
        let executor_parallelism = match algorithm {
            MultiplicationAlgorithm::SsaForced
            | MultiplicationAlgorithm::SsaProduction
            | MultiplicationAlgorithm::SsaCrt
            | MultiplicationAlgorithm::SsaDirectFermat => {
                DefaultExecutor::with_resolved(|executor| executor.parallelism())
            }
            MultiplicationAlgorithm::Schoolbook
            | MultiplicationAlgorithm::Karatsuba
            | MultiplicationAlgorithm::ToomCook3
            | MultiplicationAlgorithm::ToomCook4
            | MultiplicationAlgorithm::ToomCook6
            | MultiplicationAlgorithm::ToomCook85 => NonZeroUsize::MIN,
        };
        #[cfg(all(not(feature = "rayon"), not(target_pointer_width = "16")))]
        let executor_parallelism = NonZeroUsize::MIN;
        let scratch_len = match algorithm {
            MultiplicationAlgorithm::Schoolbook => 0,
            MultiplicationAlgorithm::Karatsuba => {
                Multiplication::karatsuba_mul_scratch_len(len_a, len_b)
            }
            MultiplicationAlgorithm::ToomCook3 => {
                Multiplication::toom3_mul_scratch_len(len_a, len_b)
            }
            MultiplicationAlgorithm::ToomCook4 => {
                Multiplication::toom4_mul_scratch_len(len_a, len_b)
            }
            MultiplicationAlgorithm::ToomCook6 => {
                Multiplication::toom6_mul_scratch_len(len_a, len_b)
            }
            MultiplicationAlgorithm::ToomCook85 => {
                Multiplication::toom8_mul_scratch_len(len_a, len_b)
            }
            #[cfg(not(target_pointer_width = "16"))]
            MultiplicationAlgorithm::SsaForced
            | MultiplicationAlgorithm::SsaProduction
            | MultiplicationAlgorithm::SsaCrt
            | MultiplicationAlgorithm::SsaDirectFermat => {
                Ssa::mul_scratch_len_for_parallelism(len_a, len_b, executor_parallelism.get())
            }
        };
        let mut scratch = ScratchBuffer::acquire(scratch_len);
        // A limb slice requires initialized elements even when a kernel
        // overwrites them. Initialization occurs before repeated execution.
        scratch.resize(scratch_len, 0);
        Self {
            algorithm,
            len_a,
            len_b,
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
    /// if an SSA plan cannot be prepared from the supplied operands.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
        a: &'buffers [Limb],
        b: &'buffers [Limb],
    ) -> PreparedMultiplication<'runner, 'buffers> {
        assert_eq!(a.len(), self.len_a, "left tuner operand width changed");
        assert_eq!(b.len(), self.len_b, "right tuner operand width changed");
        assert_eq!(
            dst.len(),
            self.destination_len,
            "tuner destination width changed"
        );
        let kernel = match self.algorithm {
            MultiplicationAlgorithm::Schoolbook => PreparedMultiplicationKernel::Schoolbook,
            MultiplicationAlgorithm::Karatsuba => PreparedMultiplicationKernel::Karatsuba,
            MultiplicationAlgorithm::ToomCook3 => PreparedMultiplicationKernel::ToomCook3,
            MultiplicationAlgorithm::ToomCook4 => PreparedMultiplicationKernel::ToomCook4,
            MultiplicationAlgorithm::ToomCook6 => PreparedMultiplicationKernel::ToomCook6,
            MultiplicationAlgorithm::ToomCook85 => PreparedMultiplicationKernel::ToomCook85,
            #[cfg(not(target_pointer_width = "16"))]
            MultiplicationAlgorithm::SsaProduction => {
                PreparedMultiplicationKernel::SsaProduction(TransformChoice::PLANNED)
            }
            #[cfg(not(target_pointer_width = "16"))]
            MultiplicationAlgorithm::SsaForced
            | MultiplicationAlgorithm::SsaCrt
            | MultiplicationAlgorithm::SsaDirectFermat => {
                let choice = if matches!(self.algorithm, MultiplicationAlgorithm::SsaCrt) {
                    TransformChoice::FORCED_CRT
                } else if matches!(self.algorithm, MultiplicationAlgorithm::SsaDirectFermat) {
                    TransformChoice::FORCED_DIRECT_FERMAT
                } else {
                    TransformChoice::FORCED
                };
                let maybe_plan =
                    SsaMultiplicationPlan::try_new(a, b, choice, self.executor_parallelism);
                let plan =
                    maybe_plan.expect("validated SSA tuning shape must produce a product plan");
                debug_assert_eq!(
                    plan.result_len,
                    dst.len(),
                    "SSA product plan destination differs from validated width"
                );
                if self.scratch.len() < plan.scratch_len {
                    self.scratch.reset_with_capacity(plan.scratch_len);
                    self.scratch.resize(plan.scratch_len, 0);
                }
                PreparedMultiplicationKernel::Ssa(plan)
            }
        };
        PreparedMultiplication {
            runner: self,
            dst,
            a,
            b,
            kernel,
        }
    }

    /// Prepares and executes one multiplication call.
    ///
    /// This convenience method prepares and runs one call. Repeated measurements
    /// should retain the object returned by [`Self::prepare`] instead.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::prepare`].
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        let mut prepared = self.prepare(dst, a, b);
        prepared.run();
    }
}

impl PreparedMultiplication<'_, '_> {
    /// Executes the prepared multiplication with retained operands and scratch.
    #[expect(
        unsafe_code,
        reason = "Preparation binds exact product widths and sufficient scratch to immutable operands and exclusive output borrows"
    )]
    #[inline]
    pub fn run(&mut self) {
        // SAFETY: prepare proves dst.len() = a.len() + b.len() and sizes the
        // selected kernel's scratch. The retained shared operand borrows and
        // exclusive destination and runner borrows preserve disjoint storage,
        // widths, and plan lifetimes throughout repeated execution.
        unsafe {
            self.runner
                .run_kernel(self.dst, self.a, self.b, &self.kernel);
        }
    }
}

impl MultiplicationRunner {
    /// Executes the already validated root tier.
    ///
    /// # Safety
    ///
    /// Both operands must be nonempty, `dst.len()` must equal `a.len() + b.len()`,
    /// and `dst` must not overlap either operand or scratch. The kernel must
    /// belong to the borrowed call produced by `prepare`, with its validated
    /// scratch and worker budget.
    #[expect(
        unsafe_code,
        reason = "Prepared calls establish the buffer, scratch, and executor contracts required by SSA execution"
    )]
    unsafe fn run_kernel(
        &mut self,
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        kernel: &PreparedMultiplicationKernel<'_>,
    ) {
        match kernel {
            PreparedMultiplicationKernel::Schoolbook => Schoolbook::mul(dst, a, b),
            PreparedMultiplicationKernel::Karatsuba => {
                Karatsuba::mul(dst, a, b, &mut self.scratch);
            }
            PreparedMultiplicationKernel::ToomCook3 => {
                Toom3::mul(dst, a, b, &mut self.scratch);
            }
            PreparedMultiplicationKernel::ToomCook4 => {
                Toom4::mul(dst, a, b, &mut self.scratch);
            }
            PreparedMultiplicationKernel::ToomCook6 => {
                Toom6::mul(dst, a, b, &mut self.scratch);
            }
            PreparedMultiplicationKernel::ToomCook85 => {
                Toom8::mul(dst, a, b, &mut self.scratch);
            }
            #[cfg(not(target_pointer_width = "16"))]
            PreparedMultiplicationKernel::Ssa(plan) => {
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
            PreparedMultiplicationKernel::SsaProduction(choice) => {
                let parallelism = self.executor_parallelism;
                let maybe_plan = SsaMultiplicationPlan::try_new(a, b, *choice, parallelism);
                let plan =
                    maybe_plan.expect("validated SSA tuning shape must produce a product plan");
                if self.scratch.len() < plan.scratch_len {
                    self.scratch.reset_with_capacity(plan.scratch_len);
                    self.scratch.resize(plan.scratch_len, 0);
                }
                DefaultExecutor::with_resolved_parallelism(parallelism, |executor| {
                    // SAFETY: the plan borrows the validated immutable operands;
                    // dst has their exact product width and cannot alias them or
                    // the runner's scratch. Scratch meets the plan's capacity,
                    // and executor reports the worker budget used in planning.
                    unsafe {
                        plan.run_with_scratch(dst, &mut self.scratch, executor);
                    }
                });
            }
        }
    }
}
