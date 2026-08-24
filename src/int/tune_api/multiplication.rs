//! Reusable forced-tier multiplication tuner.

use core::num::NonZeroUsize;

use crate::parallel::{DefaultExecutor, FixedParallelismExecutor, ParallelExecutor};

use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Ssa, SsaMultiplicationPlan, Toom3,
    Toom4, Toom6, Toom8, TransformChoice,
};

/// Root multiplication tier measured by [`Tuner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MultiplicationAlgorithm {
    /// Quadratic schoolbook multiplication.
    Schoolbook,
    /// One forced Karatsuba level with normal child dispatch.
    Karatsuba,
    /// One forced Toom-Cook 3 level with normal child dispatch.
    ToomCook3,
    /// One Toom-Cook 4 level with normal child dispatch.
    ToomCook4,
    /// One Toom-Cook 6/6.5 level with normal child dispatch.
    ToomCook6,
    /// One forced Toom-Cook 8.5 level with normal child dispatch.
    ToomCook85,
    /// Schonhage-Strassen multiplication with a forced transform geometry.
    #[cfg(not(target_pointer_width = "16"))]
    SsaForced,
    /// Schonhage-Strassen multiplication using production planning.
    #[cfg(not(target_pointer_width = "16"))]
    SsaProduction,
    /// Schonhage-Strassen multiplication forced through the two-modulus CRT path.
    #[cfg(not(target_pointer_width = "16"))]
    SsaCrt,
    /// Schonhage-Strassen multiplication forced through one full-width Fermat ring.
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

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "the prepared SSA plan is built once outside timing; boxing it would add an avoidable allocation"
)]
enum PreparedMultiplicationKernel<'buffers> {
    Schoolbook,
    Karatsuba,
    ToomCook3,
    ToomCook4,
    ToomCook6,
    ToomCook85,
    #[cfg(not(target_pointer_width = "16"))]
    Ssa(SsaMultiplicationPlan<'buffers>),
}

impl PreparedMultiplication<'_, '_> {
    /// Runs the validated multiplication without repeating shape checks.
    #[inline]
    pub fn run(&mut self) {
        // SAFETY: `prepare` validated the exact widths and disjoint spans held
        // by this borrowed call object; those borrows cannot change in-place.
        unsafe {
            self.runner
                .run_kernel(self.dst, self.a, self.b, &self.kernel);
        }
    }
}

/// Allocation-free reusable state for one multiplication crossover sample.
#[derive(Debug)]
pub struct MultiplicationRunner {
    algorithm: MultiplicationAlgorithm,
    len_a: usize,
    len_b: usize,
    destination_len: usize,
    executor_parallelism: NonZeroUsize,
    scratch: ScratchBuffer,
}

impl MultiplicationRunner {
    /// Pre-allocates the production scratch required at this shape.
    ///
    /// A forced SSA strategy may finalize and grow its strategy-specific
    /// scratch in [`Self::prepare`] before the timed call is created.
    ///
    /// # Panics
    ///
    /// Panics if either operand width is zero or SSA cannot represent the
    /// requested product width.
    #[must_use]
    pub(crate) fn new(algorithm: MultiplicationAlgorithm, len_a: usize, len_b: usize) -> Self {
        assert!(
            len_a != 0 && len_b != 0,
            "multiplication tuner operands must be nonzero-width"
        );
        assert!(
            len_a <= usize::MAX.wrapping_sub(len_b),
            "multiplication tuner product width overflows usize"
        );
        // The preceding boundary check proves this sum cannot wrap.
        let destination_len = len_a.wrapping_add(len_b);
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
        let executor_parallelism =
            DefaultExecutor::with_resolved(|executor| executor.parallelism());
        let scratch_len = match algorithm {
            MultiplicationAlgorithm::Schoolbook => 0,
            MultiplicationAlgorithm::Karatsuba => {
                Multiplication::karatsuba_mul_forced_scratch_len(len_a, len_b)
            }
            MultiplicationAlgorithm::ToomCook3 => {
                Multiplication::toom3_mul_forced_scratch_len(len_a, len_b)
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
        // SAFETY: forced tiers write or clear every scratch region before
        // reading it, and their sizing functions include child workspaces.
        unsafe {
            scratch.set_len(scratch_len);
        }
        Self {
            algorithm,
            len_a,
            len_b,
            destination_len,
            executor_parallelism,
            scratch,
        }
    }

    /// Prepares an exact borrowed call for repeated allocation-free runs.
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
            MultiplicationAlgorithm::SsaForced
            | MultiplicationAlgorithm::SsaProduction
            | MultiplicationAlgorithm::SsaCrt
            | MultiplicationAlgorithm::SsaDirectFermat => {
                let choice = match self.algorithm {
                    MultiplicationAlgorithm::SsaForced => TransformChoice::FORCED,
                    MultiplicationAlgorithm::SsaProduction => TransformChoice::PLANNED,
                    MultiplicationAlgorithm::SsaCrt => TransformChoice::FORCED_CRT,
                    MultiplicationAlgorithm::SsaDirectFermat => {
                        TransformChoice::FORCED_DIRECT_FERMAT
                    }
                    MultiplicationAlgorithm::Schoolbook
                    | MultiplicationAlgorithm::Karatsuba
                    | MultiplicationAlgorithm::ToomCook3
                    | MultiplicationAlgorithm::ToomCook4
                    | MultiplicationAlgorithm::ToomCook6
                    | MultiplicationAlgorithm::ToomCook85 => {
                        debug_assert!(false, "non-SSA algorithm reached SSA plan construction");
                        TransformChoice::PLANNED
                    }
                };
                let maybe_plan =
                    SsaMultiplicationPlan::try_new(a, b, choice, self.executor_parallelism);
                assert!(
                    maybe_plan.is_some(),
                    "validated SSA tuning shape must produce a product plan"
                );
                // SAFETY: the immediately preceding boundary assertion proves
                // that the operand-bound plan was constructed successfully.
                let plan = unsafe { maybe_plan.unwrap_unchecked() };
                debug_assert_eq!(
                    plan.destination_len(),
                    dst.len(),
                    "SSA product plan destination differs from validated width"
                );
                if self.scratch.len() < plan.scratch_len() {
                    self.scratch.reset_with_capacity(plan.scratch_len());
                    // SAFETY: prepared SSA execution writes or clears every
                    // scratch region before reading it. This adjustment occurs
                    // before the reusable timed call is returned.
                    unsafe {
                        self.scratch.set_len(plan.scratch_len());
                    }
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

    /// Multiplies with the configured root tier without caller-side allocation.
    ///
    /// This convenience method prepares and runs one call. Repeated measurements
    /// should retain the object returned by [`Self::prepare`] instead.
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        let mut prepared = self.prepare(dst, a, b);
        prepared.run();
    }

    /// Executes the already validated root tier.
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
                Karatsuba::mul_forced(dst, a, b, &mut self.scratch);
            }
            PreparedMultiplicationKernel::ToomCook3 => {
                Toom3::mul_forced(dst, a, b, &mut self.scratch);
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
                DefaultExecutor::with_resolved(|executor| {
                    let fixed = FixedParallelismExecutor::new(executor, parallelism);
                    // SAFETY: `prepare` validated the exact destination, plan,
                    // and reusable scratch capacity. `fixed` preserves the
                    // parallelism used to build the plan.
                    unsafe {
                        plan.run_with_scratch(dst, &mut self.scratch, &fixed);
                    }
                });
            }
        }
    }
}
