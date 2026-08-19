//! Reusable forced-tier multiplication tuner.

use crate::parallel::DefaultExecutor;

use super::{
    Karatsuba, Limb, Multiplication, Schoolbook, ScratchBuffer, Ssa, Toom3, Toom4, Toom6, Toom8,
    TransformChoice,
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
}

/// Borrowed, shape-validated multiplication call used inside timed loops.
#[derive(Debug)]
pub struct PreparedMultiplication<'runner, 'buffers> {
    runner: &'runner mut MultiplicationRunner,
    dst: &'buffers mut [Limb],
    a: &'buffers [Limb],
    b: &'buffers [Limb],
}

impl PreparedMultiplication<'_, '_> {
    /// Runs the validated multiplication without repeating shape checks.
    pub fn run(&mut self) {
        // SAFETY: `prepare` validated the exact widths and disjoint spans held
        // by this borrowed call object; those borrows cannot change in-place.
        unsafe { self.runner.run_kernel(self.dst, self.a, self.b) }
    }
}

/// Allocation-free reusable state for one multiplication crossover sample.
#[derive(Debug)]
pub struct MultiplicationRunner {
    algorithm: MultiplicationAlgorithm,
    len_a: usize,
    len_b: usize,
    destination_len: usize,
    scratch: ScratchBuffer,
}

impl MultiplicationRunner {
    /// Pre-allocates the exact scratch required for `algorithm` at this shape.
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
        let destination_len = len_a
            .checked_add(len_b)
            .expect("multiplication tuner product width overflows usize");
        #[cfg(not(target_pointer_width = "16"))]
        if matches!(
            algorithm,
            MultiplicationAlgorithm::SsaForced | MultiplicationAlgorithm::SsaProduction
        ) {
            assert!(
                Ssa::admits_mul(len_a, len_b),
                "SSA cannot represent the requested tuning shape"
            );
        }
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
            MultiplicationAlgorithm::SsaForced | MultiplicationAlgorithm::SsaProduction => {
                Ssa::mul_scratch_len(len_a, len_b)
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
            scratch,
        }
    }

    /// Prepares an exact borrowed call for repeated allocation-free runs.
    ///
    /// # Panics
    ///
    /// Panics if an operand or destination width differs from construction.
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
        PreparedMultiplication {
            runner: self,
            dst,
            a,
            b,
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
    unsafe fn run_kernel(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        match self.algorithm {
            MultiplicationAlgorithm::Schoolbook => Schoolbook::mul(dst, a, b),
            MultiplicationAlgorithm::Karatsuba => {
                Karatsuba::mul_forced(dst, a, b, &mut self.scratch);
            }
            MultiplicationAlgorithm::ToomCook3 => {
                Toom3::mul_forced(dst, a, b, &mut self.scratch);
            }
            MultiplicationAlgorithm::ToomCook4 => Toom4::mul(dst, a, b, &mut self.scratch),
            MultiplicationAlgorithm::ToomCook6 => Toom6::mul(dst, a, b, &mut self.scratch),
            MultiplicationAlgorithm::ToomCook85 => Toom8::mul(dst, a, b, &mut self.scratch),
            #[cfg(not(target_pointer_width = "16"))]
            MultiplicationAlgorithm::SsaForced | MultiplicationAlgorithm::SsaProduction => {
                let choice = if matches!(self.algorithm, MultiplicationAlgorithm::SsaForced) {
                    TransformChoice::FORCED
                } else {
                    TransformChoice::PLANNED
                };
                let executor = DefaultExecutor::default();
                assert!(
                    Ssa::try_mul_with_executor(
                        dst,
                        a,
                        b,
                        choice,
                        Some(&mut self.scratch),
                        &executor,
                    ),
                    "SSA rejected the tuning shape"
                );
            }
        }
    }
}
