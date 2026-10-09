//! Low-product multiplication runners for crossover measurement.

#![expect(
    unsafe_code,
    reason = "Preparation binds initialized operand prefixes, disjoint writable output, and the selected kernel's complete scratch layout"
)]

use core::{
    num::NonZeroUsize,
    ptr::{copy_nonoverlapping, eq},
};

use crate::parallel::{DefaultExecutor, ParallelExecutor};

use super::{
    Limb, LowProduct, MulPlan, MulScratch, Multiplication, Schoolbook, SquarePlan, TierCeiling,
};

/// Root low-product algorithm measured by [`LowProductRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LowProductAlgorithm {
    /// Triangular schoolbook basecase low product.
    Schoolbook,
    /// One Mulders root split with the production crossover for its children.
    /// Widths below four limbs use schoolbook multiplication.
    Mulders,
    /// Full multiplication truncated to the low `len` limbs.
    Full,
}

/// Borrowed, shape-validated low-product call used inside timed loops.
#[derive(Debug)]
pub struct PreparedLowProduct<'runner, 'buffers> {
    scratch: &'runner mut [Limb],
    dst: &'buffers mut [Limb],
    a: &'buffers [Limb],
    b: &'buffers [Limb],
    kernel: LowProductKernel,
    parallelism: NonZeroUsize,
}

/// Retained algorithm, operand width, and scratch for low-product comparisons.
#[derive(Debug)]
pub struct LowProductRunner {
    schedule: LowProductSchedule,
    len: usize,
    scratch: MulScratch,
    parallelism: NonZeroUsize,
}

/// Shape-dependent decisions retained across preparation calls.
#[derive(Clone, Copy, Debug)]
enum LowProductSchedule {
    Schoolbook,
    Mulders {
        small_len: usize,
    },
    Full {
        product_len: usize,
        product: MulPlan,
        square: SquarePlan,
    },
}

/// Root operation selected for one immutable operand pair.
#[derive(Debug)]
enum LowProductKernel {
    Schoolbook,
    Mulders {
        small_len: usize,
    },
    Product {
        product_len: usize,
        plan: MulPlan,
    },
    Square {
        product_len: usize,
        plan: SquarePlan,
    },
}

impl LowProductRunner {
    /// Allocates the selected algorithm's scratch for the operand width.
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero or scratch sizing overflows `usize`.
    #[must_use]
    pub fn new(algorithm: LowProductAlgorithm, len: usize) -> Self {
        assert!(len != 0, "low product operand width must be nonzero");
        let product = Multiplication::select_plan(len, len, TierCeiling::Full);
        let square = Multiplication::select_square_plan(len, TierCeiling::Full);
        let parallelism = if algorithm != LowProductAlgorithm::Schoolbook
            && (product.is_transform() || square.is_transform())
        {
            DefaultExecutor::with_resolved(|executor| executor.parallelism())
        } else {
            NonZeroUsize::MIN
        };
        let (schedule, scratch_len) = match algorithm {
            LowProductAlgorithm::Schoolbook => (LowProductSchedule::Schoolbook, 0),
            LowProductAlgorithm::Mulders if len < 4 => (LowProductSchedule::Schoolbook, 0),
            LowProductAlgorithm::Mulders => {
                let small_len = Multiplication::mulders_small_len::<2>(len);
                (
                    LowProductSchedule::Mulders { small_len },
                    LowProduct::mulders_scratch_len(len, small_len, parallelism.get()),
                )
            }
            LowProductAlgorithm::Full => {
                let product_len = len.checked_mul(2).expect("low product width fits");
                let inner_len = Multiplication::scratch_len_for_parallelism(
                    product,
                    len,
                    len,
                    parallelism.get(),
                )
                .max(Multiplication::square_scratch_len_for_parallelism(
                    square,
                    len,
                    parallelism.get(),
                ));
                (
                    LowProductSchedule::Full {
                        product_len,
                        product,
                        square,
                    },
                    product_len
                        .checked_add(inner_len)
                        .expect("scratch width fits"),
                )
            }
        };
        let mut scratch = MulScratch::default();
        scratch.prepare(scratch_len);
        Self {
            schedule,
            len,
            scratch,
            parallelism,
        }
    }

    /// Binds the low `len` input limbs, output prefix, and preallocated scratch.
    ///
    /// A shared input prefix selects squaring in the full-product algorithm.
    /// Additional input limbs are ignored; the destination suffix is unchanged.
    /// Root selection and scratch sizing occur outside repeated execution.
    /// The construction-time worker budget remains fixed across Rayon pools.
    ///
    /// # Panics
    ///
    /// Panics if an operand or destination width is smaller than the configured length.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
        a: &'buffers [Limb],
        b: &'buffers [Limb],
    ) -> PreparedLowProduct<'runner, 'buffers> {
        assert!(a.len() >= self.len, "left operand too short");
        assert!(b.len() >= self.len, "right operand too short");
        assert!(dst.len() >= self.len, "destination too short");
        let (destination, _) = dst.split_at_mut(self.len);
        let (left, _) = a.split_at(self.len);
        let (right, _) = b.split_at(self.len);
        let kernel = match self.schedule {
            LowProductSchedule::Schoolbook => LowProductKernel::Schoolbook,
            LowProductSchedule::Mulders { small_len } => LowProductKernel::Mulders { small_len },
            LowProductSchedule::Full {
                product_len,
                product,
                square,
            } => {
                if eq(left.as_ptr(), right.as_ptr()) {
                    LowProductKernel::Square {
                        product_len,
                        plan: square,
                    }
                } else {
                    LowProductKernel::Product {
                        product_len,
                        plan: product,
                    }
                }
            }
        };
        PreparedLowProduct {
            scratch: &mut self.scratch.buf,
            dst: destination,
            a: left,
            b: right,
            kernel,
            parallelism: self.parallelism,
        }
    }

    /// Prepares and executes one low-product call.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::prepare`].
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        let mut prepared = self.prepare(dst, a, b);
        prepared.run();
    }
}

impl PreparedLowProduct<'_, '_> {
    /// Executes the prepared low product with retained operands and scratch.
    #[inline]
    pub fn run(&mut self) {
        let len = self.dst.len();
        match self.kernel {
            LowProductKernel::Schoolbook => {
                // SAFETY: preparation binds three len-limb spans with len>0.
                // The exclusive destination is disjoint from the shared operands.
                // The first row initializes all limbs before accumulation reads them.
                unsafe {
                    Schoolbook::mullo_basecase_unchecked(
                        self.dst.as_mut_ptr(),
                        self.a.as_ptr(),
                        self.b.as_ptr(),
                        len,
                    );
                }
            }
            LowProductKernel::Mulders { small_len } => {
                DefaultExecutor::with_resolved_parallelism(self.parallelism, |executor| {
                    // SAFETY: len>=4 gives 0<small_len<=len/2. Construction sizes
                    // the initialized scratch for this split and executor budget.
                    // Preparation binds initialized inputs and a disjoint len-limb output.
                    unsafe {
                        LowProduct::mulders_at_split(
                            self.dst,
                            self.a,
                            self.b,
                            len,
                            small_len,
                            self.scratch,
                            executor,
                        );
                    }
                });
            }
            LowProductKernel::Product { product_len, plan } => {
                // SAFETY: construction reserves product_len=2*len initialized
                // limbs followed by scratch sufficient for the cached product plan.
                let (full_product, full_work) =
                    unsafe { self.scratch.split_at_mut_unchecked(product_len) };
                DefaultExecutor::with_resolved_parallelism(self.parallelism, |executor| {
                    Multiplication::execute_plan_with_executor(
                        plan,
                        full_product,
                        self.a,
                        self.b,
                        full_work,
                        executor,
                    );
                });
                // SAFETY: the selected kernel initializes all 2*len product limbs.
                // The len-limb exclusive output is disjoint from runner scratch.
                unsafe {
                    copy_nonoverlapping(full_product.as_ptr(), self.dst.as_mut_ptr(), len);
                }
            }
            LowProductKernel::Square { product_len, plan } => {
                // SAFETY: construction reserves product_len=2*len initialized
                // limbs followed by scratch sufficient for the cached square plan.
                let (full_product, full_work) =
                    unsafe { self.scratch.split_at_mut_unchecked(product_len) };
                DefaultExecutor::with_resolved_parallelism(self.parallelism, |executor| {
                    Multiplication::execute_square_plan_with_executor(
                        plan,
                        full_product,
                        self.a,
                        full_work,
                        executor,
                    );
                });
                // SAFETY: the selected kernel initializes all 2*len square limbs.
                // The len-limb exclusive output is disjoint from runner scratch.
                unsafe {
                    copy_nonoverlapping(full_product.as_ptr(), self.dst.as_mut_ptr(), len);
                }
            }
        }
    }
}
