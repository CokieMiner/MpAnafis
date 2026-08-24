//! Reusable state and lopsided multiplication benchmark entry points.

use crate::parallel::{DefaultExecutor, ParallelExecutor, SequentialExecutor};

use super::{BenchValidation, Limb, Lopsided, LowProduct, MulScratch, Multiplication, Tuner};

/// Reusable state for the configured multiplication dispatcher.
#[derive(Debug, Default)]
pub struct MultiplicationBenchState {
    scratch: MulScratch,
}

impl MultiplicationBenchState {
    /// Validates one multiplication shape and binds it to reusable scratch.
    pub fn prepare<'state, 'data>(
        &'state mut self,
        dst: &'data mut [Limb],
        a: &'data [Limb],
        b: &'data [Limb],
    ) -> PreparedMultiplication<'state, 'data> {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "reused multiplication operands must be nonempty"
        );
        BenchValidation::product(dst, a, b);
        PreparedMultiplication {
            scratch: &mut self.scratch,
            dst,
            a,
            b,
        }
    }
}

/// Validated production multiplication with reusable destination and scratch.
#[derive(Debug)]
pub struct PreparedMultiplication<'state, 'data> {
    scratch: &'state mut MulScratch,
    dst: &'data mut [Limb],
    a: &'data [Limb],
    b: &'data [Limb],
}

impl PreparedMultiplication<'_, '_> {
    /// Executes the configured multiplication tower without repeating setup checks.
    #[inline]
    pub fn run(&mut self) {
        Multiplication::mul_limbs_with_scratch(self.a, self.b, self.dst, self.scratch);
    }
}

/// Reusable state for the configured squaring dispatcher.
#[derive(Debug, Default)]
pub struct SquaringBenchState {
    scratch: MulScratch,
}

impl SquaringBenchState {
    /// Validates one squaring shape and binds it to reusable scratch.
    pub fn prepare<'state, 'data>(
        &'state mut self,
        dst: &'data mut [Limb],
        a: &'data [Limb],
    ) -> PreparedSquaring<'state, 'data> {
        assert!(!a.is_empty(), "reused squaring operand must be nonempty");
        BenchValidation::square(dst, a);
        PreparedSquaring {
            scratch: &mut self.scratch,
            dst,
            a,
        }
    }
}

/// Validated production squaring with reusable destination and scratch.
#[derive(Debug)]
pub struct PreparedSquaring<'state, 'data> {
    scratch: &'state mut MulScratch,
    dst: &'data mut [Limb],
    a: &'data [Limb],
}

impl PreparedSquaring<'_, '_> {
    /// Executes the configured squaring tower without repeating setup checks.
    #[inline]
    pub fn run(&mut self) {
        Multiplication::sqr_limbs_with_scratch(self.a, self.dst, self.scratch);
    }
}

/// Reusable state for equal-width truncated low-product benchmarks.
#[derive(Debug, Default)]
pub struct LowProductBenchState {
    scratch: MulScratch,
}

impl LowProductBenchState {
    /// Computes the triangular basecase low product directly.
    #[expect(
        unsafe_code,
        reason = "Benchmark validation proves the exact equal-width low-product spans"
    )]
    pub fn run_basecase(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        assert_low_product_buffers(dst, a, b);
        // SAFETY: `assert_low_product_buffers` validates equal nonaliasing input
        // widths and a destination containing at least `a.len()` limbs.
        unsafe {
            LowProduct::basecase(dst, a, b, a.len());
        }
    }

    /// Computes the low half of an equal-width product with reusable state.
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        assert_low_product_buffers(dst, a, b);
        LowProduct::mul(dst, a, b, a.len(), &mut self.scratch);
    }

    /// Computes a low product with a forced smaller block at the root only.
    pub fn run_forced_root(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb], small_len: usize) {
        assert_low_product_buffers(dst, a, b);
        LowProduct::mul_with_forced_root_split(dst, a, b, a.len(), small_len, &mut self.scratch);
    }
}

/// Executes the configured tower with one pooled scratch lease.
impl Tuner {
    pub fn bench_mul_tower_pooled(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        BenchValidation::product(dst, a, b);
        let mut scratch = MulScratch::default();
        Multiplication::mul_limbs_with_scratch(a, b, dst, &mut scratch);
    }

    /// Returns exact scratch for blocked multiplication with a forced block width.
    #[must_use]
    pub fn bench_lopsided_mul_scratch_len(len_a: usize, len_b: usize, block_len: usize) -> usize {
        assert!(block_len != 0, "lopsided benchmark block width is zero");
        Lopsided::mul_forced_scratch_len(len_a, len_b, block_len, 1)
    }

    /// Executes blocked multiplication with a forced block width and caller scratch.
    pub fn bench_lopsided_mul_with_scratch(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        block_len: usize,
        scratch: &mut [Limb],
    ) {
        BenchValidation::product(dst, a, b);
        assert!(block_len != 0, "lopsided benchmark block width is zero");
        BenchValidation::scratch(
            scratch,
            Lopsided::mul_forced_scratch_len(a.len(), b.len(), block_len, 1),
        );
        let executor = SequentialExecutor;
        Lopsided::mul_forced(dst, a, b, scratch, block_len, &executor);
    }

    /// Returns exact scratch for blocked multiplication at its production width.
    #[must_use]
    pub fn bench_lopsided_mul_production_scratch_len(len_a: usize, len_b: usize) -> usize {
        let smaller_len = len_a.min(len_b);
        let larger_len = len_a.max(len_b);
        DefaultExecutor::with_resolved(|executor| {
            Lopsided::mul_forced_scratch_len(
                len_a,
                len_b,
                Lopsided::block_len(larger_len, smaller_len),
                executor.parallelism().get(),
            )
        })
    }

    /// Executes blocked multiplication with the production block width.
    pub fn bench_lopsided_mul_production(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        BenchValidation::product(dst, a, b);
        BenchValidation::scratch(
            scratch,
            Self::bench_lopsided_mul_production_scratch_len(a.len(), b.len()),
        );
        DefaultExecutor::with_resolved(|executor| {
            Lopsided::mul(dst, a, b, scratch, executor);
        });
    }
}

fn assert_low_product_buffers(dst: &[Limb], a: &[Limb], b: &[Limb]) {
    assert!(!a.is_empty(), "low-product benchmark width is zero");
    assert_eq!(a.len(), b.len(), "low-product operand widths differ");
    assert!(
        dst.len() >= a.len(),
        "low-product destination has {} limbs, but {} are required",
        dst.len(),
        a.len()
    );
}
