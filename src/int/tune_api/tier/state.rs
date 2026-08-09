//! Reusable state and lopsided multiplication benchmark entry points.

use super::{BenchValidation, Limb, Lopsided, LowProduct, MulScratch, Multiplication};

/// Reusable state for the configured multiplication dispatcher.
#[derive(Debug, Default)]
pub struct MulBenchScratch {
    scratch: MulScratch,
}

impl MulBenchScratch {
    /// Executes the configured multiplication tower with reusable state.
    #[inline]
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        debug_assert!(
            !a.is_empty() && !b.is_empty(),
            "reused multiplication operands must be nonempty"
        );
        debug_assert!(
            dst.len() >= a.len().saturating_add(b.len()),
            "reused multiplication destination is shorter than the product"
        );
        Multiplication::mul_limbs_with_scratch(a, b, dst, &mut self.scratch);
    }
}

/// Reusable state for the configured squaring dispatcher.
#[derive(Debug, Default)]
pub struct SquareBenchScratch {
    scratch: MulScratch,
}

impl SquareBenchScratch {
    /// Executes the configured squaring tower with reusable state.
    #[inline]
    pub fn run(&mut self, dst: &mut [Limb], a: &[Limb]) {
        debug_assert!(!a.is_empty(), "reused squaring operand must be nonempty");
        debug_assert!(
            dst.len() >= a.len().saturating_mul(2),
            "reused squaring destination is shorter than the square"
        );
        Multiplication::sqr_limbs_with_scratch(a, dst, &mut self.scratch);
    }
}

/// Reusable state for equal-width truncated low-product benchmarks.
#[derive(Debug, Default)]
pub struct MulloBenchScratch {
    scratch: MulScratch,
}

impl MulloBenchScratch {
    /// Computes the triangular basecase low product directly.
    #[allow(
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
pub fn bench_mul_tower_pooled(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    BenchValidation::product(dst, a, b);
    let mut scratch = MulScratch::default();
    Multiplication::mul_limbs_with_scratch(a, b, dst, &mut scratch);
}

/// Returns exact scratch for blocked multiplication with a forced block width.
#[must_use]
pub fn bench_lopsided_mul_scratch_len(len_a: usize, len_b: usize, block_len: usize) -> usize {
    assert!(block_len != 0, "lopsided benchmark block width is zero");
    Lopsided::mul_forced_scratch_len(len_a, len_b, block_len)
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
        Lopsided::mul_forced_scratch_len(a.len(), b.len(), block_len),
    );
    Lopsided::mul_forced(dst, a, b, scratch, block_len);
}

/// Returns exact scratch for blocked multiplication at its production width.
#[must_use]
pub fn bench_lopsided_mul_production_scratch_len(len_a: usize, len_b: usize) -> usize {
    let smaller_len = len_a.min(len_b);
    let larger_len = len_a.max(len_b);
    Lopsided::mul_forced_scratch_len(len_a, len_b, Lopsided::block_len(larger_len, smaller_len))
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
        bench_lopsided_mul_production_scratch_len(a.len(), b.len()),
    );
    Lopsided::mul(dst, a, b, scratch);
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
