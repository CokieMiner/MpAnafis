//! NTT and SSA benchmark entry points.

use alloc::vec::Vec;

use super::{BenchValidation, Limb, Ntt, Ssa, TransformBench, TransformChoice};

/// Execute exact multi-prime NTT/CRT multiplication.
///
/// # Panics
///
/// Panics if `dst` is undersized or the operands exceed the fixed transform
/// roots used by this benchmark facade.
pub fn bench_ntt_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    BenchValidation::product(dst, a, b);
    let completed = Ntt::try_mul(dst, a, b);
    assert!(completed, "benchmark transform exceeds the fixed NTT roots");
}

/// Execute exact NTT/CRT multiplication with a forced transform plan.
///
/// # Panics
///
/// Panics if `dst` is undersized, the digit width or modulus count is invalid,
/// or the resulting transform exceeds the supported roots or CRT range.
pub fn bench_ntt_mul_forced(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    digit_bits: u32,
    modulus_count: usize,
) {
    BenchValidation::product(dst, a, b);
    let completed = Ntt::try_mul_with_forced_plan(dst, a, b, digit_bits, modulus_count);
    assert!(
        completed,
        "forced NTT plan exceeds its exact transform bounds"
    );
}

/// Execute recursive Schönhage-Strassen multiplication.
///
/// # Panics
///
/// Panics if `dst` is undersized or the operands exceed the supported SSA
/// dimensions.
pub fn bench_ssa_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    BenchValidation::product(dst, a, b);
    let completed = Ssa::try_mul(dst, a, b, TransformChoice::FORCED, None);
    assert!(completed, "benchmark dimensions exceed the SSA size bounds");
}

/// Return the exact SSA scratch width for the planner's own geometry choice.
#[must_use]
pub fn bench_ssa_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
    Ssa::mul_scratch_len(len_a, len_b)
}

/// The top-level `modulus_bits` used for SSA multiplication of the given lengths.
#[must_use]
pub fn bench_ssa_mul_modulus_bits(len_a: usize, len_b: usize) -> Option<usize> {
    TransformBench::ssa_mul_modulus_bits(len_a, len_b)
}

/// The top-level `modulus_bits` used for SSA squaring of the given length.
#[must_use]
pub fn bench_ssa_sqr_modulus_bits(len: usize) -> Option<usize> {
    TransformBench::ssa_sqr_modulus_bits(len)
}

/// Collect the exact inner ring widths visited by the planner up to a limit.
#[must_use]
pub fn bench_collect_ssa_inner_rings(max_modulus_bits: usize) -> Vec<usize> {
    TransformBench::collect_ssa_inner_rings(max_modulus_bits)
}

/// Return exact SSA scratch for one forced transform geometry.
#[must_use]
pub fn bench_ssa_mul_forced_plan_scratch_len(
    len_a: usize,
    len_b: usize,
    transform_exponent: u32,
) -> Option<usize> {
    TransformBench::ssa_mul_scratch_len_for_plan(len_a, len_b, transform_exponent)
}

/// Execute recursive SSA with caller-owned scratch.
///
/// # Panics
///
/// Panics if any buffer is undersized or the dimensions are unsupported.
pub fn bench_ssa_mul_with_scratch(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(scratch, Ssa::mul_scratch_len(a.len(), b.len()));
    let completed = Ssa::try_mul(dst, a, b, TransformChoice::FORCED, Some(scratch));
    assert!(completed, "benchmark dimensions exceed the SSA size bounds");
}

/// Execute the exact production SSA path with caller-owned scratch.
///
/// # Panics
///
/// Panics if any buffer is undersized or the dimensions are unsupported.
pub fn bench_ssa_mul_production(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(scratch, Ssa::mul_scratch_len(a.len(), b.len()));
    let completed = Ssa::try_mul(dst, a, b, TransformChoice::PLANNED, Some(scratch));
    assert!(completed, "benchmark dimensions exceed the SSA size bounds");
}

/// Execute one forced SSA transform geometry with caller-owned scratch.
///
/// # Panics
///
/// Panics if the geometry is invalid or any buffer is undersized.
pub fn bench_ssa_mul_forced_plan(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    transform_exponent: u32,
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    let required =
        TransformBench::ssa_mul_scratch_len_for_plan(a.len(), b.len(), transform_exponent)
            .expect("forced SSA geometry is invalid for this operand size");
    BenchValidation::scratch(scratch, required);
    let completed = Ssa::try_mul(
        dst,
        a,
        b,
        TransformChoice::forced_at(transform_exponent),
        Some(scratch),
    );
    assert!(completed, "forced SSA geometry exceeds its exact bounds");
}

/// Exact SSA scratch for a forced square of this width.
#[must_use]
pub fn bench_ssa_sqr_scratch_len(len: usize) -> usize {
    Ssa::sqr_scratch_len(len)
}

/// Execute recursive SSA squaring with caller-owned scratch.
///
/// Forces the transform regardless of the crossover, so that the squaring
/// crossover can be measured rather than assumed.
///
/// # Panics
///
/// Panics if any buffer is undersized or the dimensions are unsupported.
pub fn bench_ssa_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
    assert!(
        dst.len() >= a.len().saturating_mul(2),
        "SSA squaring destination is shorter than the square"
    );
    BenchValidation::scratch(scratch, Ssa::sqr_scratch_len(a.len()));
    let completed = Ssa::try_sqr(dst, a, TransformChoice::FORCED, Some(scratch));
    assert!(completed, "benchmark dimensions exceed the SSA size bounds");
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
