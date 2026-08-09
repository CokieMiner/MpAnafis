//! Raw arithmetic, multiplication, and squaring benchmark entry points.
//!
//! Callers own destination and scratch buffers so the measured algorithm body
//! does not include allocation. This is intentionally separate from the
//! crossover tuner API.

#![doc(hidden)]
#![allow(
    unsafe_code,
    reason = "architecture benchmark shims call validated raw-pointer kernels on equal-width disjoint slices"
)]

use super::{
    ArchKernels, BenchValidation, Karatsuba, Limb, Multiplication, Schoolbook, Toom3, Toom4, Toom6,
    Toom8, Toom32, Toom43,
};

/// Execute the selected simultaneous add/subtract kernel.
///
/// # Panics
///
/// Panics if the two spans have different widths.
pub fn bench_add_sub_limbs(sum: &mut [Limb], difference: &mut [Limb]) -> (Limb, Limb) {
    assert_eq!(sum.len(), difference.len(), "benchmark widths differ");
    // SAFETY: the mutable slices are independently allocated and equal-width.
    unsafe {
        ArchKernels::add_sub_limbs_unchecked(sum.as_mut_ptr(), difference.as_mut_ptr(), sum.len())
    }
}

/// Execute the selected in-place addition kernel, `dst += src`.
///
/// The direct counterpart of GMP's `mpn_add_n`, which is what makes an
/// addition comparison against another library possible at all: every other
/// shim here fuses two operations and has no single-routine equivalent.
///
/// # Panics
///
/// Panics if the two spans have different widths.
pub fn bench_add_limbs(dst: &mut [Limb], src: &[Limb]) -> Limb {
    assert_eq!(dst.len(), src.len(), "benchmark widths differ");
    // SAFETY: the two spans are independently allocated and equal-width.
    unsafe { ArchKernels::add_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), dst.len()) }
}

/// Execute the selected in-place subtraction kernel, `dst -= src`.
///
/// The counterpart of GMP's `mpn_sub_n`, and the mirror of
/// [`bench_add_limbs`]: the two kernels share a block structure, so a defect
/// in one is a defect in both and they have to be measurable side by side.
///
/// # Panics
///
/// Panics if the two spans have different widths.
pub fn bench_sub_limbs(dst: &mut [Limb], src: &[Limb]) -> Limb {
    assert_eq!(dst.len(), src.len(), "benchmark widths differ");
    // SAFETY: the two spans are independently allocated and equal-width.
    unsafe { ArchKernels::sub_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), dst.len()) }
}

/// Execute the selected dual-addition kernel.
///
/// # Panics
///
/// Panics unless all four spans have the same width.
pub fn bench_add_two_limbs(
    dst_a: &mut [Limb],
    src_a: &[Limb],
    dst_b: &mut [Limb],
    src_b: &[Limb],
) -> (Limb, Limb) {
    assert_eq!(dst_a.len(), src_a.len(), "benchmark widths differ");
    assert_eq!(dst_b.len(), src_b.len(), "benchmark widths differ");
    assert_eq!(dst_a.len(), dst_b.len(), "benchmark widths differ");
    // SAFETY: all benchmark vectors are disjoint and have equal widths.
    unsafe {
        ArchKernels::add_two_limbs_unchecked(
            dst_a.as_mut_ptr(),
            src_a.as_ptr(),
            dst_b.as_mut_ptr(),
            src_b.as_ptr(),
            dst_a.len(),
        )
    }
}

/// Execute two additions through separate selected kernels.
///
/// # Panics
///
/// Panics unless all four spans have the same width.
pub fn bench_add_two_sequential_limbs(
    dst_a: &mut [Limb],
    src_a: &[Limb],
    dst_b: &mut [Limb],
    src_b: &[Limb],
) -> (Limb, Limb) {
    assert_eq!(dst_a.len(), src_a.len(), "benchmark widths differ");
    assert_eq!(dst_b.len(), src_b.len(), "benchmark widths differ");
    assert_eq!(dst_a.len(), dst_b.len(), "benchmark widths differ");
    // SAFETY: each destination/source pair is disjoint and equal-width; the
    // calls operate on independent benchmark vectors.
    unsafe {
        (
            ArchKernels::add_limbs_unchecked(dst_a.as_mut_ptr(), src_a.as_ptr(), dst_a.len()),
            ArchKernels::add_limbs_unchecked(dst_b.as_mut_ptr(), src_b.as_ptr(), dst_b.len()),
        )
    }
}

/// Execute schoolbook multiplication.
///
/// # Panics
///
/// Panics if `dst` is shorter than the full product.
pub fn bench_schoolbook_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    BenchValidation::product(dst, a, b);
    Schoolbook::mul(dst, a, b);
}

/// Execute raw equal-width basecase multiplication after benchmark setup has
/// established its unsafe span contract.
///
/// # Panics
///
/// Debug builds panic unless the operands are equal-width with at least two
/// limbs and `dst` holds the complete product. Release benchmark builds rely
/// on the caller contract, so the timed work is the kernel alone.
#[inline]
pub fn bench_schoolbook_mul_raw(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    debug_assert_eq!(a.len(), b.len(), "raw basecase operands differ");
    debug_assert!(
        a.len() >= 2,
        "raw basecase operand is shorter than two limbs"
    );
    debug_assert!(
        dst.len() >= a.len().saturating_add(b.len()),
        "raw basecase destination is shorter than the complete product"
    );
    // SAFETY: the dedicated benchmark constructs three disjoint vectors with
    // exact equal widths and a destination of their summed length.
    unsafe {
        ArchKernels::mul_basecase_unchecked(
            dst.as_mut_ptr(),
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
        );
    }
}

/// Return exact scratch for one forced root Karatsuba level.
#[must_use]
pub fn bench_karatsuba_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::karatsuba_mul_forced_scratch_len(len_a, len_b)
}

/// Execute one forced root Karatsuba level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_karatsuba_mul_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::karatsuba_mul_forced_scratch_len(a.len(), b.len()),
    );
    Karatsuba::mul_forced(dst, a, b, scratch);
}

/// Return exact scratch for one forced root Toom-3 level.
#[must_use]
pub fn bench_toom_cook_3_forced_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom3_mul_forced_scratch_len(len_a, len_b)
}

/// Execute one forced root Toom-3 level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_3_mul_forced_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom3_mul_forced_scratch_len(a.len(), b.len()),
    );
    Toom3::mul_forced(dst, a, b, scratch);
}

/// Return scratch for one root Toom-4 level.
#[must_use]
pub fn bench_toom_cook_4_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom4_mul_scratch_len(len_a, len_b)
}

/// Execute one root Toom-4 level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_4_mul_forced_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom4_mul_scratch_len(a.len(), b.len()),
    );
    Toom4::mul(dst, a, b, scratch);
}

/// Return scratch for one Toom-3-by-2 multiplication.
#[must_use]
pub fn bench_toom_cook_32_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom32_mul_scratch_len(len_a, len_b)
}

/// Execute Toom-3-by-2 with caller-owned scratch.
///
/// The tier has no internal fallback, so the shape must be one
/// `Widths::toom32_suitable` admits; the driver's own assertion states that.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_32_mul_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom32_mul_scratch_len(a.len(), b.len()),
    );
    Toom32::mul(dst, a, b, scratch);
}

/// Return scratch for one Toom-4-by-3 multiplication.
#[must_use]
pub fn bench_toom_cook_43_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom43_mul_scratch_len(len_a, len_b)
}

/// Execute Toom-4-by-3 with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_43_mul_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom43_mul_scratch_len(a.len(), b.len()),
    );
    Toom43::mul(dst, a, b, scratch);
}

/// Return scratch for one Toom-6/6.5 multiplication.
#[must_use]
pub fn bench_toom_cook_6_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom6_mul_scratch_len(len_a, len_b)
}

/// Execute Toom-6/6.5 with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_6_mul_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom6_mul_scratch_len(a.len(), b.len()),
    );
    Toom6::mul(dst, a, b, scratch);
}

/// Return scratch for one Toom-8/8.5 multiplication.
#[must_use]
pub fn bench_toom_cook_8_scratch_len(len_a: usize, len_b: usize) -> usize {
    Multiplication::toom8_mul_scratch_len(len_a, len_b)
}

/// Execute Toom-8/8.5 with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_8_mul_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::product(dst, a, b);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom8_mul_scratch_len(a.len(), b.len()),
    );
    Toom8::mul(dst, a, b, scratch);
}

/// Execute schoolbook squaring with a reusable destination.
///
/// # Panics
///
/// Panics if `dst` is shorter than the full square.
pub fn bench_schoolbook_sqr(dst: &mut [Limb], a: &[Limb]) {
    BenchValidation::square(dst, a);
    dst.fill(0);
    Schoolbook::sqr(dst, a);
}

/// Return exact scratch for one forced root Karatsuba square level.
#[must_use]
pub fn bench_karatsuba_sqr_scratch_len(len: usize) -> usize {
    Multiplication::karatsuba_sqr_forced_scratch_len(len)
}

/// Execute one forced root Karatsuba square level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_karatsuba_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::square(dst, a);
    BenchValidation::scratch(
        scratch,
        Multiplication::karatsuba_sqr_forced_scratch_len(a.len()),
    );
    dst.fill(0);
    Karatsuba::sqr_forced(dst, a, scratch);
}

/// Return exact scratch for one forced root Toom-3 square level.
#[must_use]
pub fn bench_toom_cook_3_sqr_forced_scratch_len(len: usize) -> usize {
    Multiplication::toom3_sqr_forced_scratch_len(len)
}

/// Execute one forced root Toom-3 square level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_3_sqr_forced_with_scratch(
    dst: &mut [Limb],
    a: &[Limb],
    scratch: &mut [Limb],
) {
    BenchValidation::square(dst, a);
    BenchValidation::scratch(
        scratch,
        Multiplication::toom3_sqr_forced_scratch_len(a.len()),
    );
    dst.fill(0);
    Toom3::sqr_forced(dst, a, scratch);
}

/// Return scratch for one Toom-4 square level.
#[must_use]
pub fn bench_toom_cook_4_sqr_scratch_len(len: usize) -> usize {
    Multiplication::toom4_sqr_scratch_len(len)
}

/// Execute one Toom-4 square level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_4_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::square(dst, a);
    BenchValidation::scratch(scratch, Multiplication::toom4_sqr_scratch_len(a.len()));
    dst.fill(0);
    Toom4::sqr(dst, a, scratch);
}

/// Return scratch for one Toom-6 square level.
#[must_use]
pub fn bench_toom_cook_6_sqr_scratch_len(len: usize) -> usize {
    Multiplication::toom6_sqr_scratch_len(len)
}

/// Execute one Toom-6 square level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_6_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::square(dst, a);
    BenchValidation::scratch(scratch, Multiplication::toom6_sqr_scratch_len(a.len()));
    dst.fill(0);
    Toom6::sqr(dst, a, scratch);
}

/// Return scratch for one Toom-8 square level.
#[must_use]
pub fn bench_toom_cook_8_sqr_scratch_len(len: usize) -> usize {
    Multiplication::toom8_sqr_scratch_len(len)
}

/// Execute one Toom-8 square level with caller-owned scratch.
///
/// # Panics
///
/// Panics if the destination or scratch buffer is undersized.
pub fn bench_toom_cook_8_sqr_with_scratch(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
    BenchValidation::square(dst, a);
    BenchValidation::scratch(scratch, Multiplication::toom8_sqr_scratch_len(a.len()));
    dst.fill(0);
    Toom8::sqr(dst, a, scratch);
}
