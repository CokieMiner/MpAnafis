//! In-place addition against GMP, at the same widths as `balanced`.
//!
//! Addition was previously compared only against our own Rust fallback, which
//! answers whether the assembly is worth keeping but not whether it is
//! competitive. `mpn_add_n` is the exact counterpart of our `add_limbs`
//! kernel -- same signature, same in-place semantics, same carry return -- so
//! this is a like-for-like comparison with no wrapper on either side.
//!
//! Expect parity at large widths rather than a lead. Addition is one serial
//! carry chain, both implementations reach roughly one limb per cycle, and
//! nothing above the kernel can change that. The widths that matter here are
//! the small ones, where per-call overhead rather than throughput decides.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_add_n with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{
    Limb,
    algorithms::{bench_add_limbs, bench_sub_limbs},
};
use gmp_mpfr_sys::gmp::{self, limb_t, size_t};

use crate::shared::{SCALING_SIZES, TOWER_SIZES, operands_pair};

/// Asserts the limb widths both arms are indexed by agree.
const fn assert_one_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<limb_t>(),
        "the GMP comparison requires one limb width"
    );
}

#[divan::bench(args = TOWER_SIZES)]
fn arbi(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (mut dst, src, _unused) = operands_pair(len, len);
    bencher.bench_local(|| {
        let _carry = bench_add_limbs(black_box(&mut dst), black_box(&src));
        let _output = black_box(&dst);
    });
}

#[divan::bench(args = TOWER_SIZES)]
fn gmp(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (mut dst, src, _unused) = operands_pair(len, len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: `dst` and `src` are independently allocated, disjoint, and
        // each hold exactly `count` initialized limbs. `mpn_add_n` permits the
        // destination to alias the first source, which is the form used here.
        let _carry = unsafe {
            gmp::mpn_add_n(
                black_box(dst.as_mut_ptr().cast::<limb_t>()),
                black_box(dst.as_ptr().cast::<limb_t>()),
                black_box(src.as_ptr().cast::<limb_t>()),
                black_box(count),
            )
        };
        let _output = black_box(&dst);
    });
}

// A sample is one addition at these widths, because divan's calibration
// otherwise settles on a low iteration count for the widest cells and reports a
// median dominated by whatever interrupted the first sample.
#[divan::bench(args = SCALING_SIZES, sample_count = 20)]
fn arbi_wide(bencher: divan::Bencher<'_, '_>, len: usize) {
    arbi(bencher, len);
}

#[divan::bench(args = SCALING_SIZES, sample_count = 20)]
fn gmp_wide(bencher: divan::Bencher<'_, '_>, len: usize) {
    gmp(bencher, len);
}

// Subtraction shares `add_limbs`' block structure, so it is measured beside it:
// a structural defect in one is a defect in both, and only a side-by-side
// comparison shows whether a fix applied to one was carried to the other.

#[divan::bench(args = TOWER_SIZES)]
fn arbi_sub(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (mut dst, src, _unused) = operands_pair(len, len);
    bencher.bench_local(|| {
        let _borrow = bench_sub_limbs(black_box(&mut dst), black_box(&src));
        let _output = black_box(&dst);
    });
}

#[divan::bench(args = TOWER_SIZES)]
fn gmp_sub(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (mut dst, src, _unused) = operands_pair(len, len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: `dst` and `src` are independently allocated, disjoint, and
        // each hold exactly `count` initialized limbs. `mpn_sub_n` permits the
        // destination to alias the first source, which is the form used here.
        let _borrow = unsafe {
            gmp::mpn_sub_n(
                black_box(dst.as_mut_ptr().cast::<limb_t>()),
                black_box(dst.as_ptr().cast::<limb_t>()),
                black_box(src.as_ptr().cast::<limb_t>()),
                black_box(count),
            )
        };
        let _output = black_box(&dst);
    });
}

/// The same kernel as `arbi`, measured again under a different name.
///
/// Divan runs a group's arms in name order and a later arm benefits from warm
/// state, so two arms running identical code do not report identical times.
/// The gap between this and `arbi` is that bias and nothing else, which makes
/// it the floor any kernel comparison in this group has to clear: a candidate
/// beating `arbi` by less than `arbi_bias_control` does is not a win.
///
/// It calls the same function rather than duplicating the kernel. A copied
/// routine drifts the moment the original is edited, and a control that has
/// silently stopped being a copy is worse than no control at all.
#[divan::bench(args = TOWER_SIZES)]
fn arbi_bias_control(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (mut dst, src, _unused) = operands_pair(len, len);
    bencher.bench_local(|| {
        let _carry = bench_add_limbs(black_box(&mut dst), black_box(&src));
        let _output = black_box(&dst);
    });
}
