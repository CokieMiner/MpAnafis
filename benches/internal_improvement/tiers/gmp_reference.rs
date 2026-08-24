//! Raw GMP multiplication-tower baselines at Mp's exact limb widths.

#![expect(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::Limb;

use crate::shared::{
    HALF_SIZES, KARATSUBA_SIZES, SCHOOLBOOK_SIZES, SSA_SCORECARD_SIZES, TOOM3_SIZES, TOOM4_SIZES,
    TOOM6_SIZES, TOOM8_SIZES, TOWER_SIZES, operand, operands, operands_pair, to_gmp_limbs,
    validated_gmp_count, validated_gmp_counts,
};

type GmpToomMul =
    unsafe extern "C" fn(*mut limb_t, *const limb_t, size_t, *const limb_t, size_t, *mut limb_t);

unsafe extern "C" {
    #[link_name = "__gmpn_mul_basecase"]
    fn mpn_mul_basecase(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
    );
    #[link_name = "__gmpn_toom22_mul"]
    fn mpn_toom22_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
        scratch: *mut limb_t,
    );
    #[link_name = "__gmpn_toom33_mul"]
    fn mpn_toom33_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
        scratch: *mut limb_t,
    );
    #[link_name = "__gmpn_toom44_mul"]
    fn mpn_toom44_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
        scratch: *mut limb_t,
    );
    #[link_name = "__gmpn_toom6h_mul"]
    fn mpn_toom6h_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
        scratch: *mut limb_t,
    );
    #[link_name = "__gmpn_toom8h_mul"]
    fn mpn_toom8h_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
        scratch: *mut limb_t,
    );

    #[link_name = "__gmpn_nussbaumer_mul"]
    fn mpn_nussbaumer_mul(
        destination: *mut limb_t,
        left: *const limb_t,
        left_len: size_t,
        right: *const limb_t,
        right_len: size_t,
    );
}

#[divan::bench(args = TOWER_SIZES)]
fn mul_tower(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination, gmp_len) = gmp_operands(len);

    bencher.bench_local(|| {
        // SAFETY: left and right each contain exactly gmp_len initialized GMP
        // limbs, destination contains 2*gmp_len limbs, and the three vectors
        // are independently allocated and therefore do not overlap.
        unsafe {
            gmp::mpn_mul_n(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(right.as_ptr()),
                black_box(gmp_len),
            );
        }
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = SCHOOLBOOK_SIZES)]
fn basecase(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination, gmp_len) = gmp_operands(len);

    bencher.bench_local(|| {
        // SAFETY: left and right each contain exactly gmp_len initialized GMP
        // limbs, destination contains 2*gmp_len limbs, and all three vectors
        // are disjoint. Equal lengths satisfy the basecase ordering contract.
        unsafe {
            mpn_mul_basecase(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(gmp_len),
                black_box(right.as_ptr()),
                black_box(gmp_len),
            );
        }
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = KARATSUBA_SIZES)]
fn toom22(bencher: divan::Bencher, len: usize) {
    bench_toom_mul(bencher, len, mpn_toom22_mul);
}

#[divan::bench(args = TOOM3_SIZES)]
fn toom33(bencher: divan::Bencher, len: usize) {
    bench_toom_mul(bencher, len, mpn_toom33_mul);
}

#[divan::bench(args = TOOM4_SIZES)]
fn toom44(bencher: divan::Bencher, len: usize) {
    bench_toom_mul(bencher, len, mpn_toom44_mul);
}

#[divan::bench(args = TOOM6_SIZES)]
fn toom6h(bencher: divan::Bencher, len: usize) {
    bench_toom_mul(bencher, len, mpn_toom6h_mul);
}

#[divan::bench(args = HALF_SIZES)]
fn toom6h_half(bencher: divan::Bencher, lengths: (usize, usize)) {
    assert_compatible_limb_width();
    let (left_len, right_len) = lengths;
    assert!(
        left_len >= right_len,
        "GMP Toom-6.5 requires the longer operand first"
    );
    let (mp_left, mp_right, _) = operands_pair(left_len, right_len);
    let left = to_gmp_limbs(mp_left);
    let right = to_gmp_limbs(mp_right);
    let result_len = left_len.saturating_add(right_len);
    let mut destination = vec![limb_t::MIN; result_len];
    let mut expected = vec![limb_t::MIN; result_len];
    let (gmp_left_len, gmp_right_len) = validated_gmp_counts(left_len, right_len);
    let scratch_len = left_len.saturating_mul(64).saturating_add(65_536);
    let mut scratch = vec![limb_t::MIN; scratch_len];

    // SAFETY: the inputs contain the exact initialized spans passed to GMP,
    // both outputs contain left_len+right_len writable limbs, and every vector
    // is independently allocated. Each configured 7:6 pair satisfies GMP
    // 6.3's Toom-6.5 shape and minimum-width contracts. Its published itch is
    // below four input lengths plus threshold constants, strictly dominated by
    // the 64*left_len+65,536 scratch allocation.
    unsafe {
        let _ = gmp::mpn_mul(
            expected.as_mut_ptr(),
            left.as_ptr(),
            gmp_left_len,
            right.as_ptr(),
            gmp_right_len,
        );
        mpn_toom6h_mul(
            destination.as_mut_ptr(),
            left.as_ptr(),
            gmp_left_len,
            right.as_ptr(),
            gmp_right_len,
            scratch.as_mut_ptr(),
        );
    }
    assert_eq!(
        destination, expected,
        "forced GMP Toom-6.5 disagrees with GMP's production tower"
    );

    bencher.bench_local(|| {
        // SAFETY: the validated disjoint spans and overprovisioned scratch
        // allocation above remain alive and unchanged for every iteration.
        unsafe {
            mpn_toom6h_mul(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(gmp_left_len),
                black_box(right.as_ptr()),
                black_box(gmp_right_len),
                black_box(scratch.as_mut_ptr()),
            );
        }
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM8_SIZES)]
fn toom8h(bencher: divan::Bencher, len: usize) {
    bench_toom_mul(bencher, len, mpn_toom8h_mul);
}

#[divan::bench(args = TOWER_SIZES)]
fn nussbaumer(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination, gmp_len) = gmp_operands(len);
    let mut expected = vec![limb_t::MIN; destination.len()];
    // SAFETY: every input and output vector has the exact equal-width span
    // passed to GMP and the independent allocations cannot overlap.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            gmp_len,
        );
        mpn_nussbaumer_mul(
            destination.as_mut_ptr(),
            left.as_ptr(),
            gmp_len,
            right.as_ptr(),
            gmp_len,
        );
    }
    assert_eq!(
        destination, expected,
        "forced GMP Nussbaumer multiplication disagrees with GMP's production tower"
    );

    bencher.bench_local(|| {
        // SAFETY: the same exact disjoint vector spans validated above remain
        // alive and initialized for every benchmark iteration.
        unsafe {
            mpn_nussbaumer_mul(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(gmp_len),
                black_box(right.as_ptr()),
                black_box(gmp_len),
            );
        }
        let _output = black_box(&destination);
    });
}

/// Focused GMP reference cells paired with `algorithms::ssa_scorecard`.
#[divan::bench(args = SSA_SCORECARD_SIZES)]
fn nussbaumer_scorecard(bencher: divan::Bencher, len: usize) {
    nussbaumer(bencher, len);
}

#[divan::bench(args = TOWER_SIZES)]
fn square_tower(bencher: divan::Bencher, len: usize) {
    assert_compatible_limb_width();
    let mp_value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let value = to_gmp_limbs(mp_value);
    let mut destination = vec![limb_t::MIN; len.saturating_mul(2)];
    let gmp_len = validated_gmp_count(len);

    bencher.bench_local(|| {
        // SAFETY: value contains exactly gmp_len initialized GMP limbs,
        // destination contains 2*gmp_len limbs, and the independently
        // allocated vectors cannot overlap.
        unsafe {
            gmp::mpn_sqr(
                black_box(destination.as_mut_ptr()),
                black_box(value.as_ptr()),
                black_box(gmp_len),
            );
        }
        let _output = black_box(&destination);
    });
}

fn bench_toom_mul(bencher: divan::Bencher, len: usize, multiply: GmpToomMul) {
    let (left, right, mut destination, gmp_len) = gmp_operands(len);
    let scratch_len = len.saturating_mul(64).saturating_add(65_536);
    let mut scratch = vec![limb_t::MIN; scratch_len];
    let mut expected = vec![limb_t::MIN; destination.len()];

    // SAFETY: all input and output spans have the exact lengths passed to
    // GMP and are pairwise disjoint. GMP 6.3's published Toom itch bounds
    // are below four input lengths plus threshold constants; 64*n+65,536
    // limbs strictly dominates those bounds for every configured size.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            gmp_len,
        );
        multiply(
            destination.as_mut_ptr(),
            left.as_ptr(),
            gmp_len,
            right.as_ptr(),
            gmp_len,
            scratch.as_mut_ptr(),
        );
    }
    assert_eq!(
        destination, expected,
        "forced GMP tier disagrees with GMP's production tower"
    );

    bencher.bench_local(|| {
        // SAFETY: the validated call above uses the same disjoint pointers,
        // lengths, and overprovisioned scratch allocation on every iteration.
        unsafe {
            multiply(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(gmp_len),
                black_box(right.as_ptr()),
                black_box(gmp_len),
                black_box(scratch.as_mut_ptr()),
            );
        }
        let _output = black_box(&destination);
    });
}

fn gmp_operands(len: usize) -> (Vec<limb_t>, Vec<limb_t>, Vec<limb_t>, size_t) {
    assert_compatible_limb_width();
    let (mp_left, mp_right, _) = operands(len);
    let left = to_gmp_limbs(mp_left);
    let right = to_gmp_limbs(mp_right);
    let destination = vec![limb_t::MIN; len.saturating_mul(2)];
    let gmp_len = validated_gmp_count(len);
    (left, right, destination, gmp_len)
}

fn assert_compatible_limb_width() {
    let gmp_numb_bits = u32::try_from(gmp::NUMB_BITS).expect("GMP limb width is positive");
    assert_eq!(
        Limb::BITS,
        gmp_numb_bits,
        "raw tier comparisons require equal Mp and GMP limb widths"
    );
}
