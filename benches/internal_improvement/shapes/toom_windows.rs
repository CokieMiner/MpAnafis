//! Unbalanced Toom-Cook comparison windows.

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{
    Limb,
    algorithms::{
        bench_toom_cook_4_mul_forced_with_scratch, bench_toom_cook_4_scratch_len,
        bench_toom_cook_6_mul_with_scratch, bench_toom_cook_6_scratch_len,
        bench_toom_cook_8_mul_with_scratch, bench_toom_cook_8_scratch_len,
    },
};

use crate::shared::{HALF_SIZES, TOOM8_HALF_SIZES, operands_pair};

#[divan::bench(args = HALF_SIZES)]
fn toom4_unbalanced(bencher: divan::Bencher, lengths: (usize, usize)) {
    let (left_len, right_len) = lengths;
    let (left, right, mut destination) = operands_pair(left_len, right_len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_4_scratch_len(left_len, right_len)];
    bencher.bench_local(|| {
        bench_toom_cook_4_mul_forced_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = HALF_SIZES)]
fn toom6_half(bencher: divan::Bencher, lengths: (usize, usize)) {
    let (left_len, right_len) = lengths;
    let (left, right, mut destination) = operands_pair(left_len, right_len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_6_scratch_len(left_len, right_len)];
    bencher.bench_local(|| {
        bench_toom_cook_6_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM8_HALF_SIZES)]
fn toom8_half(bencher: divan::Bencher, lengths: (usize, usize)) {
    let (left_len, right_len) = lengths;
    let (left, right, mut destination) = operands_pair(left_len, right_len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_8_scratch_len(left_len, right_len)];
    bencher.bench_local(|| {
        bench_toom_cook_8_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM8_HALF_SIZES)]
fn toom6_for_8_half(bencher: divan::Bencher, lengths: (usize, usize)) {
    let (left_len, right_len) = lengths;
    let (left, right, mut destination) = operands_pair(left_len, right_len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_6_scratch_len(left_len, right_len)];
    bencher.bench_local(|| {
        bench_toom_cook_6_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}
