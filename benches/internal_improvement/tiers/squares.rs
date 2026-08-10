//! Forced dedicated-squaring tier measurements.

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{
    Limb,
    algorithms::{
        bench_karatsuba_sqr_scratch_len, bench_karatsuba_sqr_with_scratch, bench_schoolbook_sqr,
        bench_toom_cook_3_sqr_forced_scratch_len, bench_toom_cook_3_sqr_forced_with_scratch,
        bench_toom_cook_4_sqr_scratch_len, bench_toom_cook_4_sqr_with_scratch,
        bench_toom_cook_6_sqr_scratch_len, bench_toom_cook_6_sqr_with_scratch,
        bench_toom_cook_8_sqr_scratch_len, bench_toom_cook_8_sqr_with_scratch,
    },
};

use crate::shared::{
    KARATSUBA_SIZES, SCHOOLBOOK_SIZES, TOOM3_SIZES, TOOM4_SIZES, TOOM6_SIZES, TOOM8_SIZES, operand,
};

#[divan::bench(args = SCHOOLBOOK_SIZES)]
fn schoolbook(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    bencher.bench_local(|| {
        bench_schoolbook_sqr(black_box(&mut destination), black_box(&value));
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = KARATSUBA_SIZES)]
fn karatsuba(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_karatsuba_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_karatsuba_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM3_SIZES)]
fn toom3_forced(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_toom_cook_3_sqr_forced_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_3_sqr_forced_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM4_SIZES)]
fn toom4_forced(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_toom_cook_4_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_4_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM6_SIZES)]
fn toom6_forced(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_toom_cook_6_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_6_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM8_SIZES)]
fn toom8_forced(bencher: divan::Bencher, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_toom_cook_8_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_8_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}
