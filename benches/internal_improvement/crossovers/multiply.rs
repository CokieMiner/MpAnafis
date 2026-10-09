//! Isolated multiplication tiers and exact operand widths for tower reviews.

use core::hint::black_box;

use mp_anafis::tune_api::{MultiplicationAlgorithm, MultiplicationRunner};

use crate::shared::{gmp_equal_reference, operands_pair, validate_and_warm_product};

const BASECASE: [usize; 15] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 18, 20, 24];
const KARATSUBA: [usize; 9] = [19, 20, 21, 23, 24, 31, 32, 48, 64];
const TOOM3: [usize; 6] = [80, 96, 128, 160, 192, 256];
const TOOM4: [usize; 5] = [256, 320, 384, 512, 768];
const TOOM6: [usize; 5] = [512, 768, 1024, 1536, 2048];
const TOOM8: [usize; 5] = [1024, 1536, 2048, 3072, 4096];
const SSA: [usize; 6] = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072];

#[divan::bench(args = BASECASE)]
fn schoolbook(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::Schoolbook);
}

#[divan::bench(args = KARATSUBA)]
fn karatsuba(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::Karatsuba);
}

#[divan::bench(args = TOOM3)]
fn toom3(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::ToomCook3);
}

#[divan::bench(args = TOOM4)]
fn toom4(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::ToomCook4);
}

#[divan::bench(args = TOOM6)]
fn toom6(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::ToomCook6);
}

#[divan::bench(args = TOOM8)]
fn toom8(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::ToomCook85);
}

#[divan::bench(args = SSA)]
fn ssa(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure(bencher, len, MultiplicationAlgorithm::SsaProduction);
}

fn measure(bencher: divan::Bencher<'_, '_>, len: usize, algorithm: MultiplicationAlgorithm) {
    let (left, right, mut destination) = operands_pair(len, len);
    let expected = gmp_equal_reference(&left, &right);
    let mut runner = MultiplicationRunner::new(algorithm, len, len);
    validate_and_warm_product(&expected, "forced multiplication tier", |probe| {
        runner.prepare(probe, &left, &right).run();
    });
    let mut prepared = runner.prepare(&mut destination, &left, &right);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}
