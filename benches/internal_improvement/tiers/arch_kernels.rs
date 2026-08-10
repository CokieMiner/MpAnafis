//! Microbenchmarks for arithmetic kernels used by multiplication tiers.

use core::{hint::black_box, ops::BitOr};

use arbi_anafis::tune_api::tier::{
    Limb,
    algorithms::{bench_add_sub_limbs, bench_add_two_limbs, bench_add_two_sequential_limbs},
};

use crate::shared::{ADD_SUB_SIZES, operand};

fn rust_add_sub(sum: &mut [Limb], difference: &mut [Limb]) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    for (sum_limb, difference_limb) in sum.iter_mut().zip(difference) {
        let original_sum = *sum_limb;
        let original_difference = *difference_limb;
        let (partial_sum, overflow_a) = original_sum.overflowing_add(original_difference);
        let (final_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        let (partial_difference, underflow_a) = original_sum.overflowing_sub(original_difference);
        let (final_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        *sum_limb = final_sum;
        *difference_limb = final_difference;
        carry = BitOr::bitor(overflow_a, overflow_b);
        borrow = BitOr::bitor(underflow_a, underflow_b);
    }
    (Limb::from(carry), Limb::from(borrow))
}

fn rust_add(dst: &mut [Limb], src: &[Limb]) -> Limb {
    let mut carry = false;
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        let (partial, overflow_a) = dst_limb.overflowing_add(*src_limb);
        let (result, overflow_b) = partial.overflowing_add(Limb::from(carry));
        *dst_limb = result;
        carry = BitOr::bitor(overflow_a, overflow_b);
    }
    Limb::from(carry)
}

fn rust_add_two(
    dst_a: &mut [Limb],
    src_a: &[Limb],
    dst_b: &mut [Limb],
    src_b: &[Limb],
) -> (Limb, Limb) {
    (rust_add(dst_a, src_a), rust_add(dst_b, src_b))
}

#[divan::bench(args = ADD_SUB_SIZES)]
fn add_sub_arch(bencher: divan::Bencher, len: usize) {
    let mut sum = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut difference = operand(len, Limb::MAX.wrapping_sub(0x4321));
    bencher.bench_local(|| {
        let carries = bench_add_sub_limbs(black_box(&mut sum), black_box(&mut difference));
        let _output = black_box(carries);
    });
}

#[divan::bench(args = ADD_SUB_SIZES)]
fn add_sub_rust(bencher: divan::Bencher, len: usize) {
    let mut sum = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut difference = operand(len, Limb::MAX.wrapping_sub(0x4321));
    bencher.bench_local(|| {
        let carries = rust_add_sub(black_box(&mut sum), black_box(&mut difference));
        let _output = black_box(carries);
    });
}

#[divan::bench(args = ADD_SUB_SIZES)]
fn add_two_arch(bencher: divan::Bencher, len: usize) {
    let mut dst_a = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let src_a = operand(len, Limb::MAX.wrapping_sub(0x4321));
    let mut dst_b = operand(len, Limb::MAX.wrapping_sub(0x5678));
    let src_b = operand(len, Limb::MAX.wrapping_sub(0x8765));
    bencher.bench_local(|| {
        let carries = bench_add_two_limbs(
            black_box(&mut dst_a),
            black_box(&src_a),
            black_box(&mut dst_b),
            black_box(&src_b),
        );
        let _output = black_box(carries);
    });
}

#[divan::bench(args = ADD_SUB_SIZES)]
fn add_two_sequential_arch(bencher: divan::Bencher, len: usize) {
    let mut dst_a = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let src_a = operand(len, Limb::MAX.wrapping_sub(0x4321));
    let mut dst_b = operand(len, Limb::MAX.wrapping_sub(0x5678));
    let src_b = operand(len, Limb::MAX.wrapping_sub(0x8765));
    bencher.bench_local(|| {
        let carries = bench_add_two_sequential_limbs(
            black_box(&mut dst_a),
            black_box(&src_a),
            black_box(&mut dst_b),
            black_box(&src_b),
        );
        let _output = black_box(carries);
    });
}

#[divan::bench(args = ADD_SUB_SIZES)]
fn add_two_rust(bencher: divan::Bencher, len: usize) {
    let mut dst_a = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let src_a = operand(len, Limb::MAX.wrapping_sub(0x4321));
    let mut dst_b = operand(len, Limb::MAX.wrapping_sub(0x5678));
    let src_b = operand(len, Limb::MAX.wrapping_sub(0x8765));
    bencher.bench_local(|| {
        let carries = rust_add_two(
            black_box(&mut dst_a),
            black_box(&src_a),
            black_box(&mut dst_b),
            black_box(&src_b),
        );
        let _output = black_box(carries);
    });
}
