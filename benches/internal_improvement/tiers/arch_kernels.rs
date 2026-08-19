//! Microbenchmarks for arithmetic kernels used by multiplication tiers.

use core::{fmt, hint::black_box, ops::BitOr};

use mp_anafis::tune_api::tier::{
    Limb, Tuner,
    transform::{NttKernelBackend, NttKernelDirection},
};

use crate::shared::{ADD_SUB_SIZES, operand};

const NTT_RADIX4_QUARTER_SIZES: [usize; 4] = [8, 64, 1_024, 16_384];

#[derive(Clone, Copy, Debug)]
struct NttRadix4Case {
    backend: NttKernelBackend,
    direction: NttKernelDirection,
    quarter_len: usize,
}

impl fmt::Display for NttRadix4Case {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backend = self.backend.label();
        let direction = match self.direction {
            NttKernelDirection::Dif => "dif",
            NttKernelDirection::Dit => "dit",
            _ => "unknown-direction",
        };
        write!(
            formatter,
            "ntt-radix4/{direction}/{backend}/{}-groups",
            self.quarter_len
        )
    }
}

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
        let carries = Tuner::bench_add_sub_limbs(black_box(&mut sum), black_box(&mut difference));
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
        let carries = Tuner::bench_add_two_limbs(
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
        let carries = Tuner::bench_add_two_sequential_limbs(
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

#[divan::bench(args = ntt_radix4_cases())]
fn ntt_radix4_kernel(bencher: divan::Bencher<'_, '_>, case: NttRadix4Case) {
    let mut runner = Tuner::bench_ntt_radix4_kernel(case.backend, case.direction, case.quarter_len);
    bencher.bench_local(|| black_box(&mut runner).run());
}

fn ntt_radix4_cases() -> Vec<NttRadix4Case> {
    let mut cases = Vec::with_capacity(NTT_RADIX4_QUARTER_SIZES.len().saturating_mul(4));
    for quarter_len in NTT_RADIX4_QUARTER_SIZES {
        for direction in [NttKernelDirection::Dif, NttKernelDirection::Dit] {
            let selected = NttKernelBackend::RuntimeSelected;
            if selected.differs_from_scalar(direction) {
                cases.push(NttRadix4Case {
                    backend: selected,
                    direction,
                    quarter_len,
                });
            }
            cases.push(NttRadix4Case {
                backend: NttKernelBackend::ScalarReference,
                direction,
                quarter_len,
            });
        }
    }
    cases
}
