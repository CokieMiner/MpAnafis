//! Goldilocks 64-bit single-prime NTT multiplication with intra-transform multi-core concurrency.

use crate::parallel::ParallelExecutor;

use super::{Ntt, NttExecutionPolicy};

const PRIME: u64 = 0xffff_ffff_0000_0001;
const EPSILON: u64 = 0xffff_ffff;
const PRIMITIVE_ROOT: u64 = 7;

pub const PRIME_U128: u128 = 18_446_744_069_414_584_321;

#[derive(Clone, Copy)]
enum ButterflyDirection {
    Forward,
    Inverse,
}

/// Owned views for one Goldilocks convolution.
pub struct GoldilocksProduct<'workspace, E: ParallelExecutor> {
    dst_digits: &'workspace mut [u32],
    a: &'workspace [u32],
    b: &'workspace [u32],
    convolution_len: usize,
    digit_bits: u32,
    scratch_u64: &'workspace mut [u64],
    executor: &'workspace E,
}

impl<'workspace, E: ParallelExecutor> GoldilocksProduct<'workspace, E> {
    /// Creates a product state over disjoint output and scratch buffers.
    pub const fn new(
        dst_digits: &'workspace mut [u32],
        a: &'workspace [u32],
        b: &'workspace [u32],
        convolution_len: usize,
        digit_bits: u32,
        scratch_u64: &'workspace mut [u64],
        executor: &'workspace E,
    ) -> Self {
        Self {
            dst_digits,
            a,
            b,
            convolution_len,
            digit_bits,
            scratch_u64,
            executor,
        }
    }

    /// Executes the transform and returns the number of output digits written.
    pub fn run(self, transform_len: usize) -> usize {
        let Self {
            dst_digits,
            a,
            b,
            convolution_len,
            digit_bits,
            scratch_u64,
            executor,
        } = self;
        let (left, scratch_tail) = scratch_u64.split_at_mut(transform_len);
        let (right, twiddles) = scratch_tail.split_at_mut(transform_len);
        left.fill(0);
        right.fill(0);
        for (dst, src) in left.iter_mut().zip(a) {
            *dst = u64::from(*src);
        }
        for (dst, src) in right.iter_mut().zip(b) {
            *dst = u64::from(*src);
        }
        forward_transform_pair(left, right, twiddles, executor);

        pointwise_mul(left, right, executor);

        inverse_transform(left, twiddles, executor);
        // SAFETY: convolution_len <= transform_len.
        let active_left = unsafe { left.get_unchecked(..convolution_len) };
        coefficients_to_digits_into(dst_digits, active_left, digit_bits)
    }
}

impl Ntt {
    pub fn square_digits_into<E: ParallelExecutor>(
        dst_digits: &mut [u32],
        a: &[u32],
        convolution_len: usize,
        transform_len: usize,
        digit_bits: u32,
        scratch_u64: &mut [u64],
        executor: &E,
    ) -> usize {
        let (values, scratch_tail) = scratch_u64.split_at_mut(transform_len);
        let (twiddles, _) = scratch_tail.split_at_mut(transform_len);
        values.fill(0);
        for (dst, src) in values.iter_mut().zip(a) {
            *dst = u64::from(*src);
        }
        forward_transform_single(values, twiddles, executor);
        pointwise_square(values, executor);
        inverse_transform(values, twiddles, executor);
        // SAFETY: convolution_len <= transform_len.
        let active = unsafe { values.get_unchecked(..convolution_len) };
        coefficients_to_digits_into(dst_digits, active, digit_bits)
    }
}

fn forward_transform_pair<E: ParallelExecutor>(
    left: &mut [u64],
    right: &mut [u64],
    twiddles: &mut [u64],
    executor: &E,
) {
    let policy = NttExecutionPolicy::for_executor(executor);
    let mut block_len = left.len();
    while block_len >= 2 {
        // SAFETY: the validated transform length fits in u64.
        let exponent = PRIME
            .wrapping_sub(1)
            .div_euclid(unsafe { u64::try_from(block_len).unwrap_unchecked() });
        let block_root = pow_mod(PRIMITIVE_ROOT, exponent);
        let half_len = block_len >> 1;
        generate_twiddles(twiddles, half_len, block_root);
        let stage_twiddles: &[u64] = twiddles;
        if policy.should_split(left.len()) {
            executor.join(
                || apply_forward_stage(left, block_len, stage_twiddles, policy, executor),
                || apply_forward_stage(right, block_len, stage_twiddles, policy, executor),
            );
        } else {
            apply_forward_stage(left, block_len, stage_twiddles, policy, executor);
            apply_forward_stage(right, block_len, stage_twiddles, policy, executor);
        }
        block_len >>= 1;
    }
}

fn forward_transform_single<E: ParallelExecutor>(
    values: &mut [u64],
    twiddles: &mut [u64],
    executor: &E,
) {
    let policy = NttExecutionPolicy::for_executor(executor);
    let mut block_len = values.len();
    while block_len >= 2 {
        // SAFETY: block_len is bounded by the transform size; usize ≤ u64 on all targets.
        let exponent = PRIME
            .wrapping_sub(1)
            .div_euclid(unsafe { u64::try_from(block_len).unwrap_unchecked() });
        let block_root = pow_mod(PRIMITIVE_ROOT, exponent);
        let half_len = block_len >> 1;
        generate_twiddles(twiddles, half_len, block_root);
        apply_forward_stage(values, block_len, twiddles, policy, executor);
        block_len >>= 1;
    }
}

fn apply_forward_stage<E: ParallelExecutor>(
    values: &mut [u64],
    block_len: usize,
    twiddles: &[u64],
    policy: NttExecutionPolicy,
    executor: &E,
) {
    let blocks = values.len().div_euclid(block_len);
    if !policy.should_split(values.len()) {
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(block_len >> 1);
            for ((low_value, high_value), &twiddle) in low.iter_mut().zip(high).zip(twiddles.iter())
            {
                let lower = *low_value;
                let upper = *high_value;
                *low_value = add_mod(lower, upper);
                *high_value = mul_mod(sub_mod(lower, upper), twiddle);
            }
        }
        return;
    }
    if blocks == 1 {
        let half_len = block_len >> 1;
        let (low, high) = values.split_at_mut(half_len);
        apply_butterfly_range(
            low,
            high,
            twiddles,
            policy,
            executor,
            ButterflyDirection::Forward,
        );
        return;
    }
    // `blocks * block_len == values.len()` was established above; halving the
    // block count therefore keeps this product within the mutable slice.
    let split_len = blocks.div_euclid(2).wrapping_mul(block_len);
    let (left, right) = values.split_at_mut(split_len);
    executor.join(
        || apply_forward_stage(left, block_len, twiddles, policy, executor),
        || apply_forward_stage(right, block_len, twiddles, policy, executor),
    );
}

fn inverse_transform<E: ParallelExecutor>(values: &mut [u64], twiddles: &mut [u64], executor: &E) {
    let policy = NttExecutionPolicy::for_executor(executor);
    let inverse_root = pow_mod(PRIMITIVE_ROOT, PRIME.wrapping_sub(2));
    let mut block_len = 2;
    while block_len <= values.len() {
        // SAFETY: block_len is bounded by the transform size; usize ≤ u64 on all targets.
        let exponent = PRIME
            .wrapping_sub(1)
            .div_euclid(unsafe { u64::try_from(block_len).unwrap_unchecked() });
        let block_root = pow_mod(inverse_root, exponent);
        let half_len = block_len >> 1;
        generate_twiddles(twiddles, half_len, block_root);
        apply_inverse_stage(values, block_len, twiddles, policy, executor);
        block_len = block_len.wrapping_mul(2);
    }
    // SAFETY: values.len() is the transform length, usize ≤ u64 on all targets.
    let inverse_len = pow_mod(
        unsafe { u64::try_from(values.len()).unwrap_unchecked() },
        PRIME.wrapping_sub(2),
    );
    scale_inverse(values, inverse_len, policy, executor);
}

fn apply_inverse_stage<E: ParallelExecutor>(
    values: &mut [u64],
    block_len: usize,
    twiddles: &[u64],
    policy: NttExecutionPolicy,
    executor: &E,
) {
    let blocks = values.len().div_euclid(block_len);
    if !policy.should_split(values.len()) {
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(block_len >> 1);
            for ((low_value, high_value), &twiddle) in low.iter_mut().zip(high).zip(twiddles.iter())
            {
                let upper = mul_mod(*high_value, twiddle);
                let lower = *low_value;
                *low_value = add_mod(lower, upper);
                *high_value = sub_mod(lower, upper);
            }
        }
        return;
    }
    if blocks == 1 {
        let half_len = block_len >> 1;
        let (low, high) = values.split_at_mut(half_len);
        apply_butterfly_range(
            low,
            high,
            twiddles,
            policy,
            executor,
            ButterflyDirection::Inverse,
        );
        return;
    }
    // `blocks * block_len == values.len()` was established above; halving the
    // block count therefore keeps this product within the mutable slice.
    let split_len = blocks.div_euclid(2).wrapping_mul(block_len);
    let (left, right) = values.split_at_mut(split_len);
    executor.join(
        || apply_inverse_stage(left, block_len, twiddles, policy, executor),
        || apply_inverse_stage(right, block_len, twiddles, policy, executor),
    );
}

fn apply_butterfly_range<E: ParallelExecutor>(
    low: &mut [u64],
    high: &mut [u64],
    twiddles: &[u64],
    policy: NttExecutionPolicy,
    executor: &E,
    direction: ButterflyDirection,
) {
    if policy.should_split(low.len()) {
        let midpoint = low.len().div_euclid(2);
        let (low_left, low_right) = low.split_at_mut(midpoint);
        let (high_left, high_right) = high.split_at_mut(midpoint);
        let (twiddle_left, twiddle_right) = twiddles.split_at(midpoint);
        executor.join(
            || {
                apply_butterfly_range(
                    low_left,
                    high_left,
                    twiddle_left,
                    policy,
                    executor,
                    direction,
                );
            },
            || {
                apply_butterfly_range(
                    low_right,
                    high_right,
                    twiddle_right,
                    policy,
                    executor,
                    direction,
                );
            },
        );
        return;
    }
    for ((low_value, high_value), &twiddle) in low.iter_mut().zip(high).zip(twiddles) {
        let lower = *low_value;
        let upper = match direction {
            ButterflyDirection::Forward => *high_value,
            ButterflyDirection::Inverse => mul_mod(*high_value, twiddle),
        };
        *low_value = add_mod(lower, upper);
        *high_value = match direction {
            ButterflyDirection::Forward => mul_mod(sub_mod(lower, upper), twiddle),
            ButterflyDirection::Inverse => sub_mod(lower, upper),
        };
    }
}

fn pointwise_mul<E: ParallelExecutor>(left: &mut [u64], right: &[u64], executor: &E) {
    let policy = NttExecutionPolicy::for_executor(executor);
    apply_pointwise_mul(left, right, policy, executor);
}

fn apply_pointwise_mul<E: ParallelExecutor>(
    left: &mut [u64],
    right: &[u64],
    policy: NttExecutionPolicy,
    executor: &E,
) {
    if !policy.should_split(left.len()) {
        for (left_value, &right_value) in left.iter_mut().zip(right) {
            *left_value = mul_mod(*left_value, right_value);
        }
        return;
    }
    let midpoint = left.len().div_euclid(2);
    let (left_low, left_high) = left.split_at_mut(midpoint);
    let (right_low, right_high) = right.split_at(midpoint);
    executor.join(
        || apply_pointwise_mul(left_low, right_low, policy, executor),
        || apply_pointwise_mul(left_high, right_high, policy, executor),
    );
}

fn pointwise_square<E: ParallelExecutor>(values: &mut [u64], executor: &E) {
    let policy = NttExecutionPolicy::for_executor(executor);
    apply_pointwise_square(values, policy, executor);
}

fn apply_pointwise_square<E: ParallelExecutor>(
    values: &mut [u64],
    policy: NttExecutionPolicy,
    executor: &E,
) {
    if !policy.should_split(values.len()) {
        for value in values {
            *value = mul_mod(*value, *value);
        }
        return;
    }
    let midpoint = values.len().div_euclid(2);
    let (left, right) = values.split_at_mut(midpoint);
    executor.join(
        || apply_pointwise_square(left, policy, executor),
        || apply_pointwise_square(right, policy, executor),
    );
}

fn scale_inverse<E: ParallelExecutor>(
    values: &mut [u64],
    inverse_len: u64,
    policy: NttExecutionPolicy,
    executor: &E,
) {
    if !policy.should_split(values.len()) {
        for value in values {
            *value = mul_mod(*value, inverse_len);
        }
        return;
    }
    let midpoint = values.len().div_euclid(2);
    let (left, right) = values.split_at_mut(midpoint);
    executor.join(
        || scale_inverse(left, inverse_len, policy, executor),
        || scale_inverse(right, inverse_len, policy, executor),
    );
}

fn generate_twiddles(twiddles: &mut [u64], half_len: usize, block_root: u64) {
    let mut twiddle = 1;
    for output in twiddles.iter_mut().take(half_len) {
        *output = twiddle;
        twiddle = mul_mod(twiddle, block_root);
    }
}

fn coefficients_to_digits_into(dst: &mut [u32], coefficients: &[u64], digit_bits: u32) -> usize {
    let mask = (1_u128 << digit_bits).wrapping_sub(1);
    let mut carry = 0_u128;
    let mut count = 0_usize;
    for (&coefficient, out) in coefficients.iter().zip(dst.iter_mut()) {
        let with_carry = u128::from(coefficient).wrapping_add(carry);
        // SAFETY: mask ensures the result fits in u32; digit_bits <= 31 for Goldilocks.
        *out = unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() };
        carry = with_carry >> digit_bits;
        count = count.wrapping_add(1);
    }
    while carry != 0 && count < dst.len() {
        if let Some(out) = dst.get_mut(count) {
            // SAFETY: mask ensures the result fits in u32.
            *out = unsafe { u32::try_from(carry & mask).unwrap_unchecked() };
            carry >>= digit_bits;
            count = count.wrapping_add(1);
        }
    }
    count
}

fn pow_mod(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, base);
        }
        base = mul_mod(base, base);
        exponent >>= 1;
    }
    result
}

#[allow(clippy::inline_always, reason = "Critical in hot NTT butterfly loop")]
#[inline(always)]
const fn add_mod(a: u64, b: u64) -> u64 {
    let (sum, overflowed) = a.overflowing_add(b);
    if overflowed {
        sum.wrapping_add(EPSILON)
    } else if sum >= PRIME {
        sum.wrapping_sub(PRIME)
    } else {
        sum
    }
}

#[allow(clippy::inline_always, reason = "Critical in hot NTT butterfly loop")]
#[inline(always)]
const fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a.wrapping_sub(b)
    } else {
        a.wrapping_sub(b).wrapping_sub(EPSILON)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The casts extract the exact low and high 64-bit halves of a u128 product"
)]
#[allow(
    clippy::inline_always,
    reason = "Critical for Goldilocks NTT butterfly reduction performance"
)]
#[inline(always)]
fn mul_mod(a: u64, b: u64) -> u64 {
    let product = u128::from(a).wrapping_mul(u128::from(b));
    let low = product as u64;
    let high = (product >> 64) as u64;
    let high_low = high & EPSILON;
    let high_high = high >> 32;
    let (low_minus_high, borrowed) = low.overflowing_sub(high_high);
    let folded_low = if borrowed {
        low_minus_high.wrapping_sub(EPSILON)
    } else {
        low_minus_high
    };
    let high_low_mul = (high_low << 32).wrapping_sub(high_low);
    add_mod(folded_low, high_low_mul)
}

#[cfg(test)]
#[path = "../tests/kernels/goldilocks.rs"]
mod tests;
