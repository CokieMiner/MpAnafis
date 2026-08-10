//! The multi-prime NTT driver: split, forward transforms, pointwise, inverse.
use alloc::{vec, vec::Vec};

use super::{LIMB_BITS, Limb, PRIME_U128};

// The planner minimizes transform work across one Goldilocks prime, two 31-bit
// primes, or three 31-bit primes while proving the coefficient range exactly.
const ONE_PRIME_DIGIT_BITS: [u32; 9] = [23, 22, 21, 20, 19, 18, 17, 16, 15];
const TWO_PRIME_DIGIT_BITS: [u32; 6] = [20, 19, 18, 17, 16, 15];
const THREE_PRIME_DIGIT_BITS: u32 = 31;
// The fixed roots support at most 2^26 points. `None` is the correct cap on a
// narrower target because every representable allocation is then below 2^26.
const MAX_TRANSFORM_LEN: Option<usize> = 1_usize.checked_shl(26);

pub const MODULI: [Modulus; 3] = [
    Modulus {
        prime: 2_013_265_921,
        primitive_root: 31,
        neg_inverse: 2_013_265_919,
        radix_squared: 1_172_168_163,
    },
    Modulus {
        prime: 1_811_939_329,
        primitive_root: 13,
        neg_inverse: 1_811_939_327,
        radix_squared: 959_408_210,
    },
    Modulus {
        prime: 469_762_049,
        primitive_root: 3,
        neg_inverse: 469_762_047,
        radix_squared: 460_175_152,
    },
];

#[derive(Clone, Copy, Debug)]
pub struct Modulus {
    pub prime: u32,
    pub primitive_root: u32,
    pub neg_inverse: u32,
    pub radix_squared: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TransformPlan {
    pub digit_bits: u32,
    pub modulus_count: usize,
}

/// Namespace for the exact multi-prime number-theoretic transform tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ntt;

impl Ntt {
    /// Whether [`Self::try_mul`] can compute a product of these operand widths.
    ///
    /// The capability counterpart of [`ssa_admits_mul`](super::super::ssa_admits_mul):
    /// it reports whether the fixed prime set carries a transform long enough for
    /// this product, and says nothing about whether the transform is the fastest
    /// tier. Empty operands are accepted because the product is then a fill.
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        if len_a == 0 || len_b == 0 {
            return true;
        }
        choose_transform_plan(len_a, len_b).is_some_and(|plan| {
            estimated_transform_len(len_a, len_b, plan.digit_bits).is_some_and(|transform_len| {
                MAX_TRANSFORM_LEN.is_none_or(|max_len| transform_len <= max_len)
                    && coefficient_range_fits(transform_len, plan.digit_bits, plan.modulus_count)
            })
        })
    }

    /// Multiply two limb slices using an exact multi-prime NTT convolution.
    ///
    /// Returns `false` only when the required transform exceeds the roots supported
    /// by the fixed prime set; callers can then retain the preceding Toom tier.
    pub fn try_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        if a.is_empty() || b.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(plan) = choose_transform_plan(a.len(), b.len()) else {
            return false;
        };
        Self::try_mul_with_plan(dst, a, b, plan)
    }

    /// Multiply with an explicitly selected digit width and modulus count.
    ///
    /// This entry point is re-exported only by the feature-gated benchmark facade;
    /// production dispatch always uses [`choose_transform_plan`].
    #[cfg(feature = "_internal-tune")]
    pub fn try_mul_with_forced_plan(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        digit_bits: u32,
        modulus_count: usize,
    ) -> bool {
        if !(1..=31).contains(&digit_bits) || !(1..=MODULI.len()).contains(&modulus_count) {
            return false;
        }
        Self::try_mul_with_plan(
            dst,
            a,
            b,
            TransformPlan {
                digit_bits,
                modulus_count,
            },
        )
    }

    pub fn try_mul_with_plan(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        plan: TransformPlan,
    ) -> bool {
        if a.is_empty() || b.is_empty() {
            dst.fill(0);
            return true;
        }
        let digits_a = limbs_to_digits(a, plan.digit_bits);
        let digits_b = limbs_to_digits(b, plan.digit_bits);
        if digits_a.is_empty() || digits_b.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(convolution_len) = digits_a
            .len()
            .checked_add(digits_b.len())
            .and_then(|sum| sum.checked_sub(1))
        else {
            return false;
        };
        let Some(transform_len) = convolution_len.checked_next_power_of_two() else {
            return false;
        };
        if MAX_TRANSFORM_LEN.is_some_and(|max_len| transform_len > max_len)
            || !coefficient_range_fits(transform_len, plan.digit_bits, plan.modulus_count)
        {
            return false;
        }

        if plan.modulus_count == 1 {
            let digits = Self::multiply_digits(
                &digits_a,
                &digits_b,
                convolution_len,
                transform_len,
                plan.digit_bits,
            );
            digits_to_limbs(dst, &digits, plan.digit_bits);
            return true;
        }

        let mut right_scratch = vec![0; transform_len];
        let mut first = Vec::with_capacity(transform_len);
        convolve_mod_into(
            &mut first,
            &mut right_scratch,
            &digits_a,
            &digits_b,
            convolution_len,
            transform_len,
            MODULI[0],
        );
        let mut second = Vec::with_capacity(transform_len);
        convolve_mod_into(
            &mut second,
            &mut right_scratch,
            &digits_a,
            &digits_b,
            convolution_len,
            transform_len,
            MODULI[1],
        );
        if plan.modulus_count == 2 {
            Self::reconstruct_two_into(&mut right_scratch, &first, &second, plan.digit_bits);
        } else {
            let mut third = Vec::with_capacity(transform_len);
            convolve_mod_into(
                &mut third,
                &mut right_scratch,
                &digits_a,
                &digits_b,
                convolution_len,
                transform_len,
                MODULI[2],
            );
            Self::reconstruct_three_into(
                &mut right_scratch,
                &first,
                &second,
                &third,
                plan.digit_bits,
            );
        }
        digits_to_limbs(dst, &right_scratch, plan.digit_bits);
        true
    }
}

fn choose_transform_plan(len_a: usize, len_b: usize) -> Option<TransformPlan> {
    let three_prime_len = estimated_transform_len(len_a, len_b, THREE_PRIME_DIGIT_BITS)?;
    let three_prime_work = three_prime_len.checked_mul(3)?;
    let mut best_work = three_prime_work;
    let mut best_bits = THREE_PRIME_DIGIT_BITS;
    let mut best_count = 3;
    for digit_bits in ONE_PRIME_DIGIT_BITS {
        let transform_len = estimated_transform_len(len_a, len_b, digit_bits)?;
        if transform_len < best_work && coefficient_range_fits(transform_len, digit_bits, 1) {
            best_work = transform_len;
            best_bits = digit_bits;
            best_count = 1;
        }
    }
    let mut best_two_bits = 0;
    for digit_bits in TWO_PRIME_DIGIT_BITS {
        let transform_len = estimated_transform_len(len_a, len_b, digit_bits)?;
        let transform_work = transform_len.checked_mul(2)?;
        if transform_work < best_work && coefficient_range_fits(transform_len, digit_bits, 2) {
            best_work = transform_work;
            best_two_bits = digit_bits;
        }
    }
    if best_two_bits != 0 {
        best_bits = best_two_bits;
        best_count = 2;
    }
    Some(TransformPlan {
        digit_bits: best_bits,
        modulus_count: best_count,
    })
}

fn estimated_transform_len(len_a: usize, len_b: usize, digit_bits: u32) -> Option<usize> {
    // SAFETY: digit_bits is a configured u32 constant, always fits in usize.
    let digit_bits_usize = unsafe { usize::try_from(digit_bits).unwrap_unchecked() };
    let digits_a = len_a.checked_mul(LIMB_BITS)?.div_ceil(digit_bits_usize);
    let digits_b = len_b.checked_mul(LIMB_BITS)?.div_ceil(digit_bits_usize);
    digits_a
        .checked_add(digits_b)
        .and_then(|sum| sum.checked_sub(1))?
        .checked_next_power_of_two()
}

fn coefficient_range_fits(transform_len: usize, digit_bits: u32, count: usize) -> bool {
    let digit_bound = (1_u128 << digit_bits).wrapping_sub(1);
    // A convolution coefficient contains at most transform_len products, each
    // at most (base-1)^2. These configured bounds stay below 2^90.
    // SAFETY: transform_len ≤ 2^26, fits in u128.
    let coefficient_bound = unsafe { u128::try_from(transform_len).unwrap_unchecked() }
        .wrapping_mul(digit_bound)
        .wrapping_mul(digit_bound);
    let mut crt_range = if count == 1 { PRIME_U128 } else { 1_u128 };
    if count > 1 {
        for modulus in MODULI.iter().take(count) {
            crt_range = crt_range.wrapping_mul(u128::from(modulus.prime));
        }
    }
    coefficient_bound < crt_range
}

fn convolve_mod_into(
    left: &mut Vec<u32>,
    right: &mut [u32],
    a: &[u32],
    b: &[u32],
    convolution_len: usize,
    transform_len: usize,
    modulus: Modulus,
) {
    left.clear();
    left.resize(transform_len, 0);
    right.fill(0);
    for (dst, src) in left.iter_mut().zip(a) {
        *dst = src.rem_euclid(modulus.prime);
    }
    for (dst, src) in right.iter_mut().zip(b) {
        *dst = src.rem_euclid(modulus.prime);
    }
    forward_transform_pair(left, right, modulus);
    for (left_value, right_value) in left.iter_mut().zip(right.iter()) {
        *left_value = montgomery_mul(*left_value, *right_value, modulus);
    }
    inverse_transform(left, modulus);
    left.truncate(convolution_len);
}

fn forward_transform_pair(left: &mut [u32], right: &mut [u32], modulus: Modulus) {
    for (left_value, right_value) in left.iter_mut().zip(right.iter_mut()) {
        *left_value = to_montgomery(*left_value, modulus);
        *right_value = to_montgomery(*right_value, modulus);
    }
    let root = to_montgomery(modulus.primitive_root, modulus);
    // DIF emits bit-reversed frequency order. Both operands use the same
    // order, so pointwise products align and the inverse DIT can consume that
    // order directly without any permutation pass.
    let mut block_len = left.len();
    while block_len >= 2 {
        // SAFETY: block_len is bounded by transform length ≤ 2^28, fits in u32.
        let exponent = modulus
            .prime
            .wrapping_sub(1)
            .div_euclid(unsafe { u32::try_from(block_len).unwrap_unchecked() });
        let block_root = montgomery_pow(root, exponent, modulus);
        let half_len = block_len >> 1;
        for (left_block, right_block) in left
            .chunks_exact_mut(block_len)
            .zip(right.chunks_exact_mut(block_len))
        {
            let (left_low, left_high) = left_block.split_at_mut(half_len);
            let (right_low, right_high) = right_block.split_at_mut(half_len);
            let mut twiddle = to_montgomery(1, modulus);
            for ((left_low_value, left_high_value), (right_low_value, right_high_value)) in left_low
                .iter_mut()
                .zip(left_high)
                .zip(right_low.iter_mut().zip(right_high))
            {
                let left_lower = *left_low_value;
                let left_upper = *left_high_value;
                *left_low_value = add_mod(left_lower, left_upper, modulus.prime);
                *left_high_value = montgomery_mul(
                    sub_mod(left_lower, left_upper, modulus.prime),
                    twiddle,
                    modulus,
                );

                let right_lower = *right_low_value;
                let right_upper = *right_high_value;
                *right_low_value = add_mod(right_lower, right_upper, modulus.prime);
                *right_high_value = montgomery_mul(
                    sub_mod(right_lower, right_upper, modulus.prime),
                    twiddle,
                    modulus,
                );
                twiddle = montgomery_mul(twiddle, block_root, modulus);
            }
        }
        block_len >>= 1;
    }
}

fn inverse_transform(values: &mut [u32], modulus: Modulus) {
    let root = montgomery_pow(
        to_montgomery(modulus.primitive_root, modulus),
        modulus.prime.wrapping_sub(2),
        modulus,
    );
    let mut block_len = 2;
    while block_len <= values.len() {
        // SAFETY: block_len is bounded by transform length ≤ 2^28, fits in u32.
        let exponent = modulus
            .prime
            .wrapping_sub(1)
            .div_euclid(unsafe { u32::try_from(block_len).unwrap_unchecked() });
        let block_root = montgomery_pow(root, exponent, modulus);
        let half_len = block_len >> 1;
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(half_len);
            let mut twiddle = to_montgomery(1, modulus);
            for (low_value, high_value) in low.iter_mut().zip(high) {
                let upper = montgomery_mul(*high_value, twiddle, modulus);
                let lower = *low_value;
                *low_value = add_mod(lower, upper, modulus.prime);
                *high_value = sub_mod(lower, upper, modulus.prime);
                twiddle = montgomery_mul(twiddle, block_root, modulus);
            }
        }
        block_len = block_len.wrapping_mul(2);
    }
    let inverse_len = montgomery_pow(
        to_montgomery(
            // SAFETY: values.len() ≤ transform length ≤ 2^28, always fits in u32.
            unsafe { u32::try_from(values.len()).unwrap_unchecked() },
            modulus,
        ),
        modulus.prime.wrapping_sub(2),
        modulus,
    );
    for value in values {
        *value = montgomery_mul(montgomery_mul(*value, inverse_len, modulus), 1, modulus);
    }
}

fn limbs_to_digits(limbs: &[Limb], digit_bits: u32) -> Vec<u32> {
    let digit_mask = (1_u128 << digit_bits).wrapping_sub(1);
    // SAFETY: digit_bits is a configured u32 constant ≤ usize::BITS, always fits.
    let capacity = limbs
        .len()
        .wrapping_mul(LIMB_BITS)
        .div_ceil(unsafe { usize::try_from(digit_bits).unwrap_unchecked() });
    let mut digits = Vec::with_capacity(capacity);
    let mut accumulator = 0_u128;
    let mut available_bits = 0_u32;
    for limb in limbs {
        // SAFETY: Limb (u64/u32/u16) always fits in u128.
        accumulator |= unsafe { u128::try_from(*limb).unwrap_unchecked() } << available_bits;
        available_bits = available_bits.wrapping_add(Limb::BITS);
        while available_bits >= digit_bits {
            // SAFETY: mask limits the result to digit_bits ≤ 32, always fits in u32.
            digits.push(unsafe { u32::try_from(accumulator & digit_mask).unwrap_unchecked() });
            accumulator >>= digit_bits;
            available_bits = available_bits.wrapping_sub(digit_bits);
        }
    }
    if available_bits != 0 {
        // SAFETY: available_bits < digit_bits ≤ 32, so accumulator < 2^32, fits in u32.
        digits.push(unsafe { u32::try_from(accumulator).unwrap_unchecked() });
    }
    while digits.last().copied() == Some(0) {
        let _ = digits.pop();
    }
    digits
}

fn digits_to_limbs(dst: &mut [Limb], digits: &[u32], digit_bits: u32) {
    dst.fill(0);
    let limb_mask = (1_u128 << Limb::BITS).wrapping_sub(1);
    let mut limbs = dst.iter_mut();
    let mut accumulator = 0_u128;
    let mut available_bits = 0_u32;
    for digit in digits {
        accumulator |= u128::from(*digit) << available_bits;
        available_bits = available_bits.wrapping_add(digit_bits);
        if available_bits >= Limb::BITS {
            if let Some(limb) = limbs.next() {
                // SAFETY: limb_mask masks to Limb::BITS, always fits in Limb.
                *limb = unsafe { Limb::try_from(accumulator & limb_mask).unwrap_unchecked() };
            }
            accumulator >>= Limb::BITS;
            available_bits = available_bits.wrapping_sub(Limb::BITS);
        }
    }
    if available_bits != 0
        && let Some(limb) = limbs.next()
    {
        // SAFETY: available_bits < Limb::BITS, so accumulator < 2^Limb::BITS, fits in Limb.
        *limb = unsafe { Limb::try_from(accumulator).unwrap_unchecked() };
    }
}

fn montgomery_pow(mut base: u32, mut exponent: u32, modulus: Modulus) -> u32 {
    let mut result = to_montgomery(1, modulus);
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = montgomery_mul(result, base, modulus);
        }
        base = montgomery_mul(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn to_montgomery(value: u32, modulus: Modulus) -> u32 {
    montgomery_mul(value, modulus.radix_squared, modulus)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Montgomery reduction intentionally extracts the low radix word; REDC then proves the quotient is below 2p < 2^32"
)]
fn montgomery_mul(a: u32, b: u32, modulus: Modulus) -> u32 {
    let product = u64::from(a).wrapping_mul(u64::from(b));
    let factor = (product as u32).wrapping_mul(modulus.neg_inverse);
    let reduced = product
        .wrapping_add(u64::from(factor).wrapping_mul(u64::from(modulus.prime)))
        .wrapping_shr(32);
    // Montgomery REDC yields a value below 2p; p < 2^31 therefore proves
    // reduced < 2^32 and the conversion is exact.
    let reduced_u32 = reduced as u32;
    if reduced_u32 >= modulus.prime {
        reduced_u32.wrapping_sub(modulus.prime)
    } else {
        reduced_u32
    }
}

const fn add_mod(a: u32, b: u32, modulus: u32) -> u32 {
    let sum = a.wrapping_add(b);
    if sum >= modulus {
        sum.wrapping_sub(modulus)
    } else {
        sum
    }
}

const fn sub_mod(a: u32, b: u32, modulus: u32) -> u32 {
    if a >= b {
        a.wrapping_sub(b)
    } else {
        modulus.wrapping_sub(b.wrapping_sub(a))
    }
}
