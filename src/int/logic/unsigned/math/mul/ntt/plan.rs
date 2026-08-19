//! The multi-prime NTT planner and modulus configuration.

use crate::parallel::ParallelExecutor;

use super::{LIMB_BITS, Ntt, PRIME_U128};

// Conservative, unmeasured fallback until the generated tuning profile grows
// an executor/cache-aware NTT grain field. Keep this value centralized so the
// generated profile can replace it without scattering thresholds in kernels.
const NTT_PARALLEL_GRAIN: usize = 2_048;

/// Centralized scheduling policy for cache-contiguous NTT leaves.
#[derive(Clone, Copy, Debug)]
pub struct NttExecutionPolicy {
    workers: usize,
    grain: usize,
}

impl NttExecutionPolicy {
    /// Builds a policy from the executor's scheduling capacity.
    #[must_use]
    pub fn for_executor<E: ParallelExecutor>(executor: &E) -> Self {
        let workers = executor.parallelism().get();
        Self {
            workers,
            grain: NTT_PARALLEL_GRAIN,
        }
    }

    #[must_use]
    pub const fn should_split(self, work: usize) -> bool {
        self.workers > 1 && work >= self.grain && work >= self.workers.saturating_mul(2)
    }
}

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

impl TransformPlan {
    /// Whether this transform plan has valid parameters for the NTT engine.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.digit_bits > 0
            && self.digit_bits <= 31
            && self.modulus_count >= 1
            && self.modulus_count <= MODULI.len()
    }
}

impl Ntt {
    pub const ONE_PRIME_DIGIT_BITS: [u32; 9] = [23, 22, 21, 20, 19, 18, 17, 16, 15];
    pub const TWO_PRIME_DIGIT_BITS: [u32; 6] = [20, 19, 18, 17, 16, 15];
    pub const THREE_PRIME_DIGIT_BITS: u32 = 31;
    // The fixed roots support at most 2^26 points. `None` is the correct cap on a
    // narrower target because every representable allocation is then below 2^26.
    pub const MAX_TRANSFORM_LEN: Option<usize> = 1_usize.checked_shl(26);

    pub fn choose_transform_plan(len_a: usize, len_b: usize) -> Option<TransformPlan> {
        let three_prime_len =
            Self::estimated_transform_len(len_a, len_b, Self::THREE_PRIME_DIGIT_BITS)?;
        let three_prime_work = three_prime_len.checked_mul(3)?;
        let mut best_work = three_prime_work;
        let mut best_bits = Self::THREE_PRIME_DIGIT_BITS;
        let mut best_count = 3;
        for digit_bits in Self::ONE_PRIME_DIGIT_BITS {
            let transform_len = Self::estimated_transform_len(len_a, len_b, digit_bits)?;
            if transform_len < best_work
                && Self::coefficient_range_fits(transform_len, digit_bits, 1)
            {
                best_work = transform_len;
                best_bits = digit_bits;
                best_count = 1;
            }
        }
        for digit_bits in Self::TWO_PRIME_DIGIT_BITS {
            let transform_len = Self::estimated_transform_len(len_a, len_b, digit_bits)?;
            let transform_work = transform_len.checked_mul(2)?;
            if transform_work < best_work
                && Self::coefficient_range_fits(transform_len, digit_bits, 2)
            {
                best_work = transform_work;
                best_bits = digit_bits;
                best_count = 2;
            }
        }
        Some(TransformPlan {
            digit_bits: best_bits,
            modulus_count: best_count,
        })
    }

    pub fn estimated_transform_len(len_a: usize, len_b: usize, digit_bits: u32) -> Option<usize> {
        let digit_bits_usize = usize::try_from(digit_bits).ok()?;
        if digit_bits_usize == 0 {
            return None;
        }
        let digits_a = len_a.checked_mul(LIMB_BITS)?.div_ceil(digit_bits_usize);
        let digits_b = len_b.checked_mul(LIMB_BITS)?.div_ceil(digit_bits_usize);
        digits_a
            .checked_add(digits_b)
            .and_then(|sum| sum.checked_sub(1))?
            .checked_next_power_of_two()
    }

    pub fn coefficient_range_fits(transform_len: usize, digit_bits: u32, count: usize) -> bool {
        if digit_bits == 0 || digit_bits >= 128 || !(1..=MODULI.len()).contains(&count) {
            return false;
        }
        let digit_bound = (1_u128 << digit_bits).wrapping_sub(1);
        // A convolution coefficient contains at most transform_len products, each
        // at most (base-1)^2. These configured bounds stay below 2^90.
        let Some(transform_len_u128) = u128::try_from(transform_len).ok() else {
            return false;
        };
        let Some(coefficient_bound) = transform_len_u128
            .checked_mul(digit_bound)
            .and_then(|bound| bound.checked_mul(digit_bound))
        else {
            return false;
        };
        let mut crt_range = if count == 1 { PRIME_U128 } else { 1_u128 };
        if count > 1 {
            for modulus in MODULI.iter().take(count) {
                let Some(next_range) = crt_range.checked_mul(u128::from(modulus.prime)) else {
                    return false;
                };
                crt_range = next_range;
            }
        }
        coefficient_bound < crt_range
    }
}
