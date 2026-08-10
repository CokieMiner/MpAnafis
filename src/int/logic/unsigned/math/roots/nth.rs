//! `n`th root by Newton iteration, seeded recursively.

#![allow(
    unsafe_code,
    reason = "the single-limb fast path indexes a slice whose length the caller has already checked"
)]

use core::{cmp::Ordering, mem::swap};

use super::{Division, InternalArbiUint, LIMB_BITS, Limb, NthRootScratch};

impl NthRootScratch {
    /// Computes a single-limb root with native integer arithmetic.
    pub fn nth_root_single_limb(a: &InternalArbiUint, n: u32) -> InternalArbiUint {
        let limbs = a.limbs();
        // SAFETY: caller guarantees limbs.len() == 1
        let val = unsafe { *limbs.get_unchecked(0) };
        if val <= 1 {
            return a.clone();
        }
        let val_u64 = u64::try_from(val).expect("a usize limb always fits in u64");

        let bits = 64_u32.wrapping_sub(val_u64.leading_zeros());
        let init_shift = initial_shift(usize::try_from(bits).unwrap_or(usize::MAX), n);
        let mut estimate = if init_shift < 63 {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "init_shift is bounded by bits <= 64, which fits safely in u32"
            )]
            1_u64.wrapping_shl(init_shift as u32)
        } else {
            1_u64.wrapping_shl(63)
        };

        let n_u64 = u64::from(n);
        let n_minus_1 = n.wrapping_sub(1);
        let n_minus_1_u64 = u64::from(n_minus_1);

        loop {
            let power = estimate.checked_pow(n_minus_1);
            let Some(pow_n_minus_1) = power else {
                // estimate^(n-1) overflowed u64, so estimate is too large
                estimate = estimate.wrapping_shr(1);
                continue;
            };
            // SAFETY: pow_n_minus_1 > 0 since estimate >= 1
            let quotient = unsafe { val_u64.checked_div(pow_n_minus_1).unwrap_unchecked() };
            let scaled = n_minus_1_u64.wrapping_mul(estimate);
            let sum = scaled.wrapping_add(quotient);
            // SAFETY: n_u64 >= 2
            let next = unsafe { sum.checked_div(n_u64).unwrap_unchecked() };
            if next >= estimate {
                let estimate_limb =
                    Limb::try_from(estimate).expect("the root of a limb fits in one limb");
                return InternalArbiUint::from_limb(estimate_limb);
            }
            estimate = next;
        }
    }

    /// Computes a general multi-limb `n`th root.
    ///
    /// Callers must pass `n >= 2` and a value that is neither zero nor one.
    pub fn nth_root_multi_limb(&mut self, a: &InternalArbiUint, n: u32) -> InternalArbiUint {
        let one = InternalArbiUint::one();
        let n_uint = InternalArbiUint::from_u64(u64::from(n));
        let n_minus_1 = n_uint.sub(&one);

        let estimate = seed_estimate(a, n, self);

        let cap = a.limbs().len().wrapping_add(1);
        self.x_pow_n_minus_1.reserve(cap);
        self.temp_prod.reserve(cap);
        self.quotient.reserve(cap);
        self.rem.reserve(cap);
        self.scaled_estimate.reserve(cap);
        self.sum.reserve(cap);
        self.next_estimate.reserve(cap);
        self.base_pow.reserve(cap);

        let mut estimate = estimate;
        loop {
            if bounded_pow_into(&estimate, n.wrapping_sub(1), a, self) {
                Division::div_rem_into(
                    a,
                    &self.x_pow_n_minus_1,
                    &mut self.quotient,
                    &mut self.rem,
                    &mut self.div_scratch,
                );
            } else {
                self.quotient.clear();
            }

            self.scaled_estimate.assign_product_with_scratch(
                &n_minus_1,
                &estimate,
                &mut self.mul_scratch,
            );
            self.sum.clone_from(&self.scaled_estimate);
            self.sum.add_assign(&self.quotient);

            Division::div_rem_into(
                &self.sum,
                &n_uint,
                &mut self.next_estimate,
                &mut self.rem,
                &mut self.div_scratch,
            );

            if self.next_estimate.cmp(&estimate) != Ordering::Less {
                return estimate;
            }
            swap(&mut estimate, &mut self.next_estimate);
        }
    }
}

/// Produces a Newton seed accurate to within a relative `1 / t` of the root.
///
/// The classic `2^ceil(bits / n)` start is off by up to a factor of two, which
/// costs one full-width iteration per bit of that error. Rooting the leading
/// half of the operand instead and scaling the answer back gives a seed already
/// correct to nearly the full width, at the price of one recursion whose work
/// halves at every level.
fn seed_estimate(a: &InternalArbiUint, n: u32, scratch: &mut NthRootScratch) -> InternalArbiUint {
    /// Below this the recursion costs more than the iterations it saves.
    const HALVING_FLOOR_BITS: usize = 4 * LIMB_BITS;

    let bits = a.significant_bits();
    let n_usize = usize::try_from(n).unwrap_or(usize::MAX);

    // The shift has to be a multiple of `n` so the root scales back by an exact
    // limb-independent amount, and it must leave a non-empty head behind. A zero
    // `n` has no remainder to take, and falls through to the plain seed below.
    let half_bits = bits >> 1;
    let shift = half_bits.saturating_sub(half_bits.checked_rem(n_usize).unwrap_or(half_bits));
    if bits < HALVING_FLOOR_BITS || shift == 0 {
        let mut estimate = InternalArbiUint::one();
        estimate.shl_assign(initial_shift(bits, n));
        return estimate;
    }

    let mut head = a.clone();
    head.shr_assign(shift);
    if head.is_zero() || head.is_one() {
        let mut estimate = InternalArbiUint::one();
        estimate.shl_assign(initial_shift(bits, n));
        return estimate;
    }

    let mut seed = if head.limbs().len() == 1 {
        NthRootScratch::nth_root_single_limb(&head, n)
    } else {
        scratch.nth_root_multi_limb(&head, n)
    };

    // With `t` the root of the head, `a < (t + 1)^n * 2^shift`, so
    // `(t + 1) << (shift / n)` strictly exceeds the true root. Newton descends,
    // so it needs that overestimate to terminate on the floor root.
    seed.add_assign(&InternalArbiUint::one());
    seed.shl_assign(shift.checked_div(n_usize).unwrap_or(0));
    seed
}

/// Raises `base` to `exponent` into `scratch.x_pow_n_minus_1`, bailing out as
/// soon as the running value exceeds `limit`.
fn bounded_pow_into(
    base: &InternalArbiUint,
    mut exponent: u32,
    limit: &InternalArbiUint,
    scratch: &mut NthRootScratch,
) -> bool {
    scratch.x_pow_n_minus_1.clone_from(&InternalArbiUint::one());
    scratch.base_pow.clone_from(base);
    while exponent > 0 {
        if exponent & 1 == 1 {
            scratch.temp_prod.assign_product_with_scratch(
                &scratch.x_pow_n_minus_1,
                &scratch.base_pow,
                &mut scratch.mul_scratch,
            );
            if scratch.temp_prod.cmp(limit) == Ordering::Greater {
                return false;
            }
            swap(&mut scratch.x_pow_n_minus_1, &mut scratch.temp_prod);
        }
        exponent >>= 1;
        if exponent > 0 {
            if scratch.base_pow.cmp(limit) == Ordering::Greater {
                return false;
            }
            scratch.temp_prod.assign_product_with_scratch(
                &scratch.base_pow,
                &scratch.base_pow,
                &mut scratch.mul_scratch,
            );
            if scratch.temp_prod.cmp(limit) == Ordering::Greater {
                return false;
            }
            swap(&mut scratch.base_pow, &mut scratch.temp_prod);
        }
    }
    true
}

fn initial_shift(bits: usize, n: u32) -> usize {
    usize::try_from(n).map_or_else(
        |_| {
            // On 16-bit targets this arm means n > usize::MAX >= bits, so
            // ceil(bits / n) is 1 for every non-zero input reaching nth_root.
            usize::from(bits != 0)
        },
        |n_usize| bits.div_ceil(n_usize),
    )
}
