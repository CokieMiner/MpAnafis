//! Trial division and Miller-Rabin kernels.
//!
//! References:
//! - G. L. Miller, "Riemann's Hypothesis and Tests for Primality",
//!   Journal of Computer and System Sciences 13(3), 300-317, 1976.
//!   <https://www.cs.cmu.edu/~glmiller/Publications/Papers/Mi76.pdf>
//! - M. O. Rabin, "Probabilistic algorithm for testing primality",
//!   Journal of Number Theory 12(1), 128-138, 1980.
//!   DOI: 10.1016/0022-314X(80)90084-0.

#![expect(
    unsafe_code,
    reason = "The primality kernels eliminate impossible zero-modulus branches after validating every modulus."
)]

use core::{cmp::Ordering, mem::swap};

use super::{
    ArchKernels, InternalMpUint, LIMB_BITS, Limb, MontgomeryDomain, MontgomeryScratch, Primality,
    TRIAL_PRIME_LIMIT, TRIAL_SCREEN,
};

impl Primality {
    /// Computes the number of trailing zeros of `n - 1` for an odd integer `n >= 3`.
    ///
    /// Because `n` is odd, `n - 1` has bit 0 cleared and all other bits unchanged.
    /// This evaluates the count without allocating or materializing `n - 1`.
    #[inline]
    pub fn trailing_zeros_odd_minus_one(n: &InternalMpUint) -> usize {
        let limbs = n.limbs();
        let Some(&first) = limbs.first() else {
            return 0;
        };
        debug_assert!(first & 1 == 1, "n must be odd");
        #[expect(
            clippy::as_conversions,
            reason = "trailing_zeros is bounded by LIMB_BITS <= 64, which fits safely in usize"
        )]
        // SAFETY: an odd low limb is at least one.
        let first_tz = unsafe { first.unchecked_sub(1) }.trailing_zeros() as usize;
        if first_tz != LIMB_BITS {
            return first_tz;
        }
        for (index, &limb) in limbs.iter().enumerate().skip(1) {
            #[expect(
                clippy::as_conversions,
                reason = "trailing_zeros is bounded by LIMB_BITS <= 64, which fits safely in usize"
            )]
            let tz = limb.trailing_zeros() as usize;
            if tz != LIMB_BITS {
                // Validate only the final bit position; zero limbs require no
                // arithmetic beyond their iterator index. The offset occupies
                // the aligned base's low bits and cannot overflow it.
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("trailing-zero count exceeds usize");
                return base | tz;
            }
        }
        // Normalization makes this fallthrough the single-limb value one.
        LIMB_BITS
    }

    /// Rejects multiples of any odd prime below 2000.
    ///
    /// For a product P of screened primes, low-to-high cancellation computes
    /// `c` with `A = Q*P-c*B^j`. Since P is odd, B is a unit modulo every
    /// factor of P; testing divisibility of c is equivalent to testing A.
    pub fn trial_division(a: &InternalMpUint) -> bool {
        let limbs = a.limbs();
        if limbs.is_empty() {
            return false;
        }

        if a.is_even() {
            return a.cmp(&InternalMpUint::from_limb(2)) == Ordering::Equal;
        }

        if let [value] = limbs {
            if *value <= TRIAL_PRIME_LIMIT {
                return Self::is_prime_limb(*value);
            }
            // For q=value*p^-1 mod B, q<=floor((B-1)/p) iff p*q<B
            // and p*q=value. Each rejected factor requires one multiplication.
            return TRIAL_SCREEN
                .factors
                .iter()
                .all(|factor| value.wrapping_mul(factor.inverse) > factor.limit);
        }

        // SAFETY: the zero and single-limb cases returned, so both the final
        // limb and its nonempty initialized prefix exist.
        let (&last, prefix) = unsafe { limbs.split_last().unwrap_unchecked() };
        // SAFETY: a multi-limb input leaves at least one initialized low limb.
        let (&low, middle) = unsafe { prefix.split_first().unwrap_unchecked() };
        let mut first = 0;
        for group in &TRIAL_SCREEN.products {
            // The initial carry is zero: the first cancellation requires no
            // subtraction or borrow calculation.
            let first_digit = low.wrapping_mul(group.inverse);
            let (_, mut carry) = ArchKernels::mul_limb_lo_hi(first_digit, group.divisor);
            for &limb in middle {
                let (difference, borrow) = limb.overflowing_sub(carry);
                let digit = difference.wrapping_mul(group.inverse);
                let (_, high) = ArchKernels::mul_limb_lo_hi(digit, group.divisor);
                // high<=P-1. Equality forces the low product<=B-P, while
                // a borrow requires difference>=B-P+1; high+borrow<P.
                // SAFETY: cancellation preserves carry<P, so the proved
                // high+borrow bound fits Limb on every supported target.
                carry = unsafe { high.unchecked_add(Limb::from(borrow)) };
            }
            let residue = if last <= group.divisor {
                // A/B^j is congruent to last-carry modulo every factor of P.
                // Its sign does not affect divisibility; no high product is needed.
                carry.abs_diff(last)
            } else {
                let (difference, borrow) = last.overflowing_sub(carry);
                let digit = difference.wrapping_mul(group.inverse);
                let (_, high) = ArchKernels::mul_limb_lo_hi(digit, group.divisor);
                // SAFETY: the same cancellation bound gives high+borrow<P.
                unsafe { high.unchecked_add(Limb::from(borrow)) }
            };
            // SAFETY: constant construction partitions exactly 302 factors;
            // the previous endpoint and this endpoint bound its initialized span.
            let factors = unsafe { TRIAL_SCREEN.factors.get_unchecked(first..group.end) };
            if factors
                .iter()
                .any(|factor| residue.wrapping_mul(factor.inverse) <= factor.limit)
            {
                return false;
            }
            first = group.end;
        }

        true
    }

    /// Single Miller-Rabin round: test base `a` on `n` with
    /// `n-1 = d * 2^s`.
    #[expect(
        clippy::too_many_arguments,
        reason = "standard math notation for MR test; scratch buffers passed for allocation reuse"
    )]
    #[must_use]
    pub fn miller_rabin_test(
        a: &InternalMpUint,
        d: &InternalMpUint,
        s: usize,
        n_minus_1_mont: &InternalMpUint,
        one_mont: &InternalMpUint,
        temp_prod: &mut InternalMpUint,
        temp_rem: &mut InternalMpUint,
        domain: &MontgomeryDomain,
        mul_scratch: &mut MontgomeryScratch,
    ) -> bool {
        let x_mont = if a.limbs() == [2] {
            // Montgomery encoding is linear: twice the encoded residue
            // represents multiplication by two. The leading exponent bit
            // initializes 2R; later set bits need only a modular doubling,
            // avoiding a power table and general Montgomery products.
            debug_assert!(!d.is_zero(), "n-1=d*2^s has a positive odd factor");
            let modulus = &domain.modulus;
            let mut value = one_mont.clone();
            value.shl_assign(1);
            if value >= *modulus {
                value.sub_assign(modulus);
            }
            // SAFETY: the positive odd exponent has at least one significant bit.
            let remaining = unsafe { d.significant_bits().unchecked_sub(1) };
            for bit in (0..remaining).rev() {
                domain.square_into_with_scratch(&value, temp_rem, temp_prod, mul_scratch);
                swap(&mut value, temp_rem);
                if d.get_bit(bit) {
                    value.shl_assign(1);
                    if value >= *modulus {
                        value.sub_assign(modulus);
                    }
                }
            }
            value
        } else {
            domain.pow(a, d, mul_scratch, true)
        };

        if x_mont.cmp(n_minus_1_mont) == Ordering::Equal {
            return true;
        }

        if x_mont.cmp(one_mont) == Ordering::Equal {
            return true;
        }

        if s <= 1 {
            return false;
        }

        let mut x_mont = x_mont;
        // SAFETY: s <= 1 returned above, so the remaining round count is positive.
        let remaining = unsafe { s.unchecked_sub(1) };
        for _ in 0..remaining {
            domain.square_into_with_scratch(&x_mont, temp_rem, temp_prod, mul_scratch);
            swap(&mut x_mont, temp_rem);

            if x_mont.cmp(n_minus_1_mont) == Ordering::Equal {
                return true;
            }
            if x_mont.cmp(one_mont) == Ordering::Equal {
                // Further squaring fixes one, so minus one is unreachable.
                return false;
            }
        }
        false
    }
}
