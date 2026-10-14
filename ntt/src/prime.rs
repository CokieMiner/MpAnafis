//! Nonzero prime fields below `2^50` with power-of-two roots of unity.

#![expect(
    unsafe_code,
    reason = "Catalog admission and canonical residue contracts prove all unchecked integer operations"
)]

use core::num::NonZeroU64;

/// Parameters of a catalog prime and its power-of-two roots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prime {
    modulus: NonZeroU64,
    float: f64,
    reciprocal: f64,
    root: u64,
    two_valuation: u32,
    word_radix: u64,
}

/// Pairwise distinct prime fields, ordered by the reconstruction schedule.
pub const PRIME_CATALOG: &[Prime] = &[
    catalog_prime(1_108_307_720_798_209, 11, 44),
    catalog_prime(1_086_317_488_242_689, 3, 42),
    catalog_prime(910_395_627_798_529, 7, 42),
    catalog_prime(659_706_976_665_601, 11, 43),
    catalog_prime(699_289_395_265_537, 5, 42),
    catalog_prime(1_022_545_813_831_681, 7, 41),
    catalog_prime(1_013_749_720_809_473, 3, 41),
    catalog_prime(868_614_185_943_041, 3, 41),
];

/// Number of available prime moduli.
pub const MAX_PRIMES: usize = PRIME_CATALOG.len();

impl Prime {
    /// Computes `base^exponent mod p` by binary exponentiation.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Const widening is lossless; each narrowed remainder lies below the 50-bit prime"
    )]
    pub const fn pow_mod(&self, base: u64, mut exponent: u64) -> u64 {
        let modulus = self.modulus.get() as u128;
        let mut value = 1_u128;
        let mut factor = (base as u128).rem_euclid(modulus);
        // SAFETY: factor and value stay below p < 2^50. Their products use
        // fewer than 100 bits and fit u128, including the initial arbitrary base reduction.
        unsafe {
            while exponent != 0 {
                if exponent & 1 != 0 {
                    value = value.unchecked_mul(factor).rem_euclid(modulus);
                }
                exponent >>= 1;
                if exponent != 0 {
                    factor = factor.unchecked_mul(factor).rem_euclid(modulus);
                }
            }
        }
        value as u64
    }

    /// Computes the inverse of a nonzero residue using Fermat's identity.
    ///
    /// # Panics
    /// Panics when `value` is divisible by the modulus.
    #[must_use]
    pub const fn inv_mod(&self, value: u64) -> u64 {
        assert!(
            !value.is_multiple_of(self.modulus.get()),
            "zero has no field inverse"
        );
        // SAFETY: catalog admission proves p > 2.
        self.pow_mod(value, unsafe { self.modulus.get().unchecked_sub(2) })
    }

    /// Returns a primitive root of order `2^log_len`.
    ///
    /// # Panics
    /// Panics if `log_len` exceeds the two-adic valuation of `p - 1`.
    #[must_use]
    pub const fn primitive_root_of_unity(&self, log_len: u32) -> u64 {
        assert!(
            log_len <= self.two_valuation,
            "transform order exceeds the prime field"
        );
        // SAFETY: p > 1 and log_len <= two_valuation < 64.
        self.pow_mod(self.root, unsafe {
            self.modulus.get().unchecked_sub(1).unchecked_shr(log_len)
        })
    }

    /// Returns the integer modulus.
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.modulus.get()
    }

    /// Returns the maximum exponent of an admitted power-of-two root.
    #[must_use]
    pub const fn two_valuation(&self) -> u32 {
        self.two_valuation
    }

    /// Returns the exact modulus used by internal floating-point kernels.
    pub const fn float(&self) -> f64 {
        self.float
    }

    /// Returns the rounded reciprocal used by internal kernels.
    pub const fn reciprocal(&self) -> f64 {
        self.reciprocal
    }

    /// Returns `2^64 mod p` for wide-chunk conversion.
    pub const fn word_radix(&self) -> u64 {
        self.word_radix
    }

    /// Multiplies two canonical residues using an approximate quotient and exact correction.
    ///
    /// # Safety
    /// Both arguments are in `0..p`.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "50-bit factors are exact in f64; the quotient fits u64 and exact correction returns a canonical unsigned residue"
    )]
    pub unsafe fn mulmod_fast(&self, left: u64, right: u64) -> u64 {
        let modulus = i128::from(self.modulus.get());
        let estimated = (left as f64) * (right as f64) * self.reciprocal;
        // SAFETY: canonical factors are below 2^50. The product and quotient
        // product fit 100 bits. The finite nonnegative estimate is below 2^51;
        // its truncation differs from floor(left*right/p) by at most one.
        // The initial residue lies in (-p, 2p), and corrections fit i128.
        unsafe {
            let quotient = estimated.to_int_unchecked::<u64>();
            let product = i128::from(left).unchecked_mul(i128::from(right));
            let estimate = i128::from(quotient).unchecked_mul(modulus);
            let residue = product.unchecked_sub(estimate);
            let nonnegative = residue.unchecked_add(modulus.unchecked_mul(i128::from(residue < 0)));
            nonnegative.unchecked_sub(modulus.unchecked_mul(i128::from(nonnegative >= modulus)))
                as u64
        }
    }

    /// Reduces an arbitrary word with a reciprocal quotient and exact integer correction.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Rounding a u64 word changes its reciprocal quotient by less than one; exact correction returns an unsigned residue below p"
    )]
    pub fn reduce_word(&self, word: u64) -> u64 {
        let estimated = (word as f64) * self.reciprocal;
        let modulus = i128::from(self.modulus.get());
        // SAFETY: p > 2^49 makes the finite nonnegative quotient < 2^15. Its product with p fits
        // 65 bits, and the rounded quotient leaves a residue in (-p, 2p).
        unsafe {
            let quotient = estimated.to_int_unchecked::<u64>();
            let estimate = i128::from(quotient).unchecked_mul(modulus);
            let residue = i128::from(word).unchecked_sub(estimate);
            let nonnegative = residue.unchecked_add(modulus.unchecked_mul(i128::from(residue < 0)));
            nonnegative.unchecked_sub(modulus.unchecked_mul(i128::from(nonnegative >= modulus)))
                as u64
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Admitted moduli lie below 2^50 and are exact in f64; the radix remainder is below p and fits u64"
)]
const fn catalog_prime(modulus: u64, root: u64, two_valuation: u32) -> Prime {
    assert!(
        modulus > (1 << 49) && modulus < (1 << 50),
        "catalog primes must lie between 2^49 and 2^50"
    );
    // SAFETY: the preceding domain check proves modulus > 1 and every
    // subsequent exponent and conversion stays inside the catalog field bounds.
    unsafe {
        assert!(
            two_valuation == modulus.unchecked_sub(1).trailing_zeros(),
            "catalog valuation must equal v2(p-1)"
        );
        let reciprocal = 1.0 / modulus as f64;
        // p has 50 bits, so RN(1/p) = M/2^102 with a 53-bit mantissa M.
        // For |a*b| < 2*p^2, the quotient error is bounded by
        // (2*p*abs(p*M-2^102) + 2^47*M)/2^102 + 5/8.
        // A bound below one also bounds the 4*p^2 case below 3/2.
        let mantissa = ((reciprocal.to_bits() & ((1_u64 << 52) - 1)) | (1_u64 << 52)) as u128;
        let error = (modulus as u128)
            .unchecked_mul(mantissa)
            .abs_diff(1_u128 << 102);
        let bound = (modulus as u128)
            .unchecked_mul(2)
            .unchecked_mul(error)
            .unchecked_add(mantissa << 47);
        assert!(
            bound < (3_u128 << 99),
            "reciprocal error must preserve the lazy residue bounds"
        );
        Prime {
            modulus: NonZeroU64::new_unchecked(modulus),
            float: modulus as f64,
            reciprocal,
            root,
            two_valuation,
            word_radix: (1_u128 << 64).rem_euclid(modulus as u128) as u64,
        }
    }
}
