//! Compile-time prime products and exact scalar divisibility certificates.

use super::{Limb, Primality};

/// Exclusive extent of the grouped trial-division screen.
pub const TRIAL_PRIME_LIMIT: usize = 2_000;
const TRIAL_PRIMES: [Limb; Primality::odd_prime_count::<{ TRIAL_PRIME_LIMIT >> 1 }>()] =
    Primality::odd_primes(TRIAL_PRIME_LIMIT);
const PRODUCT_COUNT: usize = prime_product_count();

/// `p^-1 mod B` and `floor((B-1)/p)` certify divisibility by the odd prime p.
#[derive(Clone, Copy)]
pub struct PrimeFactor {
    pub inverse: Limb,
    pub limit: Limb,
}

/// One product of distinct odd primes and its exclusive factor-table endpoint.
#[derive(Clone, Copy)]
pub struct PrimeProduct {
    pub divisor: Limb,
    pub inverse: Limb,
    pub end: usize,
}

/// Read-only screen whose group widths follow the native limb capacity.
pub struct TrialScreen {
    pub products: [PrimeProduct; PRODUCT_COUNT],
    pub factors: [PrimeFactor; TRIAL_PRIMES.len()],
}

pub static TRIAL_SCREEN: TrialScreen = TrialScreen::new();

impl TrialScreen {
    /// Greedily packs consecutive primes while their product fits a limb.
    /// Counts, group endpoints and inverse certificates share the same extent;
    /// all construction executes at compile time on the target's limb width.
    #[expect(
        clippy::indexing_slicing,
        reason = "constant evaluation traverses the derived prime count and flushes exactly PRODUCT_COUNT nonempty groups"
    )]
    const fn new() -> Self {
        let mut screen = Self {
            products: [PrimeProduct {
                divisor: 1,
                inverse: 1,
                end: 0,
            }; PRODUCT_COUNT],
            factors: [PrimeFactor {
                inverse: 0,
                limit: 0,
            }; TRIAL_PRIMES.len()],
        };
        let mut group = 0_usize;
        let mut index = 0_usize;
        let mut product = PrimeProduct {
            divisor: 1,
            inverse: 1,
            end: 0,
        };
        while index < TRIAL_PRIMES.len() {
            let prime = TRIAL_PRIMES[index];
            if product.divisor.checked_mul(prime).is_none() {
                screen.products[group] = product;
                group = group.checked_add(1).expect("bounded prime groups");
                product = PrimeProduct {
                    divisor: 1,
                    inverse: 1,
                    end: index,
                };
            }
            // For odd p, Newton lifting doubles the number of valid low bits
            // of p*x=1. Wrapping arithmetic expresses the native limb ring.
            let mut inverse = 1_usize;
            let mut bits = 1_u32;
            while bits < Limb::BITS {
                inverse = inverse.wrapping_mul(2_usize.wrapping_sub(prime.wrapping_mul(inverse)));
                bits = bits.checked_mul(2).expect("limb precision at most 64 bits");
            }
            screen.factors[index] = PrimeFactor {
                inverse,
                limit: Limb::MAX.div_euclid(prime),
            };
            product.divisor = product
                .divisor
                .checked_mul(prime)
                .expect("admitted prime product");
            product.inverse = product.inverse.wrapping_mul(inverse);
            index = index.checked_add(1).expect("bounded prime count");
            product.end = index;
        }
        screen.products[group] = product;
        assert!(
            group.checked_add(1).expect("bounded prime groups") == PRODUCT_COUNT,
            "prime packing agrees with the derived group count"
        );
        screen
    }
}

impl Primality {
    /// Counts odd primes below `2*ODD_SLOTS` with compile-time sieve storage.
    /// All table extents are even; halving their bound keeps the largest
    /// sieve (25,000 bytes) addressable on the smallest supported pointer width.
    #[expect(
        clippy::indexing_slicing,
        reason = "the sieve visits odd candidates and multiples strictly below twice its derived slot count"
    )]
    pub const fn odd_prime_count<const ODD_SLOTS: usize>() -> usize {
        let limit = ODD_SLOTS.checked_mul(2).expect("prime extent fits usize");
        let mut composite = [false; ODD_SLOTS];
        let mut count = 0_usize;
        let mut candidate = 3_usize;
        while candidate < limit {
            if !composite[candidate >> 1] {
                count = count.checked_add(1).expect("bounded prime count");
                if candidate
                    <= limit
                        .checked_sub(1)
                        .expect("candidate below positive extent")
                        .div_euclid(candidate)
                {
                    let mut multiple = candidate
                        .checked_mul(candidate)
                        .expect("admitted square fits");
                    let step = candidate
                        .checked_mul(2)
                        .expect("square-admitted prime fits");
                    while multiple < limit {
                        composite[multiple >> 1] = true;
                        multiple = multiple.checked_add(step).expect("bounded sieve multiple");
                    }
                }
            }
            candidate = candidate
                .checked_add(2)
                .expect("bounded prime-table extent");
        }
        count
    }

    /// Builds an exact table with the previously derived prime count.
    #[expect(
        clippy::indexing_slicing,
        reason = "constant evaluation asserts the derived extent before writing each prime"
    )]
    pub const fn odd_primes<const COUNT: usize>(limit: usize) -> [Limb; COUNT] {
        let mut primes = [0; COUNT];
        let mut count = 0_usize;
        let mut candidate = 3_usize;
        while candidate < limit {
            if odd_prime(candidate) {
                assert!(count < COUNT, "prime count matches the selected extent");
                primes[count] = candidate;
                count = count.checked_add(1).expect("bounded prime count");
            }
            candidate = candidate
                .checked_add(2)
                .expect("bounded prime-table extent");
        }
        assert!(count == COUNT, "prime count matches the selected extent");
        primes
    }
}

const fn odd_prime(candidate: usize) -> bool {
    let mut divisor = 3_usize;
    while divisor <= candidate.div_euclid(divisor) {
        if candidate.is_multiple_of(divisor) {
            return false;
        }
        divisor = divisor.checked_add(2).expect("bounded trial divisor");
    }
    true
}

#[expect(
    clippy::indexing_slicing,
    reason = "constant evaluation visits only the derived prime table's initialized entries"
)]
const fn prime_product_count() -> usize {
    let mut count = 1_usize;
    let mut product = 1_usize;
    let mut index = 0_usize;
    while index < TRIAL_PRIMES.len() {
        let prime = TRIAL_PRIMES[index];
        if let Some(next) = product.checked_mul(prime) {
            product = next;
        } else {
            count = count.checked_add(1).expect("bounded prime groups");
            product = prime;
        }
        index = index.checked_add(1).expect("bounded prime count");
    }
    count
}
