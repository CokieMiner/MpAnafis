//! Prime-only trial factorization with exact low-to-high scalar division.

#![expect(
    unsafe_code,
    reason = "the sieve and matching tables prove each index bound; scalar divisors and exact quotients follow from accepted prime factors"
)]

use core::ops::Range;

use super::{ArchKernels, DivScratch, Division, InternalMpUint, Limb, Primality};

/// Empirical trial budget before the general factor search.
pub const TRIAL_BOUND: u64 = 50_000;
/// Number of odd primes below the fixed trial budget, checked by the sieve.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the fixed trial bound is below 65_536 and fits every supported usize"
)]
pub const PRIME_COUNT: usize = Primality::odd_prime_count::<{ (TRIAL_BOUND as usize) >> 1 }>();
/// Number of odd trial primes tested before the first native primality test.
pub const PRIMALITY_PREFIX: usize = 64;

/// Namespace for the trial-factor stages of Euler's totient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totient;

struct TrialPrimes {
    primes: [u16; PRIME_COUNT],
    inverses: [Limb; PRIME_COUNT],
    limits: [Limb; PRIME_COUNT],
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the compile-time sieve bound and its prime entries are below 65_536, fitting usize and u16 on every supported pointer width"
)]
static TRIAL_PRIMES: TrialPrimes = {
    let mut table = TrialPrimes {
        primes: [0; PRIME_COUNT],
        inverses: [0; PRIME_COUNT],
        limits: [0; PRIME_COUNT],
    };
    // Odd indices represent 2*i+1. The 25,000-byte sieve and 30,792-byte
    // result each fit isize::MAX even on a 16-bit target. Sieve storage exists
    // only during constant evaluation; the runtime table is read-only.
    let bound = TRIAL_BOUND as usize;
    let mut composite = [false; (TRIAL_BOUND as usize) >> 1];
    let mut prime = 3_usize;
    let mut count = 0_usize;
    while prime < bound {
        // SAFETY: prime < TRIAL_BOUND bounds its halved odd index.
        if !unsafe { *composite.as_ptr().add(prime >> 1) } {
            let mut inverse = 1_usize;
            let mut bits = 1_u32;
            while bits < Limb::BITS {
                // Newton lifting doubles the valid low bits of p*inverse=1.
                inverse = inverse.wrapping_mul(2_usize.wrapping_sub(prime.wrapping_mul(inverse)));
                // SAFETY: bits is a power of two below Limb::BITS <= 64;
                // doubling gives the next precision, at most 64.
                bits = unsafe { bits.unchecked_mul(2) };
            }
            assert!(count < PRIME_COUNT, "the prime table covers the sieve");
            // SAFETY: the compile-time assertion bounds all three equal-sized
            // arrays. prime >= 3 proves the denominator nonzero.
            unsafe {
                *table.primes.as_mut_ptr().add(count) = prime as u16;
                *table.inverses.as_mut_ptr().add(count) = inverse;
                *table.limits.as_mut_ptr().add(count) =
                    Limb::MAX.checked_div(prime).unwrap_unchecked();
            }
            // SAFETY: the preceding assertion establishes count < 5132.
            count = unsafe { count.unchecked_add(1) };
            // SAFETY: bound=50_000 is positive, and prime starts at three.
            let last_multiple =
                unsafe { bound.unchecked_sub(1).checked_div(prime).unwrap_unchecked() };
            if prime <= last_multiple {
                // SAFETY: the division comparison proves prime^2 < TRIAL_BOUND.
                let mut multiple = unsafe { prime.unchecked_mul(prime) };
                while multiple < bound {
                    // SAFETY: multiple < TRIAL_BOUND bounds its sieve index.
                    // prime^2 < 50_000 implies prime <= 223; the increment
                    // is <= 446 and the complete sum fits even a 16-bit limb.
                    unsafe {
                        *composite.as_mut_ptr().add(multiple >> 1) = true;
                        multiple = multiple.unchecked_add(prime << 1);
                    }
                }
            }
        }
        // SAFETY: prime < 50_000, so prime+2 fits on every pointer width.
        prime = unsafe { prime.unchecked_add(2) };
    }
    assert!(
        count == PRIME_COUNT,
        "every table entry is initialized by a prime"
    );
    table
};

impl Totient {
    /// Removes the trial primes indexed by `primes` from an odd native
    /// cofactor, updating its totient accumulator for each distinct factor.
    ///
    /// Returns `true` once the cofactor is completely factored: it is one, or
    /// it is below the square of the next trial prime, hence prime, and has
    /// been applied. `primes` lies within the table and continues the ranges
    /// already removed, so no smaller prime divides the entering cofactor.
    pub fn trial_phi_limb(n: &mut Limb, result: &mut Limb, primes: Range<usize>) -> bool {
        debug_assert!(
            primes.end <= PRIME_COUNT && *n & 1 != 0,
            "trial ranges index the odd-prime table"
        );
        let mut index = primes.start;
        while index < primes.end {
            // n changes only when a factor is removed. Bound the whole next
            // scan by sqrt(n), using the monotone prime table once instead
            // of loading and squaring a prime at every rejected candidate.
            // SAFETY: the caller bounds the range, and index < primes.end.
            let remaining = unsafe { TRIAL_PRIMES.primes.get_unchecked(index..primes.end) };
            // SAFETY: index < primes.end proves remaining is nonempty.
            let last = u64::from(unsafe { *remaining.last().unwrap_unchecked() });
            // SAFETY: all supported limb widths are at most 64 bits.
            let value = unsafe { u64::try_from(*n).unwrap_unchecked() };
            // Each prime fits u16, so its square fits u64 on every target.
            let count = if value >= last.pow(2) {
                remaining.len()
            } else {
                // Factor removal can leave one or a prime below the next
                // candidate's square. Its empty scan is proved by the first
                // entry alone, without searching the rest of the table.
                // SAFETY: index < primes.end proves remaining is nonempty.
                let first = u64::from(unsafe { *remaining.first().unwrap_unchecked() });
                if value < first.pow(2) {
                    0
                } else {
                    remaining.partition_point(|&prime| u64::from(prime).pow(2) <= value)
                }
            };
            // SAFETY: count <= primes.end-index, hence the sum is <= the
            // table length, which fits every supported pointer width.
            let end = unsafe { index.unchecked_add(count) };
            while index < end {
                // SAFETY: index < end <= primes.end bounds both tables.
                let (inverse, limit) = unsafe {
                    (
                        *TRIAL_PRIMES.inverses.get_unchecked(index),
                        *TRIAL_PRIMES.limits.get_unchecked(index),
                    )
                };
                // SAFETY: index < end <= PRIME_COUNT == 5132.
                index = unsafe { index.unchecked_add(1) };
                let quotient = n.wrapping_mul(inverse);
                if quotient <= limit {
                    // q=n/p iff q=n*p^-1 mod B <= floor((B-1)/p):
                    // then p*q<B and its residue equals n. The first copy
                    // of each remaining prime also divides result exactly.
                    // SAFETY: the accepted distinct prime divides result;
                    // modular multiplication yields its exact quotient in
                    // [0,result], so the totient subtraction cannot underflow.
                    *result = unsafe { result.unchecked_sub(result.wrapping_mul(inverse)) };
                    *n = quotient;
                    loop {
                        let next = n.wrapping_mul(inverse);
                        if next > limit {
                            break;
                        }
                        *n = next;
                    }
                    break;
                }
            }
            if index == end && end < primes.end {
                // All possible prime factors up to sqrt(value) were removed.
                // The surviving cofactor is at most value, hence one or prime.
                if *n > 1 {
                    // SAFETY: n > 1 divides the accumulator exactly, so
                    // result/n is defined and at most result.
                    *result =
                        unsafe { result.unchecked_sub(result.checked_div(*n).unwrap_unchecked()) };
                }
                *n = 1;
                return true;
            }
        }
        *n == 1
    }

    /// Removes each complete trial factor; rejected candidates write no quotient.
    pub fn trial_phi(
        mut n: InternalMpUint,
        result: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) -> InternalMpUint {
        if n.is_even() {
            n.shr_assign(n.trailing_zeros());
            result.shr_assign(1);
            if n.is_one() {
                return n;
            }
        }
        let mut native = n.to_u64();
        for (index, &prime) in TRIAL_PRIMES.primes.iter().enumerate() {
            // SAFETY: prime is u16; its square is below 2^32 and fits u64.
            let square = unsafe { u64::from(prime).unchecked_mul(u64::from(prime)) };
            if native.is_some_and(|value| value < square) {
                // Every smaller prime was removed, so n is the last prime.
                let mut quotient = InternalMpUint::zero();
                Division::div_exact_into(result, &n, &mut quotient, scratch);
                result.sub_assign(&quotient);
                return InternalMpUint::one();
            }
            let scalar = usize::from(prime);
            if Division::modexact_1_odd(n.limbs(), scalar) != 0 {
                continue;
            }
            // SAFETY: enumeration bounds the matching inverse array.
            let inverse = unsafe { *TRIAL_PRIMES.inverses.get_unchecked(index) };
            divide_trial_factor::<true>(result, scalar, inverse);
            loop {
                divide_trial_factor::<false>(&mut n, scalar, inverse);
                if Division::modexact_1_odd(n.limbs(), scalar) != 0 {
                    break;
                }
            }
            if n.is_one() {
                return n;
            }
            native = n.to_u64();
        }
        n
    }
}

/// Exact division by an odd prime, optionally fused with value-value/prime.
/// The inverse is positive modulo the limb base; prime divides value.
fn divide_trial_factor<const TOTIENT: bool>(
    value: &mut InternalMpUint,
    prime: Limb,
    inverse: Limb,
) {
    let mut carry = 0_usize;
    let mut borrow = false;
    for digit in value.limbs_mut() {
        let original = *digit;
        let quotient = original.wrapping_sub(carry).wrapping_mul(inverse);
        let (low, high) = ArchKernels::mul_limb_lo_hi(quotient, prime);
        let (_, overflow) = low.overflowing_add(carry);
        // q*p+carry <= (B-1)*p+(p-1)=B*p-1, so the next carry is < p.
        // SAFETY: the incoming carry is below prime; the product bound
        // above gives high+overflow < prime <= Limb::MAX.
        carry = unsafe { high.unchecked_add(Limb::from(overflow)) };
        if TOTIENT {
            let (partial, first) = original.overflowing_sub(quotient);
            let (difference, second) = partial.overflowing_sub(Limb::from(borrow));
            *digit = difference;
            borrow = first | second;
        } else {
            *digit = quotient;
        }
    }
    debug_assert_eq!(carry, 0, "the caller proves exact scalar divisibility");
    debug_assert!(!borrow, "value-value/prime is nonnegative");
    value.normalize();
}
