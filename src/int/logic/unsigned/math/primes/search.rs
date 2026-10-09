//! Segmented searches over odd candidates in increasing or decreasing order.

#![expect(
    unsafe_code,
    reason = "Odd candidates are at least three; fixed sieve windows and divisors bound all limb arithmetic and bitmap indices on every pointer width."
)]

use core::{num::NonZeroUsize, ops::Rem};

use super::{ArchKernels, InternalMpUint, Limb, Primality};

const SIEVE_WORDS: usize = 32;
#[expect(
    clippy::as_conversions,
    reason = "a u64 has 64 bits, which fits every supported usize"
)]
const SIEVE_SLOTS: usize = SIEVE_WORDS * u64::BITS as usize;

const SIEVE_PRIME_LIMIT: usize = 256;
pub const SIEVE_PRIMES: [Limb; Primality::odd_prime_count::<{ SIEVE_PRIME_LIMIT >> 1 }>()] =
    Primality::odd_primes(SIEVE_PRIME_LIMIT);

impl Primality {
    /// Searches odd values `candidate +/- 2*i`, starting at an odd value >= 3.
    /// The direction is fixed before entering the sieve and candidate loops.
    pub fn search_prime<const DESCENDING: bool>(mut candidate: InternalMpUint) -> InternalMpUint {
        debug_assert!(!candidate.is_even(), "prime-search candidates are odd");
        debug_assert!(
            candidate >= InternalMpUint::from_limb(3),
            "prime-search windows start at an odd value at least three"
        );
        loop {
            let mut sieve = [0_u64; SIEVE_WORDS];
            sieve_window::<DESCENDING>(&candidate, &mut sieve);
            let mut previous = 0;
            for (word_index, &word) in sieve.iter().enumerate() {
                let mut available = !word;
                while available != 0 {
                    // SAFETY: nonzero available has fewer than 64 trailing
                    // zeros, representable by usize on every pointer width.
                    let bit =
                        unsafe { usize::try_from(available.trailing_zeros()).unwrap_unchecked() };
                    // SAFETY: word_index < 32 and bit < 64. Selected indices
                    // increase, and twice their difference is below 4096.
                    let (index, step) = unsafe {
                        let index = (word_index << 6).unchecked_add(bit);
                        (index, index.unchecked_sub(previous) << 1)
                    };
                    if step != 0 {
                        if DESCENDING {
                            candidate.sub_assign(&InternalMpUint::from_limb(step));
                        } else {
                            candidate.add_assign(&InternalMpUint::from_limb(step));
                        }
                    }
                    if candidate.is_prime() {
                        return candidate;
                    }
                    previous = index;
                    // SAFETY: available!=0 proves its predecessor exists;
                    // intersecting it clears the least significant set bit.
                    available &= unsafe { available.unchecked_sub(1) };
                }
            }
            // SAFETY: previous < 2048 bounds the positive step. A descending
            // window starting at most 4097 retains the prime three and returns
            // above. A completed descending window therefore started above
            // 4097, so subtracting the remaining step leaves a candidate >= 3.
            let step = unsafe { (SIEVE_SLOTS * 2).unchecked_sub(previous << 1) };
            if DESCENDING {
                candidate.sub_assign(&InternalMpUint::from_limb(step));
            } else {
                candidate.add_assign(&InternalMpUint::from_limb(step));
            }
        }
    }
}

/// Marks proper small-prime multiples, preserving each sieve prime itself.
fn sieve_window<const DESCENDING: bool>(
    candidate: &InternalMpUint,
    sieve: &mut [u64; SIEVE_WORDS],
) {
    let limbs = candidate.limbs();
    let single = if let [value] = limbs {
        Some(*value)
    } else {
        None
    };
    let final_window = DESCENDING && single.is_some_and(|value| value <= SIEVE_SLOTS * 2 + 1);
    let limit = if final_window {
        // SAFETY: the caller supplies an odd candidate >= 3. This arm
        // has a single limb <= 4097; only indices down to three are valid.
        unsafe { limbs.get_unchecked(0).unchecked_sub(1) >> 1 }
    } else {
        SIEVE_SLOTS
    };
    for &p in &SIEVE_PRIMES {
        let mut remainder = 0;
        for &limb in limbs.iter().rev() {
            // SAFETY: p >= 3 and entering remainder < p, initially zero.
            // The quotient fits one limb and the returned remainder < p.
            remainder = unsafe { ArchKernels::divrem_1_unchecked(limb, remainder, p).1 };
        }
        // For c+2*i use i = -c/2 (mod p); for c-2*i use i = c/2 (mod p).
        // SAFETY: remainder < p <= 251; target*(p+1)/2 <= 250*126 fits
        // even a 16-bit limb, and every divisor is nonzero.
        let mut start = unsafe {
            let target = if DESCENDING || remainder == 0 {
                remainder
            } else {
                p.unchecked_sub(remainder)
            };
            let inverse_two = p.unchecked_add(1) >> 1;
            Rem::rem(
                target.unchecked_mul(inverse_two),
                NonZeroUsize::new_unchecked(p),
            )
        };
        if !DESCENDING
            && let Some(value) = single
            && value <= p
        {
            // SAFETY: value <= p bounds the difference. Both are odd,
            // so their even offset corresponds to an exact sieve index.
            let index = unsafe { p.unchecked_sub(value) >> 1 };
            if start == index {
                // SAFETY: start < p <= 251, hence the sum is below 502.
                start = unsafe { start.unchecked_add(p) };
            }
        }
        let mut index = start;
        while index < limit {
            // SAFETY: index < limit <= 2048 bounds the bitmap word.
            let word = unsafe { sieve.get_unchecked_mut(index >> 6) };
            *word |= 1_u64 << (index & 63);
            // SAFETY: index < 2048 and p <= 251 bound the next index on
            // every supported pointer width, including its final step.
            index = unsafe { index.unchecked_add(p) };
        }
        if DESCENDING
            && let Some(value) = single
            && value >= p
        {
            // SAFETY: value >= p bounds subtraction. Both are odd.
            let prime_index = unsafe { value.unchecked_sub(p) >> 1 };
            if prime_index < limit {
                // SAFETY: prime_index < limit <= 2048 bounds the word;
                // no other sieve prime divides p, so clearing retains p.
                let word = unsafe { sieve.get_unchecked_mut(prime_index >> 6) };
                *word &= !(1_u64 << (prime_index & 63));
            }
        }
    }
    // The descending terminal window may end inside a bitmap word.
    // Marking every unused slot prevents subtraction below three.
    // SAFETY: limit <= SIEVE_SLOTS, so limit >> 6 <= SIEVE_WORDS.
    let tail = unsafe { sieve.get_unchecked_mut(limit >> 6..) };
    if let Some((word, unused)) = tail.split_first_mut() {
        *word |= u64::MAX << (limit & 63);
        unused.fill(u64::MAX);
    }
}
