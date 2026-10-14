//! Generic mixed-radix reconstruction with a sequential native-limb writer.
//!
//! Garner digits satisfy `X = x0 + p0*(x1 + p1*(...))`. Every intermediate
//! Horner value is nonnegative and at most X. The admitted coefficient bound
//! therefore limits all intermediate word spans as well as the final value.

#![expect(
    unsafe_code,
    reason = "Admitted geometry proves residue indices, accumulator widths, and sequential output capacity"
)]

use crate::arch::Scalar;

use super::{LIMB_BITS, Limb, MAX_PRIMES, PRIME_CATALOG, PlannedProfile};

// SAFETY: Prime occupies at least 32 bytes, and the catalog fits isize::MAX
// bytes. Thus MAX_PRIMES*50 < usize::MAX on every supported target.
const WORDS: usize = unsafe { MAX_PRIMES.unchecked_mul(50) }.div_ceil(64);
const INVERSES: [[u64; MAX_PRIMES]; MAX_PRIMES] = inverse_table();

/// Exact reconstruction for any admitted prefix of the prime catalog.
#[derive(Debug)]
pub struct Crt;

struct Writer<'output> {
    output: &'output mut [Limb],
    position: usize,
    pending: u128,
    pending_bits: usize,
}

impl Crt {
    /// Reconstructs scaled spectra into exactly the admitted native product span.
    ///
    /// # Safety
    /// `geometry` is admitted by an immutable native plan. Active residue
    /// slices cover `conv_len` initialized integer residues below `2*p` in magnitude.
    /// They encode the nonnegative convolution of the admitted binary chunks.
    /// `dst` is exactly the native product span of that plan.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Narrowing u128 sums keeps the low word; the high word is propagated separately"
    )]
    pub unsafe fn reconstruct(
        dst: &mut [Limb],
        residues: &[&[f64]; MAX_PRIMES],
        geometry: &PlannedProfile,
    ) {
        // SAFETY: plan admission proves the coefficient width fits usize,
        // output bit count fits usize, and all arrays cover the derived word counts.
        // Products of a u64 word and a 50-bit prime fit u128; carry addition
        // fits 65 bits. Every subtraction below is ordered by its loop guard.
        unsafe {
            let width = usize::try_from(geometry.profile.chunk_bits).unwrap_unchecked();
            let terms = geometry.digits_a.min(geometry.digits_b);
            let term_bits = usize::BITS.unchecked_sub(terms.unchecked_sub(1).leading_zeros());
            let coefficient_bits = width
                .unchecked_mul(2)
                .unchecked_add(usize::try_from(term_bits).unwrap_unchecked());
            let coefficient_words = coefficient_bits.div_ceil(64);
            let carry_words = width
                .unchecked_add(usize::try_from(term_bits).unwrap_unchecked())
                .div_ceil(64);
            let mut remaining = dst.len().unchecked_mul(LIMB_BITS);
            let coefficients = geometry.conv_len.min(remaining.div_ceil(width));
            let mut writer = Writer {
                output: dst,
                position: 0,
                pending: 0,
                pending_bits: 0,
            };
            let mut carry = [0_u64; WORDS];
            for index in 0..coefficients {
                let mut value =
                    coefficient(residues, index, geometry.profile.np, coefficient_words);
                let mut high = 0_u128;
                for word_index in 0..coefficient_words {
                    let incoming = if word_index < carry_words {
                        *carry.get_unchecked(word_index)
                    } else {
                        0
                    };
                    let word = value.get_unchecked_mut(word_index);
                    let sum = u128::from(*word)
                        .unchecked_add(u128::from(incoming))
                        .unchecked_add(high);
                    *word = sum as u64;
                    high = sum >> 64;
                }
                debug_assert_eq!(
                    high, 0,
                    "coefficient and carry fit their admitted bit bound"
                );
                let emitted = width.min(remaining);
                writer.push(&value, emitted);
                remaining = remaining.unchecked_sub(emitted);
                let whole = width >> 6;
                let offset = width % 64;
                for word_index in 0..carry_words {
                    let source = word_index.unchecked_add(whole);
                    let low = value.get(source).copied().unwrap_or(0);
                    let next = value.get(source.unchecked_add(1)).copied().unwrap_or(0);
                    *carry.get_unchecked_mut(word_index) = if offset == 0 {
                        low
                    } else {
                        (low >> offset) | (next << 64_usize.unchecked_sub(offset))
                    };
                }
            }
            // After the final coefficient, at most width output bits remain:
            // (digits_a+digits_b-1)*width >= native_input_bits-width.
            writer.push(&carry, remaining);
            debug_assert_eq!(
                writer.pending_bits, 0,
                "native product ends at a limb boundary"
            );
            debug_assert_eq!(
                writer.position,
                writer.output.len(),
                "the writer initializes the complete product span"
            );
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Scaled residues are exact integers below 2*p; their canonical representatives fit u64, and Horner low-word narrowing propagates its high word"
)]
unsafe fn coefficient(
    residues: &[&[f64]; MAX_PRIMES],
    index: usize,
    count: usize,
    word_count: usize,
) -> [u64; WORDS] {
    let mut digits = [0_u64; MAX_PRIMES];
    let mut value = [0_u64; WORDS];
    // SAFETY: count is a nonempty admitted catalog prefix. Every residue slice
    // covers index; earlier digits are initialized before use. The nonnegative
    // Horner intermediates do not exceed the coefficient, whose admitted bound
    // fits word_count words. All products fit 114 bits in u128.
    unsafe {
        for k in 0..count {
            let prime = PRIME_CATALOG.get_unchecked(k);
            let residue = *residues.get_unchecked(k).get_unchecked(index);
            let centered = Scalar::reduce(residue, prime.float(), prime.reciprocal());
            let mut digit = (if centered < 0.0 {
                centered + prime.float()
            } else {
                centered
            })
            .to_int_unchecked::<u64>();
            for j in 0..k {
                let earlier = *digits.get_unchecked(j);
                let inverse = *INVERSES.get_unchecked(j).get_unchecked(k);
                // Catalog moduli lie in (2^49, 2^50); one subtraction reduces
                // any earlier canonical mixed-radix digit modulo the current prime.
                let reduced = if earlier >= prime.modulus() {
                    earlier.unchecked_sub(prime.modulus())
                } else {
                    earlier
                };
                let (difference, borrow) = digit.overflowing_sub(reduced);
                let correction = prime.modulus().unchecked_mul(u64::from(borrow));
                // A borrow is represented modulo 2^64; adding p restores the
                // exact canonical difference in 0..p.
                digit = prime.mulmod_fast(difference.wrapping_add(correction), inverse);
            }
            *digits.get_unchecked_mut(k) = digit;
        }
        *value.get_unchecked_mut(0) = *digits.get_unchecked(count.unchecked_sub(1));
        for k in (0..count.unchecked_sub(1)).rev() {
            let prime = PRIME_CATALOG.get_unchecked(k);
            let mut high = u128::from(*digits.get_unchecked(k));
            for word in value.iter_mut().take(word_count) {
                let product = u128::from(*word)
                    .unchecked_mul(u128::from(prime.modulus()))
                    .unchecked_add(high);
                *word = product as u64;
                high = product >> 64;
            }
            debug_assert_eq!(
                high, 0,
                "nonnegative Horner intermediates fit the coefficient bound"
            );
        }
    }
    value
}

impl Writer<'_> {
    #[expect(
        clippy::as_conversions,
        reason = "Each store takes the low native limb of the accumulator and then shifts those bits away"
    )]
    fn push(&mut self, words: &[u64; WORDS], width: usize) {
        let mut remaining = width;
        // SAFETY: the sole caller clips emitted width to remaining output bits.
        // pending_bits is below LIMB_BITS before each appended word, so the
        // u128 accumulator uses at most 127 bits. Full-limb stores therefore
        // remain inside output, with no per-store bounds validation required.
        unsafe {
            for &word in words.iter().take(width.div_ceil(64)) {
                let bits = remaining.min(64);
                let mask = u64::MAX >> 64_usize.unchecked_sub(bits);
                self.pending |= u128::from(word & mask) << self.pending_bits;
                self.pending_bits = self.pending_bits.unchecked_add(bits);
                while self.pending_bits >= LIMB_BITS {
                    *self.output.get_unchecked_mut(self.position) = self.pending as Limb;
                    self.position = self.position.unchecked_add(1);
                    self.pending >>= LIMB_BITS;
                    self.pending_bits = self.pending_bits.unchecked_sub(LIMB_BITS);
                }
                remaining = remaining.unchecked_sub(bits);
            }
        }
    }
}

const fn inverse_table() -> [[u64; MAX_PRIMES]; MAX_PRIMES] {
    let mut table = [[0_u64; MAX_PRIMES]; MAX_PRIMES];
    let mut j = 0;
    // SAFETY: j and k stay below the literal catalog length; each pair contains
    // distinct primes, so the first prime has a nonzero residue modulo the second.
    unsafe {
        while j < MAX_PRIMES {
            let mut k = j.unchecked_add(1);
            while k < MAX_PRIMES {
                let prime = &*PRIME_CATALOG.as_ptr().add(k);
                let inverse = prime.inv_mod(
                    (*PRIME_CATALOG.as_ptr().add(j))
                        .modulus()
                        .rem_euclid(prime.modulus()),
                );
                *table.as_mut_ptr().add(j).cast::<u64>().add(k) = inverse;
                k = k.unchecked_add(1);
            }
            j = j.unchecked_add(1);
        }
    }
    table
}
