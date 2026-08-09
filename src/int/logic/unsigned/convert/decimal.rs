//! Decimal chunk extraction and the shared single-limb division kernel.

#![allow(
    unsafe_code,
    reason = "The 100-entry digit-pair table is indexed by proved remainders in 0..100, limb loops use index < len, and emitted bytes are ASCII."
)]

use alloc::{string::String, vec::Vec};

use super::{ArchKernels, InternalMpUint, Limb, byte_from_digit, estimated_digits};

// Decimal chunking: radix 10 divides by a single-limb power of ten per step,
// emitting `DECIMAL_CHUNK_DIGITS` digits per division instead of one. The
// power of ten is the largest that fits a `Limb` on the target: 10^19 < 2^64,
// 10^9 < 2^32, and 10^4 < 2^16, so the remainder of every chunked division
// fits in a single limb on all three supported pointer widths.
#[cfg(target_pointer_width = "64")]
const DECIMAL_CHUNK_DIVISOR: Limb = 10_000_000_000_000_000_000;
#[cfg(target_pointer_width = "64")]
const DECIMAL_CHUNK_DIGITS: usize = 19;
#[cfg(target_pointer_width = "32")]
const DECIMAL_CHUNK_DIVISOR: Limb = 1_000_000_000;
#[cfg(target_pointer_width = "32")]
const DECIMAL_CHUNK_DIGITS: usize = 9;
#[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
const DECIMAL_CHUNK_DIVISOR: Limb = 10_000;
#[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
const DECIMAL_CHUNK_DIGITS: usize = 4;

/// A lookup table for formatting two decimal digits at a time without repeated division.
/// Each entry `i` contains the ASCII bytes for `i / 10` and `i % 10`.
const DECIMAL_DIGIT_PAIRS: [[u8; 2]; 100] = [
    *b"00", *b"01", *b"02", *b"03", *b"04", *b"05", *b"06", *b"07", *b"08", *b"09", *b"10", *b"11",
    *b"12", *b"13", *b"14", *b"15", *b"16", *b"17", *b"18", *b"19", *b"20", *b"21", *b"22", *b"23",
    *b"24", *b"25", *b"26", *b"27", *b"28", *b"29", *b"30", *b"31", *b"32", *b"33", *b"34", *b"35",
    *b"36", *b"37", *b"38", *b"39", *b"40", *b"41", *b"42", *b"43", *b"44", *b"45", *b"46", *b"47",
    *b"48", *b"49", *b"50", *b"51", *b"52", *b"53", *b"54", *b"55", *b"56", *b"57", *b"58", *b"59",
    *b"60", *b"61", *b"62", *b"63", *b"64", *b"65", *b"66", *b"67", *b"68", *b"69", *b"70", *b"71",
    *b"72", *b"73", *b"74", *b"75", *b"76", *b"77", *b"78", *b"79", *b"80", *b"81", *b"82", *b"83",
    *b"84", *b"85", *b"86", *b"87", *b"88", *b"89", *b"90", *b"91", *b"92", *b"93", *b"94", *b"95",
    *b"96", *b"97", *b"98", *b"99",
];

impl InternalMpUint {
    /// Formats the integer as a string in the given radix using the schoolbook algorithm.
    ///
    /// Uses the decimal chunk kernel for radix 10 and the generic per-digit
    /// division loop for every other non-power-of-two radix. The caller
    /// validates the radix and routes zero, powers of two, and the recursive
    /// domain before invoking this path.
    pub fn format_schoolbook_string(&self, radix: u32) -> String {
        let significant_bits = self.significant_bits();
        let mut output = Vec::with_capacity(estimated_digits(significant_bits, radix));
        let mut value = self.clone();

        if radix == 10 {
            write_decimal_chunks(&mut value, &mut output);
        } else {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "radix is checked to be in 2..=36 and therefore fits in Limb"
            )]
            let radix_limb = radix as Limb;
            while !value.is_zero() {
                let remainder = div_rem_small(&mut value, radix_limb);
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "remainder is less than radix, which is at most 36"
                )]
                output.push(byte_from_digit(remainder as u8));
            }
        }

        output.reverse();
        // SAFETY: byte_from_digit produces ASCII decimal digits and lowercase letters only.
        unsafe { String::from_utf8_unchecked(output) }
    }
}

/// Divides `value` by a single-limb `divisor`, storing the quotient in place
/// and returning the remainder.
///
/// The loop walks the limbs most significant first, feeding the running
/// remainder of the previous step into the architecture division kernel, and
/// renormalizes the value before returning.
pub fn div_rem_small(value: &mut InternalMpUint, divisor: Limb) -> Limb {
    let len = value.limbs().len();
    let mut remainder = 0;

    for index in (0..len).rev() {
        // SAFETY: index is in 0..len.
        let limb = unsafe { *value.limbs().as_ptr().add(index) };
        // SAFETY: division maintains remainder < divisor.
        let (quotient, next_remainder) =
            unsafe { ArchKernels::divrem_1_unchecked(limb, remainder, divisor) };
        // SAFETY: index is in 0..len, and the source limb was read first.
        unsafe {
            *value.limbs_mut().as_mut_ptr().add(index) = quotient;
        }
        remainder = next_remainder;
    }

    value.normalize();
    remainder
}

/// Writes the decimal digits of a non-zero value into `output` in
/// least-significant-group-first order, for callers that reverse the appended
/// range afterwards.
///
/// Each iteration performs one multi-limb division by
/// [`DECIMAL_CHUNK_DIVISOR`] and appends the resulting `DECIMAL_CHUNK_DIGITS`-
/// digit group from its single-limb remainder. The most significant group is
/// appended without leading zero padding: it is the final remainder of
/// dividing a non-zero value by a positive power of ten, so it is itself
/// non-zero.
pub fn write_decimal_chunks(value: &mut InternalMpUint, output: &mut Vec<u8>) {
    debug_assert!(!value.is_zero(), "only non-zero decimal values are chunked");
    loop {
        let chunk = div_rem_decimal_chunk(value);
        if value.is_zero() {
            // Leading group: strip the fixed-width zero padding. The group is
            // non-zero — `value` was already equal to `chunk < divisor` when
            // this division produced a zero quotient, so the remainder is the
            // non-zero value itself — hence at least one digit is non-zero.
            push_decimal_chunk_reversed(chunk, false, output);
            break;
        }
        push_decimal_chunk_reversed(chunk, true, output);
    }
}

/// Emits the ASCII decimal digits of a single-limb chunk into `output` in
/// reversed order (least significant digit first), for callers that reverse
/// the appended range afterwards.
///
/// With `full_width` set, exactly `DECIMAL_CHUNK_DIGITS` digits are emitted
/// (zero-padded); otherwise only the significant digits are emitted. Every
/// emitted byte comes from the 100-entry [`DECIMAL_DIGIT_PAIRS`] table or is
/// a single digit below ten, so no temporary digit array is needed: the
/// destination is written pair by pair in its reversed final order.
#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::useless_conversion,
    reason = "remainders of divisions by 100 are at most 99 and the leftover leading digit is at most 9, so the narrowing to u8 and the table indexing are in bounds; usize::from is the identity conversion on 32- and 64-bit limbs and widens on 16-bit limbs, keeping one spelling across pointer widths"
)]
fn push_decimal_chunk_reversed(mut chunk: Limb, full_width: bool, output: &mut Vec<u8>) {
    let pairs = DECIMAL_CHUNK_DIGITS.wrapping_div(2);

    if full_width {
        for _ in 0..pairs {
            let quotient = chunk.wrapping_div(100);
            let remainder = chunk.wrapping_rem(100);
            // SAFETY: the remainder of a division by 100 is in 0..100, inside
            // the 100-entry table.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(usize::from(remainder)) };
            output.push(pair[1]);
            output.push(pair[0]);
            chunk = quotient;
        }
        if !DECIMAL_CHUNK_DIGITS.is_multiple_of(2) {
            // After `pairs` divisions by 100 the leftover is the leading
            // digit, below ten for every supported chunk width.
            output.push(b'0'.wrapping_add(chunk as u8));
        }
    } else {
        while chunk >= 100 {
            let quotient = chunk.wrapping_div(100);
            let remainder = chunk.wrapping_rem(100);
            // SAFETY: the remainder of a division by 100 is in 0..100, inside
            // the 100-entry table.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(usize::from(remainder)) };
            output.push(pair[1]);
            output.push(pair[0]);
            chunk = quotient;
        }
        if chunk >= 10 {
            // SAFETY: chunk is below 100 in this branch, inside the 100-entry
            // table.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(usize::from(chunk)) };
            output.push(pair[1]);
            output.push(pair[0]);
        } else {
            output.push(b'0'.wrapping_add(chunk as u8));
        }
    }
}

#[cfg(target_pointer_width = "64")]
const DECIMAL_CHUNK_PREINV: u64 = 0xd83c_94fb_6d2a_c34a;

#[cfg(target_pointer_width = "64")]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    reason = "integer division kernels rely on checked wrapping semantics, bounded indexing on limbs, and explicit truncation"
)]
#[inline]
fn div_rem_decimal_chunk(value: &mut InternalMpUint) -> Limb {
    // `DECIMAL_CHUNK_DIVISOR` is `10^19` on 64-bit targets, so this is the
    // identity conversion; the shared constant keeps a single source of truth.
    const D: u64 = DECIMAL_CHUNK_DIVISOR as u64;
    const DI: u64 = DECIMAL_CHUNK_PREINV;

    let len = value.limbs().len();
    let mut remainder = 0_u64;

    for index in (0..len).rev() {
        // SAFETY: index is in 0..len.
        let low = unsafe { *value.limbs().as_ptr().add(index) as u64 };
        let high = remainder;

        let product = u128::from(high).wrapping_mul(u128::from(DI));
        let product_high = product.wrapping_shr(64) as u64;
        let product_low = product as u64;

        let (estimate_low, carry) = product_low.overflowing_add(low);

        let mut quotient = product_high
            .wrapping_add(high)
            .wrapping_add(1)
            .wrapping_add(u64::from(carry));

        let mut next_remainder = low.wrapping_sub(quotient.wrapping_mul(D));

        // Correct an estimate that was one too large.
        if next_remainder > estimate_low {
            quotient = quotient.wrapping_sub(1);
            next_remainder = next_remainder.wrapping_add(D);
        }

        // Correct an estimate that was one too small.
        if next_remainder >= D {
            quotient = quotient.wrapping_add(1);
            next_remainder = next_remainder.wrapping_sub(D);
        }

        // SAFETY: index is in 0..len.
        unsafe {
            *value.limbs_mut().as_mut_ptr().add(index) = quotient as Limb;
        }

        remainder = next_remainder;
    }

    value.normalize();
    remainder as Limb
}

/// Divides `value` by the decimal chunk divisor and returns the remainder.
///
/// On 64-bit limbs the preinverse kernel above avoids the generic double-limb
/// division; on narrower targets the generic single-limb kernel handles the
/// double-limb steps exactly, so the chunk remainder is `value % divisor`.
#[cfg(not(target_pointer_width = "64"))]
#[inline]
fn div_rem_decimal_chunk(value: &mut InternalMpUint) -> Limb {
    div_rem_small(value, DECIMAL_CHUNK_DIVISOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use proptest::prelude::*;

    use crate::int::types::LIMB_BITS;

    // Checks the 64-bit preinverse chunk kernel against exact `u128`
    // division, covering every remainder and quotient pattern of a two-limb
    // value including the exact-division boundaries the kernel was first
    // written to serve.
    #[cfg(target_pointer_width = "64")]
    proptest! {
        #[test]
        fn preinverse_decimal_chunk_matches_exact_u128_division(
            low in any::<u64>(),
            high in any::<u64>(),
        ) {
            const DIVISOR: u128 = 10_000_000_000_000_000_000;

            let dividend = u128::from(high).wrapping_shl(64).wrapping_add(u128::from(low));
            let mut value = InternalMpUint::from_limbs(vec![
                usize::try_from(low).expect("u64 fits in usize on 64-bit targets"),
                usize::try_from(high).expect("u64 fits in usize on 64-bit targets"),
            ]);
            let remainder = div_rem_decimal_chunk(&mut value);

            prop_assert_eq!(
                u128::try_from(remainder).expect("limb fits in u128"),
                dividend.wrapping_rem(DIVISOR)
            );

            let mut quotient = 0_u128;
            for (index, limb) in value.limbs().iter().enumerate() {
                // SAFETY: a materialized integer has a `usize` bit length, so
                // the limb index is below `usize::BITS <= 64`.
                let shift = unsafe { u32::try_from(index).unwrap_unchecked() }.wrapping_mul(64);
                quotient = quotient.wrapping_add(
                    u128::try_from(*limb)
                        .expect("limb fits in u128")
                        .wrapping_shl(shift),
                );
            }
            prop_assert_eq!(quotient, dividend.wrapping_div(DIVISOR));
        }
    }

    // The chunk division must satisfy `value = quotient * divisor + remainder`
    // on every pointer width, pinning the generic kernels and the shared
    // schoolbook path.
    proptest! {
        #[test]
        fn decimal_chunk_reconstructs_wide_values(
            low in any::<Limb>(),
            mid in any::<Limb>(),
            high in any::<Limb>(),
        ) {
            let original = InternalMpUint::from_limbs(vec![low, mid, high]);
            let mut value = original.clone();
            let remainder = div_rem_decimal_chunk(&mut value);
            let divisor = InternalMpUint::from_limb(DECIMAL_CHUNK_DIVISOR);
            let reconstructed = value
                .mul(&divisor)
                .add(&InternalMpUint::from_limb(remainder));
            prop_assert_eq!(reconstructed, original);
        }
    }

    /// Static regression for the exact-division failure class: one below the
    /// divisor, exactly the divisor, and a full-limb multiple of it.
    #[test]
    fn decimal_chunk_exact_division_boundaries() {
        // One below the chunk divisor: the quotient must be zero and the
        // remainder the value itself.
        let mut value = InternalMpUint::from_limb(DECIMAL_CHUNK_DIVISOR.wrapping_sub(1));
        let remainder = div_rem_decimal_chunk(&mut value);
        assert!(
            value.is_zero(),
            "one below the divisor must yield a zero quotient"
        );
        assert_eq!(remainder, DECIMAL_CHUNK_DIVISOR.wrapping_sub(1));

        // Exactly the chunk divisor: quotient one, remainder zero.
        let exact = InternalMpUint::from_limb(DECIMAL_CHUNK_DIVISOR);
        let mut exact_value = exact.clone();
        let exact_remainder = div_rem_decimal_chunk(&mut exact_value);
        assert_eq!(exact_value, InternalMpUint::from_limb(1));
        assert_eq!(exact_remainder, 0);

        // `divisor << LIMB_BITS`: quotient `1 << LIMB_BITS`, remainder zero.
        let mut shifted_value = exact;
        shifted_value.shl_assign(LIMB_BITS);
        let shifted_remainder = div_rem_decimal_chunk(&mut shifted_value);
        assert_eq!(shifted_value, InternalMpUint::from_limbs(alloc::vec![0, 1]));
        assert_eq!(shifted_remainder, 0);
    }
}
