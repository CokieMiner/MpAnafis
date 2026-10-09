//! Limb-sized radix parameters, digit bounds, and compile-time ASCII tables.

#![expect(
    unsafe_code,
    reason = "Validated radices and digit widths bound ASCII arithmetic, const-table indices, shifts, and normalized reciprocal calculations"
)]

use core::num::NonZeroU32;

use super::{
    DoubleLimb, Limb, RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD,
    RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD, RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD,
    RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD, RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD,
    RADIX_PARSE_LEAF_CHUNKS, RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD,
};

/// Limb-sized parameters shared by radix parsing and formatting.
#[derive(Debug, Clone, Copy)]
pub struct RadixParameters {
    /// Largest `k` such that `radix^k` fits in a [`Limb`].
    pub max_digits: usize,
    /// `radix^max_digits`.
    pub max_power: Limb,
}

/// Namespace for radix conversion and formatting helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Convert;

pub const BINARY_BYTE_DIGITS: [[u8; 8]; 256] = power_of_two_byte_digits::<8>(1);
pub const BASE4_BYTE_DIGITS: [[u8; 4]; 256] = power_of_two_byte_digits::<4>(2);
pub const HEX_BYTE_DIGITS: [[u8; 2]; 256] = power_of_two_byte_digits::<2>(4);

// Both block radices use the same table shape; masked digits occupy at most
// five bits, and radix eight uses only the first eight entries.
pub const BASE8_DIGITS: [u8; 32] = power_of_two_digit_bytes::<3>();
pub const BASE32_DIGITS: [u8; 32] = power_of_two_digit_bytes::<5>();

/// Normalized single-limb reciprocals of the maximal native radix powers.
/// Entry `r` is `floor((2^(2W)-1) / (r^m << leading_zeros(r^m))) - 2^W`.
pub const RADIX_CHUNK_RECIPROCALS: [Limb; 37] = radix_chunk_reciprocals();

const RADIX_PARAMETERS: [(usize, Limb); 37] = radix_parameters();

impl RadixParameters {
    /// Returns the limb-sized parameters for a radix in `2..=36`.
    #[inline]
    pub const fn for_limb(radix: u32) -> Self {
        debug_assert!(
            matches!(radix, 2..=36),
            "radix parameters require radix 2..=36"
        );
        #[expect(
            clippy::as_conversions,
            reason = "A validated radix in 2..=36 fits usize on every supported pointer width and converts in const evaluation"
        )]
        let index = radix as usize;
        // SAFETY: conversion dispatchers validate 2 <= radix <= 36, so the
        // index selects one of the table's 37 initialized entries.
        let (max_digits, max_power) = unsafe { *RADIX_PARAMETERS.as_ptr().add(index) };
        Self {
            max_digits,
            max_power,
        }
    }
}

impl Convert {
    /// Bounds the digit count for a caller-validated radix in `2..=36`.
    ///
    /// For `k = floor(log2(radix))`, `radix >= 2^k`. A value with `bits`
    /// significant bits is below `2^bits <= radix^ceil(bits / k)`, so the
    /// ceiling alone bounds its digit count. It remains representable even
    /// for `bits == usize::MAX`; no additive slack or rounded numerator is needed.
    pub const fn estimated_digits(significant_bits: usize, radix: u32) -> usize {
        debug_assert!(
            radix >= 2 && radix <= 36,
            "digit estimation requires radix 2..=36"
        );
        // SAFETY: every conversion dispatcher validates radix >= 2. Supplying
        // that nonzero invariant directly avoids an impossible logarithm panic.
        let floor = unsafe { NonZeroU32::new_unchecked(radix) }.ilog2();
        match floor {
            1 => significant_bits,
            2 => significant_bits.div_ceil(2),
            3 => significant_bits.div_ceil(3),
            4 => significant_bits.div_ceil(4),
            _ => {
                // radix <= 36 bounds the final logarithm by five.
                debug_assert!(floor == 5, "the remaining radix logarithm is five");
                significant_bits.div_ceil(5)
            }
        }
    }

    /// Returns the schoolbook-to-recursive threshold for a validated radix.
    ///
    /// Callers handle invalid radices and powers of two before this dispatch.
    #[inline]
    pub const fn recursive_threshold(radix: u32) -> usize {
        debug_assert!(
            radix >= 3 && radix <= 36 && !radix.is_power_of_two(),
            "recursive thresholds require a supported non-power-of-two radix"
        );
        match radix {
            3..=9 => RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD,
            10 => RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD,
            _ => {
                debug_assert!(radix >= 11 && radix <= 36, "the remaining radix is large");
                RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD
            }
        }
    }

    /// Returns parsing entry and leaf cutoffs, both in limb-sized radix chunks.
    ///
    /// The profile validates entry > leaf >= 1. Multiplication dispatch uses
    /// its own independent thresholds inside the recursive combine products.
    #[inline]
    pub const fn parsing_thresholds(radix: u32) -> (usize, usize) {
        debug_assert!(
            radix >= 3 && radix <= 36 && !radix.is_power_of_two(),
            "parsing thresholds require a supported non-power-of-two radix"
        );
        let entry = match radix {
            3..=9 => RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD,
            10 => RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD,
            _ => {
                debug_assert!(radix >= 11 && radix <= 36, "the remaining radix is large");
                RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD
            }
        };
        (entry, RADIX_PARSE_LEAF_CHUNKS)
    }

    /// Maps a caller-proved digit in `0..=35` to its lowercase ASCII byte.
    pub const fn byte_from_digit(digit: u8) -> u8 {
        debug_assert!(digit < 36, "radix digits lie in 0..=35");
        if digit < 10 {
            // SAFETY: digit <= 9 gives at most ASCII '9' = 57, within u8.
            unsafe { b'0'.unchecked_add(digit) }
        } else {
            // SAFETY: 10 <= digit <= 35 gives a letter offset in 0..=25;
            // ASCII 'a' + offset <= 'z' = 122 fits u8 on every target.
            unsafe { b'a'.unchecked_add(digit.unchecked_sub(10)) }
        }
    }
}

/// Derives exact native powers during constant evaluation on the target.
#[expect(
    clippy::indexing_slicing,
    reason = "constant evaluation visits only radix indices 2..=36 in the 37-entry table"
)]
const fn radix_parameters() -> [(usize, Limb); 37] {
    let mut table = [(0, 0); 37];
    let mut radix = 2_usize;
    while radix <= 36 {
        let mut power = 1_usize;
        let mut digits = 0_usize;
        while let Some(next) = power.checked_mul(radix) {
            power = next;
            // SAFETY: radix>=2 permits fewer than Limb::BITS<=64 powers.
            digits = unsafe { digits.unchecked_add(1) };
        }
        table[radix] = (digits, power);
        // SAFETY: radix<=36 bounds its successor by 37 on every target.
        radix = unsafe { radix.unchecked_add(1) };
    }
    table
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "Const evaluation masks each digit to four bits and bounds both fixed-array indices; no runtime checks or casts remain"
)]
const fn power_of_two_byte_digits<const DIGITS_PER_BYTE: usize>(
    bits_per_digit: u32,
) -> [[u8; DIGITS_PER_BYTE]; 256] {
    // SAFETY: the three instantiations use digit widths 1, 2, and 4; each
    // shift is below the minimum 16-bit limb width and produces a positive mask.
    let digit_mask = unsafe { 1_usize.unchecked_shl(bits_per_digit).unchecked_sub(1) };
    let mut table = [[b'0'; DIGITS_PER_BYTE]; 256];
    let mut byte = 0_usize;
    while byte < 256 {
        let mut value = byte;
        let mut index = DIGITS_PER_BYTE;
        while index != 0 {
            // SAFETY: index != 0 bounds the descending table position.
            index = unsafe { index.unchecked_sub(1) };
            table[byte][index] = Convert::byte_from_digit((value & digit_mask) as u8);
            value = value.wrapping_shr(bits_per_digit);
        }
        // SAFETY: byte < 256 gives at most 256, fitting every native width.
        byte = unsafe { byte.unchecked_add(1) };
    }
    table
}

/// Builds a common 32-entry ASCII table for three- or five-bit digits.
#[expect(
    clippy::as_conversions,
    clippy::indexing_slicing,
    reason = "Const evaluation bounds digit to 0..=31 within the fixed table; its widening conversion fits all supported pointer widths"
)]
const fn power_of_two_digit_bytes<const BITS_PER_DIGIT: u32>() -> [u8; 32] {
    // SAFETY: the two instantiations use digit widths 3 and 5, both below
    // eight; the shifted one is positive and permits subtraction.
    let digit_mask = unsafe { 1_u8.unchecked_shl(BITS_PER_DIGIT).unchecked_sub(1) };
    let mut table = [b'0'; 32];
    let mut digit = 0_u8;
    while digit < 32 {
        table[digit as usize] = Convert::byte_from_digit(digit & digit_mask);
        // SAFETY: digit < 32 bounds its increment by 32, within u8.
        digit = unsafe { digit.unchecked_add(1) };
    }
    table
}

#[expect(
    clippy::as_conversions,
    clippy::indexing_slicing,
    reason = "Const evaluation bounds radix indices to 2..=36 and reciprocals below 2^W; widening Limb to DoubleLimb is exact on every target"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "quotient-2^W is strictly below 2^W and fits Limb on 32-bit targets"
    )
)]
const fn radix_chunk_reciprocals() -> [Limb; 37] {
    let mut table = [0; 37];
    let mut radix = 2_u32;
    while radix <= 36 {
        let power = RadixParameters::for_limb(radix).max_power;
        // SAFETY: each supported radix power is nonzero, so leading_zeros()
        // is below W and normalization gives a divisor in [2^(W-1), 2^W).
        let normalized = unsafe { power.unchecked_shl(power.leading_zeros()) };
        // SAFETY: normalization preserves a nonzero divisor; its widened
        // value permits unsigned division without zero or signed overflow.
        let quotient = unsafe {
            DoubleLimb::MAX
                .checked_div(normalized as DoubleLimb)
                .unwrap_unchecked()
        };
        let unit: DoubleLimb = 1;
        // SAFETY: DoubleLimb has 2W bits on every supported pointer width.
        let limb_base = unsafe { unit.unchecked_shl(Limb::BITS) };
        // SAFETY: the divisor bounds quotient in [2^W, 2^(W+1)); subtracting
        // 2^W leaves a reciprocal below 2^W on every supported pointer width.
        let reciprocal = unsafe { quotient.unchecked_sub(limb_base) };
        table[radix as usize] = reciprocal as Limb;
        // SAFETY: radix <= 36 bounds its increment by 37, within u32.
        radix = unsafe { radix.unchecked_add(1) };
    }
    table
}
