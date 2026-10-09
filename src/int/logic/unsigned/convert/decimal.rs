//! Decimal chunk parsing and single-limb radix division.

#![expect(
    unsafe_code,
    reason = "Validated decimal blocks bound packed arithmetic; nonzero divisors and initialized limb spans establish the division bounds"
)]

#[cfg(not(target_pointer_width = "16"))]
use core::ptr::read_unaligned;

use alloc::{string::String, vec::Vec};

use crate::error::{ParseMpUintError, ParseMpUintErrorKind};

use super::{Convert, Division, InternalMpUint, Limb, RADIX_CHUNK_RECIPROCALS, RadixParameters};

/// `floor((2^128 - 1) / 10^19) - 2^64` for the normalized decimal divisor.
#[cfg(target_pointer_width = "64")]
#[expect(
    clippy::as_conversions,
    reason = "radix ten selects an initialized const-table entry; Limb and u64 coincide on 64-bit targets"
)]
const DECIMAL_CHUNK_PREINV: u64 = RADIX_CHUNK_RECIPROCALS[10] as u64;

impl InternalMpUint {
    /// Formats the integer as a string in the given radix using the schoolbook algorithm.
    ///
    /// Extracts limb-sized radix blocks, with a constant divisor for decimal.
    /// The caller validates the radix and routes zero, powers of two, and the recursive
    /// domain before invoking this path.
    pub fn format_schoolbook_string(&self, radix: u32) -> String {
        let significant_bits = self.significant_bits();
        let mut output = Vec::with_capacity(Convert::estimated_digits(significant_bits, radix));
        let mut value = self.clone();

        if radix == 10 {
            Convert::write_decimal_chunks(&mut value, &mut output);
        } else {
            #[expect(
                clippy::as_conversions,
                reason = "radix is checked to be in 2..=36 and therefore fits in Limb"
            )]
            let radix_limb = radix as Limb;
            let parameters = RadixParameters::for_limb(radix);
            // SAFETY: radix is validated in 3..=36 and selects the reciprocal
            // of the normalized parameters.max_power divisor.
            let reciprocal = unsafe { *RADIX_CHUNK_RECIPROCALS.get_unchecked(radix_limb) };
            loop {
                let remainder =
                    Convert::div_rem_small(&mut value, parameters.max_power, reciprocal);
                let full_width = !value.is_zero();
                Convert::write_radix_chunk::<true>(
                    remainder,
                    radix_limb,
                    parameters,
                    full_width,
                    &mut output,
                );
                if !full_width {
                    break;
                }
            }
        }

        output.reverse();
        // SAFETY: byte_from_digit produces ASCII decimal digits and lowercase letters only.
        unsafe { String::from_utf8_unchecked(output) }
    }
}

impl Convert {
    // Decimal chunking: radix 10 divides by a single-limb power of ten per step,
    // emitting `DECIMAL_CHUNK_DIGITS` digits per division instead of one. The
    // power of ten is the largest that fits a `Limb` on the target: 10^19 < 2^64,
    // 10^9 < 2^32, and 10^4 < 2^16, so the remainder of every chunked division
    // fits in a single limb on all three supported pointer widths.
    pub const DECIMAL_CHUNK_DIVISOR: Limb = RadixParameters::for_limb(10).max_power;
    pub const DECIMAL_CHUNK_DIGITS: usize = RadixParameters::for_limb(10).max_digits;

    /// Divides `value` by a single-limb `divisor`, storing the quotient in place
    /// and returning the remainder.
    ///
    /// The caller must establish `divisor > 0` and supply
    /// `floor((2^(2W)-1) / (divisor << leading_zeros(divisor))) - 2^W`
    /// as its normalized reciprocal. Every limb step preserves a normalized
    /// remainder below the divisor.
    pub fn div_rem_small(value: &mut InternalMpUint, divisor: Limb, reciprocal: Limb) -> Limb {
        debug_assert_ne!(divisor, 0, "radix division requires a nonzero divisor");
        let shift = divisor.leading_zeros();
        // SAFETY: divisor > 0 gives shift < W; normalization discards no bits.
        let d_norm = unsafe { divisor.unchecked_shl(shift) };
        let mut remainder = 0;
        let limbs = value.limbs_mut();
        if shift == 0 {
            for limb in limbs.iter_mut().rev() {
                (*limb, remainder) =
                    Division::divrem_2by1_reciprocal(remainder, *limb, d_norm, reciprocal);
            }
        } else {
            // SAFETY: a nonzero divisor has shift < LIMB_BITS, and this branch
            // has shift > 0, so the complementary limb shift is in 1..W.
            let complement = unsafe { Limb::BITS.unchecked_sub(shift) };
            for limb in limbs.iter_mut().rev() {
                // Normalized remainders are multiples of 2^shift. The next
                // limb's high bits occupy only the zero low-bit span, so OR
                // appends them without undoing and redoing normalization.
                let high = remainder | (*limb >> complement);
                (*limb, remainder) =
                    Division::divrem_2by1_reciprocal(high, *limb << shift, d_norm, reciprocal);
            }
        }

        value.normalize();
        remainder >> shift
    }

    /// Divides `value` by the decimal chunk divisor and returns the remainder.
    #[cfg(target_pointer_width = "64")]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "On 64-bit targets Limb and u64 coincide; the u128 product is split into its exact high limb and low limb modulo 2^64"
    )]
    #[inline]
    pub fn div_rem_decimal_chunk(value: &mut InternalMpUint) -> Limb {
        // `DECIMAL_CHUNK_DIVISOR` is `10^19` on 64-bit targets, so this is the
        // identity conversion; the shared constant keeps a single source of truth.
        const D: u64 = Convert::DECIMAL_CHUNK_DIVISOR as u64;
        const DI: u64 = DECIMAL_CHUNK_PREINV;

        let mut remainder = 0_u64;

        for limb in value.limbs_mut().iter_mut().rev() {
            let low = *limb as u64;
            let high = remainder;

            // SAFETY: both factors fit u64; their product is below 2^128.
            let product = unsafe { u128::from(high).unchecked_mul(u128::from(DI)) };
            let product_high = product.wrapping_shr(64) as u64;
            let product_low = product as u64;

            let (estimate_low, carry) = product_low.overflowing_add(low);

            // Estimates and correction residues are carried modulo 2^64;
            // the two corrections restore the exact quotient and a remainder < D.
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

            *limb = quotient as Limb;

            remainder = next_remainder;
        }

        value.normalize();
        remainder as Limb
    }

    /// Divides `value` by the decimal chunk divisor and returns the remainder.
    ///
    /// Uses the normalized radix-10 reciprocal for each limb division.
    #[cfg(not(target_pointer_width = "64"))]
    #[inline]
    pub fn div_rem_decimal_chunk(value: &mut InternalMpUint) -> Limb {
        // SAFETY: radix 10 selects the normalized decimal chunk reciprocal.
        let reciprocal = unsafe { *RADIX_CHUNK_RECIPROCALS.get_unchecked(10) };
        Self::div_rem_small(value, Self::DECIMAL_CHUNK_DIVISOR, reciprocal)
    }

    /// Validates and decodes at most [`Self::DECIMAL_CHUNK_DIGITS`] ASCII decimal digits.
    #[inline]
    pub fn parse_decimal_chunk(bytes: &[u8]) -> Result<Limb, ParseMpUintError> {
        debug_assert!(
            bytes.len() <= Self::DECIMAL_CHUNK_DIGITS,
            "decimal chunks fit one limb"
        );
        let mut value: Limb = 0;
        #[cfg(not(target_pointer_width = "16"))]
        let remainder = {
            let mut remaining = bytes;
            while remaining.len() >= 8 {
                // SAFETY: the slice holds eight initialized bytes; an unaligned
                // read accepts every byte address. Little-endian decoding assigns
                // the first digit to the least-significant packed byte on every target.
                let word =
                    u64::from_le(unsafe { read_unaligned(remaining.as_ptr().cast::<u64>()) });
                // Arithmetic here is modulo 2^64. Valid bytes produce neither
                // a borrow when subtracting 0x30 nor a carry when adding 0x46.
                // At the first invalid byte, at least one operation sets its high
                // bit; preceding valid lanes cannot hide that flag.
                let digits = word.wrapping_sub(0x3030_3030_3030_3030);
                if (digits | word.wrapping_add(0x4646_4646_4646_4646)) & 0x8080_8080_8080_8080 != 0
                {
                    return Err(ParseMpUintError::new(ParseMpUintErrorKind::InvalidDigit));
                }
                // SAFETY: the guard proves each packed digit is in 0..=9. The
                // 16-bit pairs stay below 100 and the 32-bit groups below 10_000;
                // no product or addition crosses its lane. The final eight-digit
                // value is below 100_000_000 and fits 32- and 64-bit Limb. The
                // accumulated prefix has at most DECIMAL_CHUNK_DIGITS digits,
                // so its scaled value and sum remain below DECIMAL_CHUNK_DIVISOR.
                unsafe {
                    let pairs = (digits & 0x00ff_00ff_00ff_00ff)
                        .unchecked_mul(10)
                        .unchecked_add((digits >> 8) & 0x00ff_00ff_00ff_00ff);
                    let groups = (pairs & 0x0000_ffff_0000_ffff)
                        .unchecked_mul(100)
                        .unchecked_add((pairs >> 16) & 0x0000_ffff_0000_ffff);
                    let block = (groups & 0xffff_ffff)
                        .unchecked_mul(10_000)
                        .unchecked_add(groups >> 32);
                    value = value
                        .unchecked_mul(100_000_000)
                        .unchecked_add(Limb::try_from(block).unwrap_unchecked());
                    remaining = remaining.get_unchecked(8..);
                }
            }
            remaining
        };
        #[cfg(target_pointer_width = "16")]
        let remainder = bytes;
        for &byte in remainder {
            let digit = byte.wrapping_sub(b'0');
            if digit >= 10 {
                return Err(ParseMpUintError::new(ParseMpUintErrorKind::InvalidDigit));
            }
            // SAFETY: digit <= 9 and the complete chunk is limb-sized. Every
            // validated prefix, including its next product and sum, is below
            // 10^DECIMAL_CHUNK_DIGITS, which fits Limb on 16/32/64-bit targets.
            value = unsafe { value.unchecked_mul(10).unchecked_add(Limb::from(digit)) };
        }
        Ok(value)
    }
}
