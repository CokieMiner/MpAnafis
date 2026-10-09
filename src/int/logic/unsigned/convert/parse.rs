//! ASCII validation and limb-sized radix chunk decoding.
//!
//! Reference: R. P. Brent and P. Zimmermann, "Modern Computer Arithmetic",
//! Cambridge University Press, 2011, Section 1.7 (base conversion).

#![expect(
    unsafe_code,
    reason = "Validated radix parameters and checked significant widths bound source chunks, initialize reserved destinations, and prove normalized length commitments"
)]

use core::num::NonZeroUsize;

use alloc::vec::Vec;

use crate::error::{ParseMpUintError, ParseMpUintErrorKind};

use super::{Convert, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, RadixParameters};

impl InternalMpUint {
    /// Parses unsigned ASCII digits in `radix`, accepting either letter case.
    ///
    /// # Errors
    /// Rejects empty input, radices outside `2..=36`, negative signs, invalid
    /// digits, and an output bit width that cannot be represented by `usize`.
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseMpUintError> {
        if !(2..=36).contains(&radix) {
            return Err(ParseMpUintError::new(ParseMpUintErrorKind::InvalidRadix));
        }
        if s.is_empty() {
            return Err(ParseMpUintError::new(ParseMpUintErrorKind::Empty));
        }
        if s.starts_with('-') {
            return Err(ParseMpUintError::new(ParseMpUintErrorKind::Negative));
        }
        let significant = s.trim_start_matches('0');
        if significant.is_empty() {
            return Ok(Self::zero());
        }
        if radix.is_power_of_two() {
            return parse_power_of_two(significant.as_bytes(), radix);
        }
        let parameters = RadixParameters::for_limb(radix);
        let (entry, leaf) = Convert::parsing_thresholds(radix);
        debug_assert!(entry > leaf, "the parsing entry exceeds its leaf cutoff");
        // SAFETY: validated profiles have entry > leaf >= 1.
        let schoolbook_chunks = unsafe { entry.unchecked_sub(1) };
        // Saturation selects a tier for every addressable string; it does not
        // determine a buffer size.
        let recursive = significant.len() > parameters.max_digits.saturating_mul(schoolbook_chunks);
        Self::parse_non_power_of_two(significant.as_bytes(), radix, parameters, recursive, leaf)
    }

    /// Parses nonempty bytes with parameters for a non-power-of-two `radix`.
    ///
    /// `radix` is in `3..=36` and `leaf_chunks` is nonzero. Both tiers validate
    /// all digits before returning a magnitude.
    ///
    /// # Errors
    /// Returns an invalid-digit error for bytes outside the supplied radix.
    pub fn parse_non_power_of_two(
        bytes: &[u8],
        radix: u32,
        parameters: RadixParameters,
        recursive: bool,
        leaf_chunks: usize,
    ) -> Result<Self, ParseMpUintError> {
        debug_assert!(
            (3..=36).contains(&radix) && !radix.is_power_of_two() && !bytes.is_empty(),
            "parsing requires a validated radix and nonempty input"
        );
        debug_assert!(
            leaf_chunks != 0,
            "parsing leaves contain at least one chunk"
        );
        if radix == 10 {
            parse_digits::<true>(
                bytes,
                10,
                RadixParameters::for_limb(10),
                recursive,
                leaf_chunks,
            )
        } else {
            parse_digits::<false>(bytes, radix, parameters, recursive, leaf_chunks)
        }
    }
}

/// Validates chunks before schoolbook or recursive reconstruction.
fn parse_digits<const DECIMAL: bool>(
    bytes: &[u8],
    radix: u32,
    parameters: RadixParameters,
    recursive: bool,
    leaf_chunks: usize,
) -> Result<InternalMpUint, ParseMpUintError> {
    let RadixParameters {
        max_digits,
        max_power,
    } = parameters;
    debug_assert_ne!(
        max_digits, 0,
        "every supported radix has a limb-sized chunk"
    );
    // SAFETY: the validated radix parameters supply a positive chunk width.
    let divisor = unsafe { NonZeroUsize::new_unchecked(max_digits) };
    let chunk_count = bytes.len().div_ceil(divisor.get());
    if !recursive {
        // SAFETY: nonempty bytes and a positive divisor give a first width in
        // 1..=min(max_digits, bytes.len()) on every pointer width.
        let first_width = unsafe {
            bytes
                .len()
                .unchecked_sub(1)
                .rem_euclid(divisor.get())
                .unchecked_add(1)
        };
        // SAFETY: the first chunk fits the initialized input slice.
        let first_bytes = unsafe { bytes.get_unchecked(..first_width) };
        let first = if DECIMAL {
            Convert::parse_decimal_chunk(first_bytes)?
        } else {
            parse_radix_chunk(first_bytes, radix)?
        };
        if chunk_count == 1 {
            return Ok(InternalMpUint::from_limb(first));
        }
        // Six native radix chunks can fit four binary limbs. Stack guards
        // retain inline storage until the reconstructed magnitude needs more.
        let mut inline = [0; INLINE_LIMBS + 2];
        let small = chunk_count <= inline.len();
        let mut result = InternalMpUint::zero();
        let dst = if small {
            inline.as_mut_ptr()
        } else {
            result.prepare_limb_write(chunk_count).as_mut_ptr()
        };
        // SAFETY: either exclusive destination reserves at least two aligned limbs.
        unsafe {
            dst.write(first);
        }
        let mut len = usize::from(first != 0);
        // SAFETY: first_width <= bytes.len(); the remainder contains whole chunks.
        for chunk in unsafe { bytes.get_unchecked(first_width..) }.chunks_exact(divisor.get()) {
            let add = if DECIMAL {
                Convert::parse_decimal_chunk(chunk)?
            } else {
                parse_radix_chunk(chunk, radix)?
            };
            // SAFETY: j chunks represent less than max_power^j < B^j. Before
            // the next chunk, len <= j < chunk_count leaves one aligned guard;
            // the initialized prefix is exclusive and add < max_power.
            len = unsafe { Convert::mul_small_add(dst, len, max_power, add) };
        }
        if small {
            if len <= INLINE_LIMBS {
                return Ok(InternalMpUint::from_limbs_4(
                    inline[0], inline[1], inline[2], inline[3],
                ));
            }
            // SAFETY: len <= chunk_count <= inline.len(); all active limbs are
            // initialized and the last carry established a nonzero top limb.
            return Ok(unsafe {
                InternalMpUint::from_limbs_normalized(inline.get_unchecked(..len).to_vec())
            });
        }
        // SAFETY: the fixed reservation contains the initialized normalized
        // len-limb prefix. No operation invalidated dst before this commitment.
        unsafe {
            result.set_len(len);
        }
        return Ok(result);
    }
    let mut chunks: Vec<Limb> = Vec::with_capacity(chunk_count);
    let dst = chunks.as_mut_ptr();
    for (index, group) in bytes.rchunks(divisor.get()).enumerate() {
        let chunk = if DECIMAL {
            Convert::parse_decimal_chunk(group)?
        } else {
            parse_radix_chunk(group, radix)?
        };
        // SAFETY: rchunks produces chunk_count groups. Each in-bounds exclusive
        // slot receives its first write; an error drops the vector at length zero.
        unsafe {
            dst.add(index).write(chunk);
        }
    }
    // SAFETY: every reserved chunk slot has been initialized exactly once.
    unsafe {
        chunks.set_len(chunk_count);
    }
    Ok(Convert::reconstruct_chunks(&chunks, max_power, leaf_chunks))
}

/// Packs radix-2^k digits directly into limbs, with `k` in `1..=5`.
#[expect(
    clippy::as_conversions,
    reason = "Digits are below 32 and digit widths are at most five, fitting 16-, 32-, and 64-bit limbs"
)]
fn parse_power_of_two(bytes: &[u8], radix: u32) -> Result<InternalMpUint, ParseMpUintError> {
    let digit_bits = radix.trailing_zeros() as usize;
    // SAFETY: the driver removed leading zero digits and returned for empty input.
    let first = digit_from_ascii_byte(unsafe { *bytes.first().unwrap_unchecked() }, radix)
        .ok_or(ParseMpUintError::new(ParseMpUintErrorKind::InvalidDigit))?;
    // SAFETY: leading_zeros() is in 0..=u32::BITS; the difference is at most
    // five for a supported power-of-two radix and fits every native width.
    let top_bits = unsafe { u32::BITS.unchecked_sub(first.leading_zeros()) } as usize;
    // SAFETY: the first valid byte establishes a nonempty input.
    let lower_digits = unsafe { bytes.len().unchecked_sub(1) };
    let bits = lower_digits
        .checked_mul(digit_bits)
        .and_then(|lower| lower.checked_add(top_bits))
        .ok_or(ParseMpUintError::new(ParseMpUintErrorKind::TooLarge))?;
    let num_limbs = bits.div_ceil(LIMB_BITS);
    let mut result = InternalMpUint::with_capacity(num_limbs);
    let mut write = result.prepare_limb_write(num_limbs);
    let dst = write.as_mut_ptr();
    let mut window: Limb = 0;
    let mut held = 0_usize;
    let mut index = 0_usize;
    for &byte in bytes.iter().rev() {
        let digit = digit_from_ascii_byte(byte, radix)
            .ok_or(ParseMpUintError::new(ParseMpUintErrorKind::InvalidDigit))?;
        // held < LIMB_BITS; the low held bits are disjoint from the next digit.
        let digit_limb = digit as Limb;
        window |= digit_limb << held;
        // SAFETY: held < LIMB_BITS <= 64 and digit_bits <= 5, so their sum
        // is at most 68 and fits usize on all supported pointer widths.
        held = unsafe { held.unchecked_add(digit_bits) };
        if held >= LIMB_BITS {
            // SAFETY: complete significant windows satisfy index < num_limbs.
            // The write guard supplies aligned exclusive storage for each window.
            unsafe {
                dst.add(index).write(window);
            }
            // SAFETY: index < num_limbs bounds its increment; held >=
            // LIMB_BITS permits subtraction. Its residual is below digit_bits,
            // so the complementary digit shift is in 1..=digit_bits <= 5.
            unsafe {
                index = index.unchecked_add(1);
                held = held.unchecked_sub(LIMB_BITS);
                window = digit_limb >> digit_bits.unchecked_sub(held);
            }
        }
    }
    if index < num_limbs {
        // SAFETY: at most one partial limb remains, within the reserved span.
        unsafe {
            dst.add(index).write(window);
        }
    }
    // SAFETY: complete windows and the partial top window initialize num_limbs;
    // the first nonzero digit establishes a nonzero highest limb.
    let _ = unsafe { write.commit() };
    Ok(result)
}

/// Validates and accumulates at most `max_digits` bytes in a native limb.
#[inline]
fn parse_radix_chunk(bytes: &[u8], radix: u32) -> Result<Limb, ParseMpUintError> {
    // SAFETY: radix <= 36 fits every supported limb width.
    let base = unsafe { Limb::try_from(radix).unwrap_unchecked() };
    let mut value: Limb = 0;
    for &byte in bytes {
        let digit = digit_from_ascii_byte(byte, radix)
            .ok_or(ParseMpUintError::new(ParseMpUintErrorKind::InvalidDigit))?;
        // SAFETY: digit < radix <= 36 fits Limb. The chunk has at most
        // max_digits bytes, so every prefix stays below radix^max_digits <= Limb::MAX.
        value = unsafe {
            value
                .unchecked_mul(base)
                .unchecked_add(Limb::try_from(digit).unwrap_unchecked())
        };
    }
    Ok(value)
}

/// Decodes one ASCII digit and rejects values outside `radix`.
fn digit_from_ascii_byte(byte: u8, radix: u32) -> Option<u32> {
    let digit = match byte {
        // SAFETY: the matched digit range bounds the difference in 0..=9.
        b'0'..=b'9' => u32::from(unsafe { byte.unchecked_sub(b'0') }),
        // SAFETY: the matched letter range bounds the difference in 0..=25;
        // adding ten gives at most 35, within u8 and u32 on every target.
        b'a'..=b'z' => u32::from(unsafe { byte.unchecked_sub(b'a').unchecked_add(10) }),
        // SAFETY: the matched letter range bounds the difference in 0..=25;
        // adding ten gives at most 35, within u8 and u32 on every target.
        b'A'..=b'Z' => u32::from(unsafe { byte.unchecked_sub(b'A').unchecked_add(10) }),
        _ => return None,
    };
    (digit < radix).then_some(digit)
}
