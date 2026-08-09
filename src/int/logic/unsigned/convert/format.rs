//! Radix string formatting entry points and power-of-two digit paths.

#![allow(
    unsafe_code,
    reason = "Radix is asserted in 2..=36; digit and byte tables are indexed by proved 0..=31 / 0..=255 / byte bounds; row copies stay within the recorded output range; byte extraction stays below ceil(significant_bits / 8); limb loops use index < len; emitted bytes are ASCII."
)]

#[cfg(feature = "std")]
use core::cell::RefCell;
use core::{
    fmt::{Display, Error, Formatter, Result as FmtResult, Write},
    hint::unreachable_unchecked,
    ptr::copy_nonoverlapping,
};
#[cfg(feature = "std")]
use std::thread_local;

use alloc::{string::String, vec::Vec};

use super::{
    FormatCache, InternalMpUint, LIMB_BYTES, Limb, RADIX_DECIMAL_RECURSIVE_THRESHOLD,
    RADIX_LARGE_RECURSIVE_THRESHOLD, RADIX_SMALL_RECURSIVE_THRESHOLD,
};

const BINARY_BYTE_DIGITS: [[u8; 8]; 256] = power_of_two_byte_digits::<8>(1);
const BASE4_BYTE_DIGITS: [[u8; 4]; 256] = power_of_two_byte_digits::<4>(2);
const HEX_BYTE_DIGITS: [[u8; 2]; 256] = power_of_two_byte_digits::<2>(4);

// Radix 8 and 32 digit tables: radix 8 uses digits 0..=7 and radix 32 uses
// digits 0..=31 mapped to `b'0'..=b'9'` then `b'a'..=b'v'`. Both radices
// share the 32-entry lookup shape, and a table load removes the per-digit
// branch of `byte_from_digit`.
const BASE8_DIGITS: [u8; 32] = power_of_two_digit_bytes::<3>();
const BASE32_DIGITS: [u8; 32] = power_of_two_digit_bytes::<5>();

impl InternalMpUint {
    /// Formats the integer as a string in the given radix (2..=36).
    ///
    /// Uses lowercase letters for digits above 9. Invalid radices, zero,
    /// powers of two, and schoolbook-sized values are handled before the
    /// thread-local cache is touched; the recursive divide-and-conquer path
    /// falls back to a fresh cache when the cached borrow is already held
    /// (reentrant formatting) or when `std` is disabled.
    pub fn format_radix_writer(&self, radix: u32, w: &mut dyn Write) -> FmtResult {
        if !(2..=36).contains(&radix) {
            return Err(Error);
        }

        if self.is_zero() {
            return w.write_str("0");
        }

        if radix.is_power_of_two() {
            let string = format_power_of_two(self, radix);
            return w.write_str(&string);
        }

        if self.limbs().len() < recursive_threshold(radix) {
            let string = self.format_schoolbook_string(radix);
            return w.write_str(&string);
        }

        #[cfg(feature = "std")]
        {
            FORMAT_CACHE.with(|tls_cache| {
                if let Ok(mut slot) = tls_cache.try_borrow_mut() {
                    let cache = slot.get_or_insert_with(FormatCache::new);
                    self.format_recursive_writer_with_cache(radix, w, cache)
                } else {
                    let mut fallback_cache = FormatCache::new();
                    self.format_recursive_writer_with_cache(radix, w, &mut fallback_cache)
                }
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let mut cache = FormatCache::new();
            self.format_recursive_writer_with_cache(radix, w, &mut cache)
        }
    }
}

impl Display for InternalMpUint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        if formatter.width().is_some() || formatter.precision().is_some() {
            let string = self.to_string_radix(10);
            formatter.pad_integral(true, "", &string)
        } else {
            self.format_radix_writer(10, formatter)
        }
    }
}

impl InternalMpUint {
    /// Formats the integer as a string in the given radix (2..=36).
    ///
    /// Uses lowercase letters for digits above 9.
    ///
    /// # Panics
    ///
    /// Panics if `radix` is outside `2..=36`.
    #[must_use]
    #[track_caller]
    pub fn to_string_radix(&self, radix: u32) -> String {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");

        if self.is_zero() {
            return String::from("0");
        }

        if radix.is_power_of_two() {
            return format_power_of_two(self, radix);
        }

        if self.limbs().len() < recursive_threshold(radix) {
            return self.format_schoolbook_string(radix);
        }

        // The output stays a concrete `String` so the recursive writer keeps
        // static dispatch; only the slow path touches the thread-local cache.
        let mut output = String::with_capacity(estimated_digits(self.significant_bits(), radix));

        #[cfg(feature = "std")]
        let fmt_result = FORMAT_CACHE.with(|tls_cache| {
            if let Ok(mut slot) = tls_cache.try_borrow_mut() {
                let cache = slot.get_or_insert_with(FormatCache::new);
                self.format_recursive_writer_with_cache(radix, &mut output, cache)
            } else {
                let mut fallback_cache = FormatCache::new();
                self.format_recursive_writer_with_cache(radix, &mut output, &mut fallback_cache)
            }
        });
        #[cfg(not(feature = "std"))]
        let fmt_result = {
            let mut cache = FormatCache::new();
            self.format_recursive_writer_with_cache(radix, &mut output, &mut cache)
        };
        fmt_result.expect("writing to String is infallible");
        output
    }

    /// Forced schoolbook digit-extraction path for the crossover tuner.
    ///
    /// Always uses the linear per-digit division loop regardless of limb count,
    /// bypassing the recursive threshold.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub fn to_string_radix_schoolbook(&self, radix: u32) -> String {
        debug_assert!(
            (2..=36).contains(&radix) && !radix.is_power_of_two(),
            "forced schoolbook path requires a non-power-of-two radix in 2..=36"
        );
        self.format_schoolbook_string(radix)
    }

    /// Forced recursive Barrett divide-and-conquer path for the crossover tuner.
    ///
    /// Always uses the recursive path regardless of limb count, bypassing
    /// the recursive threshold. The caller's `cache` is reused across calls,
    /// so the tuning harness should warm it once before taking timed samples.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub fn to_string_radix_recursive_with_cache(
        &self,
        radix: u32,
        cache: &mut FormatCache,
    ) -> String {
        debug_assert!(
            (2..=36).contains(&radix) && !radix.is_power_of_two(),
            "forced recursive path requires a non-power-of-two radix in 2..=36"
        );
        let mut output = String::with_capacity(estimated_digits(self.significant_bits(), radix));
        self.format_recursive_writer_with_cache(radix, &mut output, cache)
            .expect("writing to String is infallible");
        output
    }
}

/// Formats a non-zero value in a caller-validated power-of-two radix.
///
/// Radices 2, 4, and 16 map each source byte to a lookup row; radices 8 and
/// 32 extract digits from digit-aligned byte blocks. Neither path divides by
/// the radix, and every emitted digit holds at most five bits.
#[inline]
fn format_power_of_two(value: &InternalMpUint, radix: u32) -> String {
    debug_assert!(
        (2..=32).contains(&radix) && radix.is_power_of_two(),
        "the dispatcher accepts only supported power-of-two radices"
    );
    debug_assert!(
        !value.is_zero(),
        "zero is formatted before the power-of-two dispatcher"
    );

    match radix {
        2 => format_byte_aligned_power_of_two(value, 1, &BINARY_BYTE_DIGITS),
        4 => format_byte_aligned_power_of_two(value, 2, &BASE4_BYTE_DIGITS),
        16 => format_byte_aligned_power_of_two(value, 4, &HEX_BYTE_DIGITS),
        8 => format_block_power_of_two::<3, 24, 3>(value, &BASE8_DIGITS),
        32 => format_block_power_of_two::<5, 40, 5>(value, &BASE32_DIGITS),
        // SAFETY: the dispatcher precondition restricts `radix` to the five
        // supported powers of two, all covered above.
        _ => unsafe { unreachable_unchecked() },
    }
}

/// Formats a non-zero integer in radix 2, 4, or 16 from byte lookup rows.
///
/// Those digit widths divide eight exactly, so each source byte maps to a
/// fixed row of 8, 4, or 2 ASCII digits. Only the most-significant row can be
/// partial; copying its suffix removes leading zero digits.
#[inline]
fn format_byte_aligned_power_of_two<const DIGITS_PER_BYTE: usize>(
    value: &InternalMpUint,
    bits_per_digit: usize,
    table: &[[u8; DIGITS_PER_BYTE]; 256],
) -> String {
    debug_assert!(
        matches!(DIGITS_PER_BYTE, 2 | 4 | 8),
        "only byte-aligned power-of-two digit widths are supported"
    );
    let digit_count = value.significant_bits().div_ceil(bits_per_digit);
    let limbs = value.limbs();
    let mut output: Vec<u8> = Vec::with_capacity(digit_count);
    let output_ptr = output.as_mut_ptr();
    let mut output_index = digit_count;
    let mut limb_index = 0_usize;
    let mut current_limb = 0_usize;
    let mut bytes_left = 0_usize;

    while output_index != 0 {
        if bytes_left == 0 {
            debug_assert!(
                limb_index < limbs.len(),
                "remaining output digits require another normalized source limb"
            );
            // SAFETY: the non-zero normalized value contains enough limbs for
            // `digit_count`; this arm is entered only while digits remain.
            current_limb = unsafe { *limbs.get_unchecked(limb_index) };
            limb_index = limb_index.wrapping_add(1);
            bytes_left = LIMB_BYTES;
        }

        // SAFETY: masking to eight bits makes the conversion infallible.
        let byte = unsafe { u8::try_from(current_limb & 0xff).unwrap_unchecked() };
        // SAFETY: `byte` indexes the 256-row table.
        let row = unsafe { table.get_unchecked(usize::from(byte)) };
        if output_index >= DIGITS_PER_BYTE {
            output_index = output_index.wrapping_sub(DIGITS_PER_BYTE);
            // SAFETY: this branch leaves `DIGITS_PER_BYTE` output slots at
            // `output_index`; the complete static row has that exact length and
            // cannot overlap the vector allocation. Keeping the copy count
            // constant lets the compiler lower each row to fixed-width loads.
            unsafe {
                copy_nonoverlapping(row.as_ptr(), output_ptr.add(output_index), DIGITS_PER_BYTE);
            }
        } else {
            let digits = output_index;
            output_index = 0;
            // SAFETY: this final partial row writes the remaining `digits <
            // DIGITS_PER_BYTE` slots. Its suffix omits only leading zero digits.
            unsafe {
                copy_nonoverlapping(
                    row.as_ptr().add(DIGITS_PER_BYTE.wrapping_sub(digits)),
                    output_ptr,
                    digits,
                );
            }
        }
        current_limb = current_limb.wrapping_shr(8);
        bytes_left = bytes_left.wrapping_sub(1);
    }

    // SAFETY: the loop initialized every output byte from an ASCII lookup row.
    unsafe {
        output.set_len(digit_count);
        String::from_utf8_unchecked(output)
    }
}

/// Formats a non-zero integer in radix 8 or 32 by extracting digits from
/// digit-aligned byte blocks.
///
/// Radix 32 (five bits per digit) groups five bytes into a 40-bit block of
/// exactly eight digits; radix 8 (three bits per digit) groups three bytes
/// into a 24-bit block of exactly eight digits. Because `BITS_PER_DIGIT`
/// divides the block width and the block width is a whole number of bytes,
/// every block boundary is digit-aligned and no carry state is needed: digit
/// `j` of a block is `(block >> (bits_per_digit * j)) & mask`, extracted from
/// the bottom of the block. Only the most significant block can be partial.
///
/// The generic parameters instantiate to `(3, 24, 3)` for radix 8 and
/// `(5, 40, 5)` for radix 32, standing for bits per digit, block bits, and
/// block bytes respectively. The caller provides the packed ASCII digit table
/// for the radix.
#[inline]
fn format_block_power_of_two<
    const BITS_PER_DIGIT: usize,
    const BLOCK_BITS: usize,
    const BLOCK_BYTES: usize,
>(
    value: &InternalMpUint,
    digits: &[u8; 32],
) -> String {
    debug_assert!(
        BLOCK_BITS == BLOCK_BYTES.wrapping_mul(8) && BLOCK_BYTES == BITS_PER_DIGIT,
        "only digit widths that divide a whole-byte block are supported"
    );
    let sig = value.significant_bits();
    let digit_count = sig.div_ceil(BITS_PER_DIGIT);
    let nbytes = sig.div_ceil(8);
    let limbs = value.limbs();
    let mut output: Vec<u8> = Vec::with_capacity(digit_count);
    let output_ptr = output.as_mut_ptr();
    let mut output_index = digit_count;

    // All blocks except the most significant hold exactly `BLOCK_BITS` bits
    // and emit `BLOCK_BITS / BITS_PER_DIGIT = 8` digits (the assert above
    // fixes `BLOCK_BITS = 8 * BITS_PER_DIGIT`); each consumes exactly
    // `BLOCK_BYTES` little-endian bytes.
    let full_block_count = sig.div_ceil(BLOCK_BITS).wrapping_sub(1);
    for block_index in 0..full_block_count {
        emit_block_digits::<BITS_PER_DIGIT>(
            limbs,
            output_ptr,
            &mut output_index,
            BLOCK_BYTES.wrapping_mul(block_index),
            BLOCK_BYTES,
            8,
            digits,
        );
    }
    // The most significant block is guaranteed partial or exactly full:
    // `block_count = ceil(sig / BLOCK_BITS)` bounds
    // `0 < top_block_bits = sig - BLOCK_BITS * full_block_count <=
    // BLOCK_BITS`, and its byte span
    // `top_block_bytes = nbytes - BLOCK_BYTES * full_block_count` is at least
    // 1 and at most `BLOCK_BYTES`, because rounding `sig / (8 * BLOCK_BYTES)`
    // up and scaling by `BLOCK_BYTES` never falls below `ceil(sig / 8)`.
    // Proof that its digits reconstruct exactly the top `top_block_bits`
    // bits with no spurious digits: the assembled block value `v` covers
    // value bits `[8 * top_block_byte_offset, 8 * nbytes)`; bits from `sig`
    // up are zero because the value is normalized, and `v`'s bits above the
    // loaded bytes are zero, so `ceil(top_block_bits / BITS_PER_DIGIT)`
    // digits cover every significant bit with at most
    // `BITS_PER_DIGIT - 1` implicit zero padding bits in the final digit.
    // The total is exact: `8 * full_block_count` digits from the full blocks
    // plus `ceil(top_block_bits / BITS_PER_DIGIT)` equals
    // `ceil((BLOCK_BITS * full_block_count + top_block_bits) / BITS_PER_DIGIT)
    // = ceil(sig / BITS_PER_DIGIT) = digit_count`, since `BITS_PER_DIGIT`
    // divides `BLOCK_BITS`.
    let top_block_byte_offset = BLOCK_BYTES.wrapping_mul(full_block_count);
    let top_block_bits = sig.wrapping_sub(BLOCK_BITS.wrapping_mul(full_block_count));
    let top_block_bytes = nbytes.wrapping_sub(top_block_byte_offset);
    let top_block_digits = top_block_bits.div_ceil(BITS_PER_DIGIT);
    emit_block_digits::<BITS_PER_DIGIT>(
        limbs,
        output_ptr,
        &mut output_index,
        top_block_byte_offset,
        top_block_bytes,
        top_block_digits,
        digits,
    );

    debug_assert_eq!(
        output_index, 0,
        "every allocated digit slot is written exactly once"
    );
    // SAFETY: the loops above emitted exactly `digit_count` digits, each
    // writing a distinct slot below `digit_count == output.capacity()`, and
    // every byte came from the ASCII digit table.
    unsafe {
        output.set_len(digit_count);
        String::from_utf8_unchecked(output)
    }
}

/// Extracts `block_digits` digits from `block_bytes` little-endian source
/// bytes starting at absolute byte offset `block_byte_offset`.
///
/// The block's bytes are assembled into a `u64` block value `v` (low byte
/// first) by OR-ing masked limb reads, then each digit is an independent
/// shift-and-mask of `v`, so the compiler can overlap the shifts. Digit
/// extractions are low-to-high and each digits lands in the descending
/// `output_index` slot, producing big-endian output ordering.
#[inline]
fn emit_block_digits<const BITS_PER_DIGIT: usize>(
    limbs: &[Limb],
    output_ptr: *mut u8,
    output_index: &mut usize,
    block_byte_offset: usize,
    block_bytes: usize,
    block_digits: usize,
    digits: &[u8; 32],
) {
    let mut v: u64 = 0;
    for byte_index in 0..block_bytes {
        // The byte index is at most 4 for a five-byte block, so the shift is
        // at most 32 and `v` never loses an assembled byte.
        let byte = byte_from_limbs(limbs, block_byte_offset.wrapping_add(byte_index));
        v |= u64::from(byte) << byte_index.wrapping_mul(8);
    }

    // `digit_index < block_digits <= 8`, so the shift is at most 40 and the
    // mask keeps every digit in `0..=31`.
    let digit_mask = (1_u64 << BITS_PER_DIGIT).wrapping_sub(1);
    for digit_index in 0..block_digits {
        let digit = (v >> BITS_PER_DIGIT.wrapping_mul(digit_index)) & digit_mask;
        // SAFETY: `digit` is masked to at most five bits, hence in `0..=31`,
        // so both the `u8` fit and the 32-entry table index are in bounds.
        let byte = unsafe { *digits.get_unchecked(usize::try_from(digit).unwrap_unchecked()) };
        debug_assert!(
            *output_index > 0,
            "digit budget guarantees an output slot below output_index"
        );
        *output_index = output_index.wrapping_sub(1);
        // SAFETY: the caller budgets exactly one output slot per emitted
        // digit, so this index stays below `digit_count == output.capacity()`
        // while digits remain, and every iteration writes a distinct slot
        // inside the output vector allocation.
        unsafe {
            output_ptr.add(*output_index).write(byte);
        }
    }
}
/// Reads the byte at `byte_offset` from a little-endian limb array by
/// masking, keeping the access endian-neutral on every pointer width.
#[inline]
fn byte_from_limbs(limbs: &[Limb], byte_offset: usize) -> u8 {
    let limb_index = byte_offset.wrapping_div(LIMB_BYTES);
    let shift_bits = byte_offset.wrapping_rem(LIMB_BYTES).wrapping_mul(8);
    debug_assert!(
        limb_index < limbs.len(),
        "callers pass byte offsets below ceil(significant_bits / 8)"
    );
    // SAFETY: callers only read bytes whose absolute offset is below
    // `ceil(significant_bits / 8) <= limbs.len() * LIMB_BYTES`, because the
    // normalized value stores every significant bit in `limbs`; hence
    // `limb_index = byte_offset / LIMB_BYTES < limbs.len()`.
    let limb = unsafe { *limbs.get_unchecked(limb_index) };
    // SAFETY: `shift_bits` is a multiple of 8 strictly below `LIMB_BITS` —
    // at most 56, 24, or 8 on 64-, 32-, and 16-bit limbs — so the shift
    // stays in range and the conversion to `u32` is infallible.
    let shift_bits_u32 = unsafe { u32::try_from(shift_bits).unwrap_unchecked() };
    // SAFETY: masking to the low eight bits makes the narrowing conversion
    // infallible.
    unsafe { u8::try_from(limb.wrapping_shr(shift_bits_u32) & 0xff).unwrap_unchecked() }
}

/// Upper-bound estimate of the digit count of a value with `significant_bits`
/// bits written in `radix`.
///
/// Every non-power-of-two radix has `ilog2(radix) >= 2`, so dividing the bit
/// count by the bits-per-digit floor and adding one slack digit bounds the true
/// count from above; the slack covers the radix's non-integer bits-per-digit
/// (`log_radix(2) < 1`). The estimate is used only as a capacity hint, so a
/// loose upper bound is sufficient.
pub const fn estimated_digits(significant_bits: usize, radix: u32) -> usize {
    match radix.ilog2() {
        1 => significant_bits.wrapping_add(1),
        2 => significant_bits
            .wrapping_add(1)
            .wrapping_div(2)
            .wrapping_add(1),
        3 => significant_bits
            .wrapping_add(2)
            .wrapping_div(3)
            .wrapping_add(1),
        4 => significant_bits
            .wrapping_add(3)
            .wrapping_div(4)
            .wrapping_add(1),
        5 => significant_bits
            .wrapping_add(4)
            .wrapping_div(5)
            .wrapping_add(1),
        // SAFETY: radix is in 2..=36, so ilog2 is in 1..=5.
        _ => unsafe { unreachable_unchecked() },
    }
}

/// Returns the schoolbook-to-recursive limb threshold for `radix`.
///
/// The grouped thresholds are shared by the small (3..=9), decimal (10), and
/// large (11..=36) radix families. Every caller routes invalid radices and
/// powers of two away before consulting the threshold.
#[inline]
const fn recursive_threshold(radix: u32) -> usize {
    match radix {
        3..=9 => RADIX_SMALL_RECURSIVE_THRESHOLD,
        10 => RADIX_DECIMAL_RECURSIVE_THRESHOLD,
        11..=36 => RADIX_LARGE_RECURSIVE_THRESHOLD,
        // SAFETY: callers exclude invalid radices and powers of two first.
        _ => unsafe { unreachable_unchecked() },
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "This const-evaluated table builder masks every digit to at most four bits and its loops prove both fixed-array indices; no checks or casts remain at runtime"
)]
const fn power_of_two_byte_digits<const DIGITS_PER_BYTE: usize>(
    bits_per_digit: u32,
) -> [[u8; DIGITS_PER_BYTE]; 256] {
    let digit_mask = 1_usize.wrapping_shl(bits_per_digit).wrapping_sub(1);
    let mut table = [[b'0'; DIGITS_PER_BYTE]; 256];
    let mut byte = 0_usize;
    while byte < 256 {
        let mut value = byte;
        let mut index = DIGITS_PER_BYTE;
        while index != 0 {
            index = index.wrapping_sub(1);
            table[byte][index] = byte_from_digit((value & digit_mask) as u8);
            value = value.wrapping_shr(bits_per_digit);
        }
        byte = byte.wrapping_add(1);
    }
    table
}

/// Builds the 32-entry ASCII digit table for a power-of-two radix.
///
/// With `BITS_PER_DIGIT` bits per digit there are `2^BITS_PER_DIGIT` digit
/// values; the table is always sized 32 so the radix-8 and radix-32 paths
/// share one lookup shape.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing,
    reason = "the loop bounds `digit` to 0..=31, inside the fixed 32-entry table; the widening cast is lossless on every supported pointer width"
)]
const fn power_of_two_digit_bytes<const BITS_PER_DIGIT: u32>() -> [u8; 32] {
    let digit_mask = 1_u8.wrapping_shl(BITS_PER_DIGIT).wrapping_sub(1);
    let mut table = [b'0'; 32];
    let mut digit = 0_u8;
    while digit < 32 {
        table[digit as usize] = byte_from_digit(digit & digit_mask);
        digit = digit.wrapping_add(1);
    }
    table
}

/// Maps a digit in `0..=35` to its ASCII representation: `b'0'..=b'9'` for
/// digits below ten and `b'a'..=b'z'` for the letters.
pub const fn byte_from_digit(digit: u8) -> u8 {
    if digit < 10 {
        b'0'.wrapping_add(digit)
    } else {
        b'a'.wrapping_add(digit.wrapping_sub(10))
    }
}

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static FORMAT_CACHE: RefCell<Option<FormatCache>> = const { RefCell::new(None) };
}

// OS-key TLS cannot eagerly materialize a const value. `RefCell::from` keeps
// that necessarily lazy initializer distinct from the const-capable branch and
// produces the identical initial value without suppressing Clippy.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static FORMAT_CACHE: RefCell<Option<FormatCache>> = RefCell::from(None);
}
