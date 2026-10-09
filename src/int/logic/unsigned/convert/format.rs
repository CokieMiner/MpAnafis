//! Radix string formatting entry points and power-of-two digit paths.

#![expect(
    unsafe_code,
    reason = "Radix is asserted in 2..=36; digit and byte tables are indexed by proved 0..=31 / 0..=255 / byte bounds; row copies stay within the recorded output range; byte extraction stays below ceil(significant_bits / 8); limb loops use index < len; emitted bytes are ASCII."
)]

#[cfg(feature = "std")]
use core::cell::RefCell;
use core::{
    fmt::{Display, Error, Formatter, Result as FmtResult, Write},
    ptr::copy_nonoverlapping,
};
#[cfg(feature = "std")]
use std::thread_local;

use alloc::{string::String, vec::Vec};

use super::{
    BASE4_BYTE_DIGITS, BASE8_DIGITS, BASE32_DIGITS, BINARY_BYTE_DIGITS, Convert, FormatCache,
    HEX_BYTE_DIGITS, InternalMpUint, LIMB_BITS, LIMB_BYTES, Limb,
};

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static FORMAT_CACHE: RefCell<Option<FormatCache>> = const { RefCell::new(None) };
}

// OS-key TLS initializes the cache cell lazily on its first access.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static FORMAT_CACHE: RefCell<Option<FormatCache>> = RefCell::from(None);
}

impl Display for InternalMpUint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        if formatter.width().is_some() || formatter.precision().is_some() || formatter.sign_plus() {
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
    /// Uses lowercase letters for digits above 9. Invalid radices, zero,
    /// powers of two, and schoolbook-sized values are handled before the
    /// thread-local cache is touched; the recursive divide-and-conquer path
    /// falls back to a fresh cache when the cached borrow is already held
    /// (reentrant formatting) or when `std` is disabled.
    fn format_radix_writer(&self, radix: u32, w: &mut dyn Write) -> FmtResult {
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

        if self.limbs().len() < Convert::recursive_threshold(radix) {
            let string = self.format_schoolbook_string(radix);
            return w.write_str(&string);
        }

        // Validate the bit-count boundary before entering the infallible
        // recursive kernel; every active radix block is below this count.
        let significant_bits = self.significant_bits();
        debug_assert!(
            significant_bits > 0,
            "zero was handled before recursive dispatch"
        );

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

        if self.limbs().len() < Convert::recursive_threshold(radix) {
            return self.format_schoolbook_string(radix);
        }

        // The output stays a concrete `String` so the recursive writer keeps
        // static dispatch; only the slow path touches the thread-local cache.
        let mut output =
            String::with_capacity(Convert::estimated_digits(self.significant_bits(), radix));

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
        // SAFETY: significant_bits was validated before allocating output.
        // Each active radix block satisfies radix^block_digits <= self, hence
        // block_digits < significant_bits and its width fits usize. The
        // recursive kernel propagates only Write errors; String is infallible.
        unsafe {
            fmt_result.unwrap_unchecked();
        }
        output
    }

    /// Forced schoolbook digit-extraction path for the crossover tuner.
    ///
    /// Always uses repeated single-limb block division regardless of limb count,
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
        let mut output =
            String::with_capacity(Convert::estimated_digits(self.significant_bits(), radix));
        // SAFETY: the capacity calculation validated significant_bits. Every
        // active block width is below that count and fits usize. The kernel
        // propagates only Write errors; the concrete String is infallible.
        unsafe {
            self.format_recursive_writer_with_cache(radix, &mut output, cache)
                .unwrap_unchecked();
        }
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
        _ => {
            // The validated power-of-two radices are 2, 4, 8, 16, and 32;
            // the preceding arms leave only the five-bit radix.
            debug_assert_eq!(radix, 32, "the remaining power-of-two radix is 32");
            format_block_power_of_two::<5, 40, 5>(value, &BASE32_DIGITS)
        }
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
    // SAFETY: the dispatcher handles zero before this nonzero formatter.
    let (&top, lower) = unsafe { limbs.split_last().unwrap_unchecked() };
    debug_assert_ne!(top, 0, "a normalized magnitude has a nonzero highest limb");
    // SAFETY: top != 0 gives leading_zeros < W <= 64. The conversion fits
    // every usize, and the significant top width lies in 1..=W.
    let top_bits =
        unsafe { LIMB_BITS.unchecked_sub(usize::try_from(top.leading_zeros()).unwrap_unchecked()) };
    // SAFETY: top_bits >= 1; the final source byte is handled separately.
    let full_top_bytes = unsafe { top_bits.unchecked_sub(1) }.wrapping_div(8);
    let mut output: Vec<u8> = Vec::with_capacity(digit_count);
    let output_ptr = output.as_mut_ptr();
    let mut output_index = digit_count;
    for &limb in lower {
        for byte_index in 0..LIMB_BYTES {
            // SAFETY: byte_index < LIMB_BYTES bounds 8*byte_index below W.
            // Masking selects an initialized table row. Every lower limb
            // emits complete rows within the precomputed digit budget; each
            // fixed-width copy initializes a distinct output span disjoint
            // from the static lookup table.
            unsafe {
                let shift = u32::try_from(byte_index.unchecked_mul(8)).unwrap_unchecked();
                let row = table.get_unchecked(limb.unchecked_shr(shift) & 0xff);
                output_index = output_index.unchecked_sub(DIGITS_PER_BYTE);
                copy_nonoverlapping(row.as_ptr(), output_ptr.add(output_index), DIGITS_PER_BYTE);
            }
        }
    }
    for byte_index in 0..full_top_bytes {
        // SAFETY: full_top_bytes < LIMB_BYTES bounds the source shift. Each
        // complete row leaves at least the final significant source byte,
        // so its digit span fits below output_index and cannot overlap the table.
        unsafe {
            let shift = u32::try_from(byte_index.unchecked_mul(8)).unwrap_unchecked();
            let row = table.get_unchecked(top.unchecked_shr(shift) & 0xff);
            output_index = output_index.unchecked_sub(DIGITS_PER_BYTE);
            copy_nonoverlapping(row.as_ptr(), output_ptr.add(output_index), DIGITS_PER_BYTE);
        }
    }
    debug_assert!(
        (1..=DIGITS_PER_BYTE).contains(&output_index),
        "only the most-significant lookup row can be partial"
    );
    // SAFETY: full_top_bytes < LIMB_BYTES bounds the final source shift.
    // The remaining 1..=DIGITS_PER_BYTE slots receive the row's significant
    // suffix. Together with the complete rows, this initializes all allocated
    // digits exactly once from ASCII bytes, without overlapping the table.
    unsafe {
        let shift = u32::try_from(full_top_bytes.unchecked_mul(8)).unwrap_unchecked();
        let row = table.get_unchecked(top.unchecked_shr(shift) & 0xff);
        copy_nonoverlapping(
            row.as_ptr()
                .add(DIGITS_PER_BYTE.unchecked_sub(output_index)),
            output_ptr,
            output_index,
        );
        output.set_len(digit_count);
        String::from_utf8_unchecked(output)
    }
}

/// Formats a nonzero integer using digit-aligned blocks of eight digits.
///
/// Radix 8 uses three-byte, 24-bit blocks; radix 32 uses five-byte, 40-bit
/// blocks. Only the most significant block can be partial. Digit `j` is
/// `(block >> (BITS_PER_DIGIT * j)) & mask`, with no inter-block carry.
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
        matches!(
            (BITS_PER_DIGIT, BLOCK_BITS, BLOCK_BYTES),
            (3, 24, 3) | (5, 40, 5)
        ),
        "only digit widths that divide a whole-byte block are supported"
    );
    let sig = value.significant_bits();
    let digit_count = sig.div_ceil(BITS_PER_DIGIT);
    let nbytes = sig.div_ceil(8);
    let limbs = value.limbs();
    let mut output: Vec<u8> = Vec::with_capacity(digit_count);
    let output_ptr = output.as_mut_ptr();
    let mut output_index = digit_count;

    // SAFETY: the caller handled zero; sig > 0 gives at least one block.
    let full_block_count = unsafe { sig.div_ceil(BLOCK_BITS).unchecked_sub(1) };
    for block_index in 0..full_block_count {
        // SAFETY: block_index < ceil(sig / BLOCK_BITS) bounds the product
        // below ceil(sig / 8), which fits usize on every pointer width.
        let block_byte_offset = unsafe { BLOCK_BYTES.unchecked_mul(block_index) };
        emit_block_digits::<BITS_PER_DIGIT>(
            limbs,
            output_ptr,
            &mut output_index,
            block_byte_offset,
            BLOCK_BYTES,
            8,
            digits,
        );
    }
    // SAFETY: full_block_count = floor((sig - 1) / BLOCK_BITS), so its bit
    // product is below sig. Its byte product is below nbytes = ceil(sig / 8).
    // Both products and differences therefore fit usize on every pointer width.
    let (top_block_byte_offset, top_block_bits, top_block_bytes) = unsafe {
        let byte_offset = BLOCK_BYTES.unchecked_mul(full_block_count);
        (
            byte_offset,
            sig.unchecked_sub(BLOCK_BITS.unchecked_mul(full_block_count)),
            nbytes.unchecked_sub(byte_offset),
        )
    };
    // The top block contains 1..=BLOCK_BITS bits and 1..=BLOCK_BYTES bytes.
    // Eight digits per full block plus ceil(top_block_bits / BITS_PER_DIGIT)
    // equals digit_count because BLOCK_BITS = 8 * BITS_PER_DIGIT.
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
    // writing a distinct slot below `digit_count <= output.capacity()`, and
    // every byte came from the ASCII digit table.
    unsafe {
        output.set_len(digit_count);
        String::from_utf8_unchecked(output)
    }
}

/// Assembles at most five little-endian bytes and emits at most eight digits.
///
/// Low-to-high extraction writes descending output slots, yielding big-endian
/// ASCII digits. The caller budgets the exact remaining slots for every block.
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
        // SAFETY: the caller's byte span ends below ceil(sig / 8), so this
        // offset fits usize. byte_index <= 4 bounds its bit offset by 32.
        let (byte_offset, shift) = unsafe {
            (
                block_byte_offset.unchecked_add(byte_index),
                byte_index.unchecked_mul(8),
            )
        };
        let byte = byte_from_limbs(limbs, byte_offset);
        v |= u64::from(byte) << shift;
    }

    // SAFETY: BITS_PER_DIGIT is 3 or 5; the shifted one is positive.
    let digit_mask = unsafe { (1_u64 << BITS_PER_DIGIT).unchecked_sub(1) };
    for digit_index in 0..block_digits {
        // SAFETY: digit_index < block_digits <= 8 and BITS_PER_DIGIT <= 5
        // bound the product by 35, below the 64-bit block width.
        let shift = unsafe { BITS_PER_DIGIT.unchecked_mul(digit_index) };
        let digit = (v >> shift) & digit_mask;
        // SAFETY: `digit` is masked to at most five bits, hence in `0..=31`,
        // so both the `u8` fit and the 32-entry table index are in bounds.
        let byte = unsafe { *digits.get_unchecked(usize::try_from(digit).unwrap_unchecked()) };
        debug_assert!(
            *output_index > 0,
            "digit budget guarantees an output slot below output_index"
        );
        // SAFETY: the caller's exact digit budget gives output_index > 0;
        // decrement selects a distinct slot below the output vector capacity.
        unsafe {
            *output_index = output_index.unchecked_sub(1);
            output_ptr.add(*output_index).write(byte);
        }
    }
}
/// Reads the byte at `byte_offset` from a little-endian limb array by
/// masking, keeping the access endian-neutral on every pointer width.
#[inline]
fn byte_from_limbs(limbs: &[Limb], byte_offset: usize) -> u8 {
    let limb_index = byte_offset.wrapping_div(LIMB_BYTES);
    // SAFETY: the byte remainder is below LIMB_BYTES <= 8, so its bit offset
    // is at most 56, within usize on every supported pointer width.
    let shift_bits = unsafe { byte_offset.rem_euclid(LIMB_BYTES).unchecked_mul(8) };
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
