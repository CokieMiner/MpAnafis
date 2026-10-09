//! Native radix block extraction and ASCII digit writes.

#![expect(
    unsafe_code,
    reason = "Native radix powers bound fractional products and digit pairs; initialized spans remain within reserved output or stack storage"
)]

use core::{mem::MaybeUninit, slice::from_raw_parts};

use alloc::vec::Vec;

use super::{
    Convert, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb, RADIX_CHUNK_RECIPROCALS,
    RadixParameters,
};

/// ASCII encodings of the integers in `0..100`.
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

impl Convert {
    /// Appends decimal blocks in least-significant-digit-first order.
    ///
    /// The caller supplies a nonzero value. Every division returns a native
    /// block; only the final, nonzero block omits its leading zero padding.
    pub fn write_decimal_chunks(value: &mut InternalMpUint, output: &mut Vec<u8>) {
        debug_assert!(!value.is_zero(), "only nonzero decimal values are chunked");
        loop {
            let chunk = Self::div_rem_decimal_chunk(value);
            if value.is_zero() {
                push_decimal_chunk::<true>(chunk, false, output);
                break;
            }
            push_decimal_chunk::<true>(chunk, true, output);
        }
    }

    /// Appends one native radix block in the selected digit order.
    ///
    /// `REVERSED` selects least-significant-digit-first output. The caller
    /// establishes `chunk < parameters.max_power` for its validated radix.
    /// Full blocks contain `max_digits` digits; leading blocks omit high zeros.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "DoubleLimb embeds Limb; fractional products retain their low limb modulo 2^LIMB_BITS and their high digit is below radix <= 36"
    )]
    pub fn write_radix_chunk<const REVERSED: bool>(
        chunk: Limb,
        radix: Limb,
        parameters: RadixParameters,
        full_width: bool,
        output: &mut Vec<u8>,
    ) {
        debug_assert!(
            (3..=36).contains(&radix),
            "radix blocks use a validated radix"
        );
        debug_assert!(
            chunk < parameters.max_power,
            "a radix block fits its divisor"
        );
        if radix == 10 {
            push_decimal_chunk::<REVERSED>(chunk, full_width, output);
            return;
        }

        // For B = radix^m < 2^W, f = floor(chunk*2^W/B)+1 satisfies
        // chunk < f*B/2^W < chunk+1. Successive high limbs of f*radix
        // therefore recover exactly the m base-radix digits of chunk.
        let shift = parameters.max_power.leading_zeros();
        // SAFETY: 0 <= chunk < B and B > 0 give shift < W and
        // chunk << shift < B << shift < 2^W.
        let (high, divisor) = unsafe {
            (
                chunk.unchecked_shl(shift),
                parameters.max_power.unchecked_shl(shift),
            )
        };
        // SAFETY: the validated radix selects its normalized reciprocal.
        let reciprocal = unsafe { *RADIX_CHUNK_RECIPROCALS.get_unchecked(radix) };
        let (initial_fraction, _) = Division::divrem_2by1_reciprocal(high, 0, divisor, reciprocal);
        // SAFETY: B < 2^W gives floor((B-1)*2^W/B) < 2^W-1.
        let mut fraction = unsafe { initial_fraction.unchecked_add(1) };
        let mut buffer = [MaybeUninit::<u8>::uninit(); LIMB_BITS];
        let digits = parameters.max_digits;
        debug_assert!(
            (1..LIMB_BITS).contains(&digits),
            "a native block has fewer than W digits"
        );
        for index in 0..digits {
            // SAFETY: fraction < 2^W and radix <= 36 bound the product
            // below 2^(W+6), within DoubleLimb for W = 16, 32, 64.
            let product = unsafe { (fraction as DoubleLimb).unchecked_mul(radix as DoubleLimb) };
            let digit = (product >> LIMB_BITS) as u8;
            fraction = product as Limb;
            // SAFETY: index < digits < LIMB_BITS bounds both orders.
            // Each position in 0..digits receives one initialized ASCII byte.
            unsafe {
                let position = if REVERSED {
                    digits.unchecked_sub(1).unchecked_sub(index)
                } else {
                    index
                };
                buffer
                    .as_mut_ptr()
                    .add(position)
                    .write(MaybeUninit::new(Self::byte_from_digit(digit)));
            }
        }
        let mut start = 0_usize;
        let mut width = digits;
        if !full_width {
            while width > 1 {
                // SAFETY: 1 < width <= digits; both positions lie in the
                // fully initialized span. Trimming preserves one zero digit.
                let high_digit = unsafe {
                    let position = if REVERSED {
                        width.unchecked_sub(1)
                    } else {
                        start
                    };
                    buffer.get_unchecked(position).assume_init()
                };
                if high_digit != b'0' {
                    break;
                }
                // SAFETY: width > 1 bounds the decrement. In normal order,
                // start+width == digits, so the next position remains in range.
                unsafe {
                    width = width.unchecked_sub(1);
                    if !REVERSED {
                        start = start.unchecked_add(1);
                    }
                }
            }
        }
        // SAFETY: the entire 0..digits span is initialized; start+width <= digits.
        // MaybeUninit<u8> has the same size and alignment as u8.
        output.extend_from_slice(unsafe {
            from_raw_parts(buffer.as_ptr().cast::<u8>().add(start), width)
        });
    }
}

/// Appends a decimal block, retaining its fixed width or significant digits.
#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Remainders modulo 100 index the digit-pair table; each remaining leading digit is below ten and fits u8"
)]
fn push_decimal_chunk<const REVERSED: bool>(
    mut chunk: Limb,
    full_width: bool,
    output: &mut Vec<u8>,
) {
    let digits = Convert::DECIMAL_CHUNK_DIGITS;
    let pairs = digits.wrapping_div(2);
    if full_width {
        let start = output.len();
        output.reserve(digits);
        // SAFETY: reserve establishes a writable digit block disjoint from
        // the existing initialized prefix and the static pair table.
        let output_ptr = unsafe { output.as_mut_ptr().add(start) };
        for index in 0..pairs {
            let quotient = chunk.wrapping_div(100);
            let remainder = chunk.wrapping_rem(100);
            // SAFETY: remainder < 100 selects an initialized digit pair.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(remainder) };
            // SAFETY: index < floor(digits/2) gives 2*index+2 <= digits.
            // Both orders write disjoint pairs within the reserved block;
            // [u8; 2] has alignment one on every supported target.
            unsafe {
                let offset = if REVERSED {
                    index.unchecked_mul(2)
                } else {
                    digits
                        .unchecked_sub(2)
                        .unchecked_sub(index.unchecked_mul(2))
                };
                let encoded = if REVERSED { [pair[1], pair[0]] } else { pair };
                output_ptr.add(offset).cast::<[u8; 2]>().write(encoded);
            }
            chunk = quotient;
        }
        if !digits.is_multiple_of(2) {
            // SAFETY: an odd block leaves one slot at the selected end.
            // After all pair divisions, chunk is a single leading digit.
            unsafe {
                let offset = if REVERSED { pairs.unchecked_mul(2) } else { 0 };
                output_ptr
                    .add(offset)
                    .write(b'0'.unchecked_add(chunk as u8));
            }
        }
        // SAFETY: reserve proves start+digits <= capacity <= isize::MAX;
        // the pair writes and optional leading digit initialized that span.
        unsafe {
            output.set_len(start.unchecked_add(digits));
        }
    } else if REVERSED {
        while chunk >= 100 {
            let quotient = chunk.wrapping_div(100);
            let remainder = chunk.wrapping_rem(100);
            // SAFETY: remainder < 100 selects an initialized digit pair.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(remainder) };
            output.push(pair[1]);
            output.push(pair[0]);
            chunk = quotient;
        }
        if chunk >= 10 {
            // SAFETY: the loop established chunk < 100.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(chunk) };
            output.push(pair[1]);
            output.push(pair[0]);
        } else {
            // SAFETY: chunk < 10, so the ASCII sum is at most '9' = 57.
            output.push(unsafe { b'0'.unchecked_add(chunk as u8) });
        }
    } else {
        let mut buffer = [MaybeUninit::<u8>::uninit(); Convert::DECIMAL_CHUNK_DIGITS];
        let mut start = digits;
        while chunk >= 100 {
            let quotient = chunk.wrapping_div(100);
            let remainder = chunk.wrapping_rem(100);
            // SAFETY: remainder < 100 selects an initialized digit pair.
            let pair = unsafe { *DECIMAL_DIGIT_PAIRS.get_unchecked(remainder) };
            // SAFETY: chunk initially has at most digits decimal digits.
            // Each removed pair consumes two remaining slots, so start >= 2.
            // The pair initializes a distinct suffix span with alignment one.
            unsafe {
                start = start.unchecked_sub(2);
                buffer
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(start)
                    .cast::<[u8; 2]>()
                    .write(pair);
            }
            chunk = quotient;
        }
        if chunk >= 10 {
            // SAFETY: chunk < 100 and its two remaining digits fit below
            // start. The pair initializes the next disjoint suffix span.
            unsafe {
                start = start.unchecked_sub(2);
                let pair = *DECIMAL_DIGIT_PAIRS.get_unchecked(chunk);
                buffer
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(start)
                    .cast::<[u8; 2]>()
                    .write(pair);
            }
        } else {
            // SAFETY: chunk < 10 leaves one remaining slot, including zero.
            // Its ASCII digit fits u8 and initializes the suffix's first byte.
            unsafe {
                start = start.unchecked_sub(1);
                buffer
                    .as_mut_ptr()
                    .add(start)
                    .write(MaybeUninit::new(b'0'.unchecked_add(chunk as u8)));
            }
        }
        // SAFETY: 0 <= start < digits and every byte in start..digits was
        // initialized. MaybeUninit<u8> has the size and alignment of u8.
        output.extend_from_slice(unsafe {
            from_raw_parts(
                buffer.as_ptr().cast::<u8>().add(start),
                digits.unchecked_sub(start),
            )
        });
    }
}
