//! Exact-width operand generators for both libraries.
//!
//! Equal seeds produce equal magnitudes across engines. The leading
//! hexadecimal digit has its high bit set, establishing the requested width.

use mp_anafis::{BoundedPrecision, MpInt, MpUint};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use super::FlintInt;
use super::SAMPLES;

/// Generates a deterministic, exact-width hexadecimal operand.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[must_use]
pub fn random_hex(bits: usize, seed: u32) -> String {
    assert!(bits > 0, "benchmark bit width must be nonzero");
    assert!(
        bits.is_multiple_of(4),
        "benchmark widths must be nibble aligned"
    );

    let digit_count = bits.checked_div(4).expect("four is nonzero");
    let mut state = seed;
    let digits = (0..digit_count)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random_nibble =
                u8::try_from(state.wrapping_shr(28)).expect("the high nibble of a u32 fits in u8");
            let exact_width_nibble = if index == 0 {
                random_nibble | 8
            } else {
                random_nibble
            };
            hex_digit(exact_width_nibble)
        })
        .collect();
    String::from_utf8(digits).expect("the hexadecimal generator emits ASCII")
}

/// Generates a deterministic, exact-width odd hexadecimal operand.
///
/// The final digit is `f`, giving an odd modulus for Montgomery and Jacobi.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[must_use]
pub fn odd_hex(bits: usize, seed: u32) -> String {
    let mut value = random_hex(bits, seed);
    let final_digit = value
        .len()
        .checked_sub(1)
        .expect("nonzero width produces at least one digit");
    value.replace_range(final_digit.., "f");
    value
}

/// Generates the negation of [`random_hex`] as a signed literal.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[must_use]
pub fn negative_hex(bits: usize, seed: u32) -> String {
    let magnitude = random_hex(bits, seed);
    let capacity = magnitude
        .len()
        .checked_add(1)
        .expect("signed literal capacity fits usize");
    let mut signed = String::with_capacity(capacity);
    signed.push('-');
    signed.push_str(&magnitude);
    signed
}

/// Generates a deterministic, exact-width `MpUint`.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[must_use]
pub fn mp_uint(bits: usize, seed: u32) -> MpUint {
    MpUint::from_str_radix(&random_hex(bits, seed), 16)
        .expect("generated hexadecimal must parse as MpUint")
}

/// Generates a deterministic, exact-width `MpInt` of the requested sign.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[must_use]
pub fn mp_int(bits: usize, seed: u32, negative: bool) -> MpInt {
    let text = if negative {
        negative_hex(bits, seed)
    } else {
        random_hex(bits, seed)
    };
    MpInt::from_str_radix(&text, 16).expect("generated hexadecimal must parse as MpInt")
}

/// Generates a deterministic, exact-width `MpUint` carrying bounded precision.
///
/// The precision equals the generated magnitude's bit width.
///
/// # Panics
///
/// Panics when `bits` is not a valid bounded precision or the operand does not
/// fit it.
#[must_use]
pub fn bounded_mp_uint(bits: usize, seed: u32) -> MpUint {
    let width = BoundedPrecision::new(bits).expect("benchmark widths are valid bounded precision");
    MpUint::with_precision_checked(mp_uint(bits, seed), width)
        .expect("the exact-width benchmark operand fits its precision")
}

/// Reduces a deterministic signed operand to the requested bounded precision.
///
/// # Panics
///
/// Panics when `bits` is not a valid bounded precision.
#[must_use]
pub fn bounded_mp_int(bits: usize, seed: u32, negative: bool) -> MpInt {
    let width = BoundedPrecision::new(bits).expect("benchmark widths are valid bounded precision");
    MpInt::with_precision_wrapping(mp_int(bits, seed, negative), width)
}

/// Generates a deterministic, exact-width Rug `Integer`.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_uint(bits: usize, seed: u32) -> Integer {
    Integer::from_str_radix(&random_hex(bits, seed), 16)
        .expect("generated hexadecimal must parse as Rug Integer")
}

/// Generates a deterministic, exact-width FLINT `FlintInt`.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
#[must_use]
pub fn flint_uint(bits: usize, seed: u32) -> FlintInt {
    FlintInt::from_str_radix(&random_hex(bits, seed), 16)
}

/// Generates a deterministic, exact-width Rug `Integer` of the requested sign.
///
/// # Panics
///
/// Panics when `bits` is zero or is not divisible by four.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_int(bits: usize, seed: u32, negative: bool) -> Integer {
    let text = if negative {
        negative_hex(bits, seed)
    } else {
        random_hex(bits, seed)
    };
    Integer::from_str_radix(&text, 16).expect("generated hexadecimal must parse as Rug Integer")
}

/// Generates [`SAMPLES`] equal-width `MpUint` operand pairs.
#[must_use]
pub fn mp_uint_pairs(bits: usize) -> Vec<(MpUint, MpUint)> {
    (0..SAMPLES)
        .map(|index| {
            (
                mp_uint(bits, 42_u32.wrapping_add(index)),
                mp_uint(bits, 1_337_u32.wrapping_add(index)),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] unbalanced `MpUint` operand pairs with explicit bit widths.
#[must_use]
pub fn mp_uint_lopsided_pairs(left_bits: usize, right_bits: usize) -> Vec<(MpUint, MpUint)> {
    (0..SAMPLES)
        .map(|index| {
            (
                mp_uint(left_bits, 42_u32.wrapping_add(index)),
                mp_uint(right_bits, 1_337_u32.wrapping_add(index)),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] equal-width `MpInt` operand pairs of fixed signs.
#[must_use]
pub fn mp_int_pairs(bits: usize, left_negative: bool, right_negative: bool) -> Vec<(MpInt, MpInt)> {
    (0..SAMPLES)
        .map(|index| {
            (
                mp_int(bits, 42_u32.wrapping_add(index), left_negative),
                mp_int(bits, 1_337_u32.wrapping_add(index), right_negative),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] equal-width Rug operand pairs.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_uint_pairs(bits: usize) -> Vec<(Integer, Integer)> {
    (0..SAMPLES)
        .map(|index| {
            (
                rug_uint(bits, 42_u32.wrapping_add(index)),
                rug_uint(bits, 1_337_u32.wrapping_add(index)),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] unbalanced Rug operand pairs with explicit bit widths.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_uint_lopsided_pairs(left_bits: usize, right_bits: usize) -> Vec<(Integer, Integer)> {
    (0..SAMPLES)
        .map(|index| {
            (
                rug_uint(left_bits, 42_u32.wrapping_add(index)),
                rug_uint(right_bits, 1_337_u32.wrapping_add(index)),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] equal-width Rug operand pairs of fixed signs.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_int_pairs(
    bits: usize,
    left_negative: bool,
    right_negative: bool,
) -> Vec<(Integer, Integer)> {
    (0..SAMPLES)
        .map(|index| {
            (
                rug_int(bits, 42_u32.wrapping_add(index), left_negative),
                rug_int(bits, 1_337_u32.wrapping_add(index), right_negative),
            )
        })
        .collect()
}

const fn hex_digit(value: u8) -> u8 {
    if value < 10 {
        b'0'.wrapping_add(value)
    } else {
        b'a'.wrapping_add(value.wrapping_sub(10))
    }
}
