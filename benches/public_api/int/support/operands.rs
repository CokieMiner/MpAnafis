//! Exact-width operand generators for both libraries.
//!
//! Every generator is seeded, so `mp_uint(bits, seed)` and
//! `rug_uint(bits, seed)` are the same number in two representations. Widths are
//! exact: the leading nibble always has its high bit set, so a request for 1024
//! bits never yields a 1021-bit value that would silently shorten a limb count.

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
/// Odd operands are required by the Montgomery domain and by Jacobi symbols,
/// and are the realistic shape for a cryptographic modulus.
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
    let mut signed = String::with_capacity(magnitude.len().saturating_add(1));
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
/// Saturating and wrapping policies are no-ops on unlimited values, so the
/// benchmarks that exercise them need an operand whose precision is a real cap.
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

/// Generates a deterministic, exact-width odd FLINT `FlintInt`.
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
pub fn flint_odd_uint(bits: usize, seed: u32) -> FlintInt {
    FlintInt::from_str_radix(&odd_hex(bits, seed), 16)
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
    mp_uint_pairs_with_widths(bits, bits)
}

/// Generates [`SAMPLES`] `MpUint` operand pairs of independent widths.
#[must_use]
pub fn mp_uint_pairs_with_widths(left_bits: usize, right_bits: usize) -> Vec<(MpUint, MpUint)> {
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
    rug_uint_pairs_with_widths(bits, bits)
}

/// Generates [`SAMPLES`] Rug operand pairs of independent widths.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_uint_pairs_with_widths(left_bits: usize, right_bits: usize) -> Vec<(Integer, Integer)> {
    (0..SAMPLES)
        .map(|index| {
            (
                rug_uint(left_bits, 42_u32.wrapping_add(index)),
                rug_uint(right_bits, 1_337_u32.wrapping_add(index)),
            )
        })
        .collect()
}

/// Generates [`SAMPLES`] equal-width Flint operand pairs.
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
#[must_use]
pub fn flint_uint_pairs(bits: usize) -> Vec<(FlintInt, FlintInt)> {
    (0..SAMPLES)
        .map(|index| {
            (
                flint_uint(bits, 42_u32.wrapping_add(index)),
                flint_uint(bits, 1_337_u32.wrapping_add(index)),
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
