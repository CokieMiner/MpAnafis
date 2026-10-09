//! Division fixtures with explicit quotient geometry and remainder shape.
//! Arguments specify divisor bits. Every batch contains ten identical-across-
//! engine pairs, constructed outside timing from `U = Q*D + R`.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::{SAMPLES, mp_uint};

/// Quotient width relative to the divisor, or a prescribed small quotient.
#[derive(Clone, Copy, Debug)]
pub enum DivisionShape {
    Balanced,
    ThreeByTwo,
    Long,
    SmallQuotient,
    PowerOfTwo,
}

/// Exact and near-multiple cases exercise opposite quotient correction edges.
#[derive(Clone, Copy, Debug)]
pub enum DivisionResidue {
    Random,
    Zero,
    Maximal,
}

/// Odd-part geometries for destructive divisibility cancellation.
#[derive(Clone, Copy, Debug)]
pub enum DivisibilityShape {
    Odd,
    Shifted,
    ShiftedScalar,
}

/// Constructs divisors of `bits` bits and a 128-bit quotient. Nonmultiples
/// retain the divisor's complete power-of-two factor, so the valuation check
/// cannot decide the predicate. Shifts contain both whole and partial limbs.
pub fn mp_divisibility_pairs(
    bits: usize,
    shape: DivisibilityShape,
    exact: bool,
) -> Vec<(MpUint, MpUint)> {
    assert!(
        bits >= 256 && bits.is_multiple_of(4),
        "predicate fixtures require nibble-aligned widths of at least 256 bits"
    );
    let shift = match shape {
        DivisibilityShape::Odd => 0,
        DivisibilityShape::Shifted => (bits >> 1).checked_add(4).expect("shift fits"),
        DivisibilityShape::ShiftedScalar => bits.checked_sub(5).expect("five-bit odd part"),
    };
    (0..SAMPLES)
        .map(|index| {
            let seed = index.wrapping_add(42);
            let odd = if matches!(shape, DivisibilityShape::ShiftedScalar) {
                MpUint::from(29_u32)
            } else {
                mp_uint(bits.checked_sub(shift).expect("positive odd width"), seed) | MpUint::one()
            };
            let divisor = odd << shift;
            let quotient = mp_uint(128, seed.wrapping_add(1_337));
            let mut dividend = &quotient * &divisor;
            if !exact {
                dividend += MpUint::one() << shift;
            }
            assert_eq!(
                dividend.rem_trunc(&divisor).is_zero(),
                exact,
                "fixture residue must match the selected divisibility class"
            );
            (dividend, divisor)
        })
        .collect()
}

/// Converts the same predicate fixtures for GMP outside timing.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn rug_divisibility_pairs(
    bits: usize,
    shape: DivisibilityShape,
    exact: bool,
) -> Vec<(Integer, Integer)> {
    mp_divisibility_pairs(bits, shape, exact)
        .into_iter()
        .map(|(left, right)| {
            (
                Integer::from_str_radix(&left.to_string_radix(16), 16).expect("dividend parses"),
                Integer::from_str_radix(&right.to_string_radix(16), 16).expect("divisor parses"),
            )
        })
        .collect()
}

/// Constructs ten positive pairs with known quotient and remainder.
pub fn mp_division_pairs(
    bits: usize,
    shape: DivisionShape,
    residue: DivisionResidue,
) -> Vec<(MpUint, MpUint)> {
    assert!(
        bits >= 8 && bits.is_multiple_of(4),
        "division widths are nibble aligned"
    );
    let quotient_bits = match shape {
        DivisionShape::ThreeByTwo => (bits >> 3) << 2,
        DivisionShape::Long => bits.checked_mul(3).expect("long quotient width fits"),
        DivisionShape::Balanced | DivisionShape::SmallQuotient | DivisionShape::PowerOfTwo => bits,
    };
    (0..SAMPLES)
        .map(|index| {
            let seed = index.wrapping_add(if index < (SAMPLES >> 1) { 42 } else { 2_021 });
            let divisor = if matches!(shape, DivisionShape::PowerOfTwo) {
                MpUint::one() << bits.checked_sub(1).expect("positive width")
            } else {
                mp_uint(bits, seed.wrapping_add(1_337))
            };
            let quotient = if matches!(shape, DivisionShape::SmallQuotient) {
                MpUint::from(match index {
                    0 => 1_u32,
                    1 => 2,
                    2 => 3,
                    3 => 4,
                    4 => 7,
                    5 => 8,
                    6 => 15,
                    7 => 16,
                    8 => 32,
                    _ => 65,
                })
            } else {
                mp_uint(quotient_bits, seed)
            };
            let remainder = match residue {
                DivisionResidue::Random => {
                    let candidate = mp_uint(bits, seed.wrapping_add(9_999));
                    if candidate < divisor {
                        candidate
                    } else {
                        candidate - &divisor
                    }
                }
                DivisionResidue::Zero => MpUint::zero(),
                DivisionResidue::Maximal => &divisor - MpUint::one(),
            };
            assert!(
                remainder < divisor,
                "constructed remainder is below the divisor"
            );
            let dividend = &quotient * &divisor + remainder;
            (dividend, divisor)
        })
        .collect()
}

/// Parses the same constructed values for GMP; conversion is outside timing.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn rug_division_pairs(
    bits: usize,
    shape: DivisionShape,
    residue: DivisionResidue,
) -> Vec<(Integer, Integer)> {
    mp_division_pairs(bits, shape, residue)
        .into_iter()
        .map(|(left, right)| {
            (
                Integer::from_str_radix(&left.to_string_radix(16), 16).expect("dividend parses"),
                Integer::from_str_radix(&right.to_string_radix(16), 16).expect("divisor parses"),
            )
        })
        .collect()
}
