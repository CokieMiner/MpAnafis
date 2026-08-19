//! Untimed benchmark input and result verification.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "The untimed checks intentionally exercise the public arithmetic operators to verify division identities."
)]

use mp_anafis::{MpInt, MpUint};

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use super::FlintInt;

/// Verifies every public unsigned division result used by the benchmark batch.
///
/// This runs while constructing a benchmark cell, outside its timed closure.
/// The checks cover the quotient/remainder identity, the quotient-only and
/// remainder-only paths, and the unsigned rounding aliases.
pub fn verify_mp_uint_division_pairs(inputs: &[(MpUint, MpUint)]) {
    for (left, right) in inputs {
        let (quotient, remainder) = left
            .div_rem(right)
            .expect("division benchmark divisor must be non-zero");
        let reconstructed = (&quotient * right) + &remainder;
        assert_eq!(&reconstructed, left, "unsigned division identity");
        assert!(
            right > &remainder,
            "unsigned division remainder must be smaller than divisor"
        );
        assert_eq!(left.div_trunc(right), quotient, "unsigned quotient path");
        assert_eq!(left.rem_trunc(right), remainder, "unsigned remainder path");
        assert_eq!(
            left.div_euclid(right),
            quotient,
            "unsigned Euclidean quotient"
        );
        assert_eq!(
            left.rem_euclid(right),
            remainder,
            "unsigned Euclidean remainder"
        );
        assert_eq!(left.div_floor(right), quotient, "unsigned floor quotient");
        assert_eq!(left.mod_floor(right), remainder, "unsigned floor remainder");
        assert_eq!(
            left.checked_div(right),
            Some(quotient.clone()),
            "unsigned checked quotient"
        );
        assert_eq!(
            left.checked_rem(right),
            Some(remainder.clone()),
            "unsigned checked remainder"
        );
        assert_eq!(
            left.checked_div_ceil(right),
            Some(left.div_ceil(right)),
            "unsigned checked ceiling quotient"
        );
    }
}

/// Verifies every public signed division result used by the benchmark batch.
///
/// The batch uses a negative dividend, so the checks exercise distinct
/// truncating, Euclidean, floor, and ceiling semantics rather than only the
/// common positive-input case.
pub fn verify_mp_int_division_pairs(inputs: &[(MpInt, MpInt)]) {
    for (left, right) in inputs {
        let (quotient, remainder) = left
            .div_rem(right)
            .expect("division benchmark divisor must be non-zero");
        let reconstructed = (&quotient * right) + &remainder;
        assert_eq!(&reconstructed, left, "signed division identity");
        assert!(
            remainder.abs() < right.abs(),
            "signed division remainder must be smaller than divisor magnitude"
        );
        assert_eq!(left.div_trunc(right), quotient, "signed quotient path");
        assert_eq!(left.rem_trunc(right), remainder, "signed remainder path");

        let (euclidean_quotient, euclidean_remainder) = left
            .div_rem_euclid(right)
            .expect("division benchmark divisor must be non-zero");
        let euclidean_reconstructed = (&euclidean_quotient * right) + &euclidean_remainder;
        assert_eq!(&euclidean_reconstructed, left, "signed Euclidean identity");
        assert!(
            !euclidean_remainder.is_negative(),
            "Euclidean remainder sign"
        );
        assert_eq!(
            left.div_euclid(right),
            euclidean_quotient,
            "signed Euclidean quotient"
        );
        assert_eq!(
            left.rem_euclid(right),
            euclidean_remainder,
            "signed Euclidean remainder"
        );

        let (floor_quotient, floor_remainder) = left
            .div_rem_floor(right)
            .expect("division benchmark divisor must be non-zero");
        let floor_reconstructed = (&floor_quotient * right) + &floor_remainder;
        assert_eq!(&floor_reconstructed, left, "signed floor identity");
        if !floor_remainder.is_zero() {
            assert_eq!(
                floor_remainder.is_negative(),
                right.is_negative(),
                "floor remainder sign"
            );
        }
        assert_eq!(
            left.div_floor(right),
            floor_quotient,
            "signed floor quotient"
        );
        assert_eq!(
            left.mod_floor(right),
            floor_remainder,
            "signed floor remainder"
        );
        assert_eq!(
            left.div_ceil(right),
            left.checked_div_ceil(right)
                .expect("ceiling division must fit"),
            "signed ceiling quotient"
        );
    }
}

/// Verifies that a FLINT input or result is numerically equal to an `MpUint`.
///
/// The conversion and comparison run while constructing a benchmark cell, so
/// neither the reference conversion nor this assertion is timed.
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub fn verify_flint_matches_mp(expected: &MpUint, actual: &FlintInt) {
    let expected_flint = FlintInt::from_str_radix(&format!("{expected:x}"), 16);
    assert!(
        actual == &expected_flint,
        "FLINT benchmark value differs from the MpUint reference"
    );
}
