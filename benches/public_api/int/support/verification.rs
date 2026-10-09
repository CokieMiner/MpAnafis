//! Untimed benchmark input and result verification.

use mp_anafis::{MpInt, MpUint};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use super::FlintInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::Outcome;

/// Checks operand identity and operation outcomes outside the timed loop.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_pair<I: Outcome, J: Outcome, O: Outcome, P: Outcome>(
    inputs: &[I],
    peers: &[J],
    operation: impl Fn(&I) -> O,
    peer_operation: impl Fn(&J) -> P,
) {
    assert_eq!(inputs.len(), peers.len(), "paired batch lengths must match");
    for (input, peer) in inputs.iter().zip(peers) {
        assert_eq!(
            input.encode(),
            peer.encode(),
            "paired operands must be identical"
        );
        assert_eq!(
            operation(input).encode(),
            peer_operation(peer).encode(),
            "paired outcomes must agree"
        );
    }
}

/// Validates assignment outputs and operand identity before measurement.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_assignment<I: Outcome, J: Outcome, S: Outcome, T: Outcome>(
    inputs: &[I],
    peers: &[J],
    mut state: S,
    mut peer_state: T,
    operation: impl Fn(&mut S, &I),
    peer_operation: impl Fn(&mut T, &J),
) {
    assert_eq!(inputs.len(), peers.len(), "paired batch lengths must match");
    for (input, peer) in inputs.iter().zip(peers) {
        assert_eq!(
            input.encode(),
            peer.encode(),
            "paired operands must be identical"
        );
        operation(&mut state, input);
        peer_operation(&mut peer_state, peer);
        assert_eq!(
            state.encode(),
            peer_state.encode(),
            "paired assignment outcomes must agree"
        );
    }
}

/// Verifies every public unsigned division result used by the benchmark batch.
///
/// This runs while constructing a benchmark cell, outside its timed closure.
/// The checks cover the quotient/remainder identity, the quotient-only and
/// remainder-only paths, and the unsigned rounding aliases.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Untimed public unsigned arithmetic reconstructs and checks the division identity"
)]
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
            Some(quotient),
            "unsigned checked quotient"
        );
        assert_eq!(
            left.checked_rem(right),
            Some(remainder),
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
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Untimed public signed arithmetic reconstructs and checks each rounded division identity"
)]
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

/// Compares GCDs exactly and validates both libraries' unsigned coefficient
/// contracts in GMP arithmetic. Representatives are nonunique when gcd > 1;
/// exact tuple equality would reject valid coefficient choices.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Untimed GMP products and remainders verify the extended-GCD coefficient congruences"
)]
pub fn verify_extended_gcd_pairs(
    inputs: &[(MpUint, MpUint)],
    peers: &[(Integer, Integer)],
    operation: impl Fn(&(MpUint, MpUint)) -> Option<(MpUint, MpUint, MpUint)>,
    peer_operation: impl Fn(&(Integer, Integer)) -> Option<(Integer, Integer, Integer)>,
) {
    assert_eq!(inputs.len(), peers.len(), "paired batch lengths match");
    for (input, peer) in inputs.iter().zip(peers) {
        assert_eq!(input.encode(), peer.encode(), "paired inputs are identical");
        let (gcd, x, y) = operation(input).expect("positive extended-GCD operands");
        let (reference, peer_x, peer_y) = peer_operation(peer).expect("positive GMP operands");
        assert_eq!(gcd.encode(), reference.encode(), "GCD agrees with GMP");
        let coefficient_x =
            Integer::from_str_radix(&x.to_string_radix(16), 16).expect("coefficient parses");
        let coefficient_y =
            Integer::from_str_radix(&y.to_string_radix(16), 16).expect("coefficient parses");
        let (a, b) = peer;
        for (first, second) in [(coefficient_x, coefficient_y), (peer_x, peer_y)] {
            assert!(first >= 0 && &first < b, "first coefficient is in [0,b)");
            assert!(second >= 0 && &second < a, "second coefficient is in [0,a)");
            assert_eq!(
                Integer::from(a * &first) % b,
                Integer::from(&reference % b),
                "a*x agrees with gcd modulo b"
            );
            assert_eq!(
                Integer::from(b * &second) % a,
                Integer::from(&reference % a),
                "b*y agrees with gcd modulo a"
            );
        }
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
