//! Hash contracts and borrowed selection shared by both integer domains.

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::{Outcome, verify_pair};

/// Checks numeric operand identity and equal-value hashes within each library.
/// Hash encodings are library-specific and need not agree across libraries.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_hashes<I: Outcome + Clone, J: Outcome + Clone>(
    inputs: &[I],
    peers: &[J],
    operation: impl Fn(&I) -> u64,
    peer_operation: impl Fn(&J) -> u64,
) {
    assert_eq!(inputs.len(), peers.len(), "paired batch lengths must match");
    for (input, peer) in inputs.iter().zip(peers) {
        assert_eq!(
            input.encode(),
            peer.encode(),
            "paired operands must be identical"
        );
        assert_eq!(
            operation(input),
            operation(&input.clone()),
            "equal Mp values must hash equally"
        );
        assert_eq!(
            peer_operation(peer),
            peer_operation(&peer.clone()),
            "equal Rug values must hash equally"
        );
    }
}

/// Encodes borrowed selections during verification, outside the timed loop.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_selection<I: Outcome, J: Outcome, S: Outcome, T: Outcome>(
    inputs: &[I],
    peers: &[J],
    operation: impl for<'input> Fn(&'input I) -> &'input S,
    peer_operation: impl for<'input> Fn(&'input J) -> &'input T,
) {
    verify_pair(
        inputs,
        peers,
        |input| operation(input).encode(),
        |peer| peer_operation(peer).encode(),
    );
}

/// Computes one hash with a fresh deterministic hasher.
pub fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

// Named functions preserve the input/output lifetime relation when passed to
// paired timing and verification. Each returns a borrowed operand without a clone.
pub fn minimum<T: Ord>((left, right): &(T, T)) -> &T {
    left.min(right)
}

pub fn maximum<T: Ord>((left, right): &(T, T)) -> &T {
    left.max(right)
}

pub fn clamped<T: Ord>((value, lower, upper): &(T, T, T)) -> &T {
    value.clamp(lower, upper)
}
