//! Shared public constructors and mathematical range oracles.

#[cfg(feature = "std")]
use core::hash::{Hash, Hasher};
#[cfg(feature = "std")]
use std::collections::hash_map::DefaultHasher;

use alloc::vec::Vec;

use proptest::prelude::{Strategy, any};

use crate::{BoundedPrecision, MpInt, MpUint};

pub fn nz(bits: usize) -> BoundedPrecision {
    BoundedPrecision::new(bits).expect("valid bounded width")
}

pub fn uint(value: u64) -> MpUint {
    MpUint::zero() + MpUint::from(value)
}

/// Parses native words in little-endian order and removes ambient precision.
pub fn uint_from_words(words: &[usize]) -> MpUint {
    let length = words
        .len()
        .checked_mul(size_of::<usize>())
        .expect("test byte length fits");
    let mut bytes = Vec::with_capacity(length);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    MpUint::zero() + MpUint::from_le_bytes(&bytes)
}

pub fn exact_limb_vec(len: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(any::<usize>(), len)
}

/// Tests the mathematical signed interval without inspecting storage.
pub fn signed_fits(value: &MpInt, bits: usize) -> bool {
    let minimum = MpInt::min_for_precision(bits);
    let maximum = MpInt::max_for_precision(bits);
    minimum <= *value && *value <= maximum
}

#[cfg(feature = "std")]
pub fn hash_u64(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
