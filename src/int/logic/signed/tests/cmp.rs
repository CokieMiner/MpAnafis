//! Equality, total ordering, and hash consistency of canonical representations.

extern crate std;

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use alloc::vec::Vec;

use proptest::test_runner::{Config, TestRunner};

use crate::int::{InternalMpInt, InternalMpUint};

use super::strategies::{public, signed, small_signed};

#[test]
fn ordering_matches_numeric_values_and_equal_representations_hash_equally() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(small_signed(), small_signed()),
            |((a, left), (b, right))| {
                assert_eq!(left.cmp(&right), a.cmp(&b));
                assert_eq!(left.partial_cmp(&right), Some(a.cmp(&b)));
                assert_eq!(left == right, a == b);
                Ok(())
            },
        )
        .expect("native equality and order agree");
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
            ),
            |(left, right, third)| {
                assert_eq!(left.cmp(&right), public(&left).cmp(&public(&right)));
                assert_eq!(left.partial_cmp(&right), Some(left.cmp(&right)));
                assert_eq!(left.cmp(&right), right.cmp(&left).reverse());
                assert_eq!(left == right, left.cmp(&right).is_eq());
                if left <= right && right <= third {
                    assert!(left <= third);
                }
                let mut padded = Vec::from(left.abs.limbs());
                padded.extend_from_slice(&[0, 0, 0]);
                let normalized = InternalMpInt {
                    abs: InternalMpUint::from_limbs(padded),
                    is_positive: left.is_positive,
                };
                assert_eq!(normalized, left);
                let mut original_hash = DefaultHasher::new();
                let mut normalized_hash = DefaultHasher::new();
                left.hash(&mut original_hash);
                normalized.hash(&mut normalized_hash);
                assert_eq!(original_hash.finish(), normalized_hash.finish());
                assert_eq!(left.cmp(&left), core::cmp::Ordering::Equal);
                Ok(())
            },
        )
        .expect("wide ordering and normalized hashes agree");
}
