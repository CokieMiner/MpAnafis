//! Signed integer comparison, total ordering, and hashing implementations.
//!
//! Canonical zero has a positive sign, so equality and hashing use the same
//! normalized sign and magnitude. Negative values reverse magnitude ordering.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::InternalMpInt;

impl PartialEq for InternalMpInt {
    /// Compares normalized sign and magnitude.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.is_positive == other.is_positive && self.abs == other.abs
    }
}

impl Eq for InternalMpInt {}

impl PartialOrd for InternalMpInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalMpInt {
    /// Total ordering on signed integers.
    ///
    /// Positive values order by magnitude ($a < b \iff |a| < |b|$).
    /// Negative values reverse magnitude ordering ($-a < -b \iff |b| < |a|$).
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_positive, other.is_positive) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (true, true) => self.abs.cmp(&other.abs),
            (false, false) => other.abs.cmp(&self.abs),
        }
    }
}

impl Hash for InternalMpInt {
    /// Hashes normalized sign and magnitude.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.is_positive.hash(state);
        self.abs.hash(state);
    }
}
