//! Comparison and hashing trait implementations for integer API types.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{ArbiInt, ArbiUint};

impl PartialEq for ArbiInt {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for ArbiInt {}

impl Hash for ArbiInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialOrd for ArbiInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ArbiInt {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialEq for ArbiUint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for ArbiUint {}

impl Hash for ArbiUint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialOrd for ArbiUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ArbiUint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialEq<ArbiInt> for ArbiUint {
    fn eq(&self, other: &ArbiInt) -> bool {
        if other.is_negative() {
            return false;
        }
        self.value == other.value.abs
    }
}

impl PartialEq<ArbiUint> for ArbiInt {
    fn eq(&self, other: &ArbiUint) -> bool {
        if self.is_negative() {
            return false;
        }
        self.value.abs == other.value
    }
}

impl PartialOrd<ArbiInt> for ArbiUint {
    fn partial_cmp(&self, other: &ArbiInt) -> Option<Ordering> {
        if other.is_negative() {
            return Some(Ordering::Greater);
        }
        self.value.partial_cmp(&other.value.abs)
    }
}

impl PartialOrd<ArbiUint> for ArbiInt {
    fn partial_cmp(&self, other: &ArbiUint) -> Option<Ordering> {
        if self.is_negative() {
            return Some(Ordering::Less);
        }
        self.value.abs.partial_cmp(&other.value)
    }
}
