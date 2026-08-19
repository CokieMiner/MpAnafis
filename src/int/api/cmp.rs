//! Comparison and hashing trait implementations for integer API types.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{MpInt, MpUint};

impl PartialEq for MpInt {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for MpInt {}

impl Hash for MpInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialOrd for MpInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MpInt {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialEq for MpUint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for MpUint {}

impl Hash for MpUint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialOrd for MpUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MpUint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialEq<MpInt> for MpUint {
    fn eq(&self, other: &MpInt) -> bool {
        if other.is_negative() {
            return false;
        }
        self.value == other.value.abs
    }
}

impl PartialEq<MpUint> for MpInt {
    fn eq(&self, other: &MpUint) -> bool {
        if self.is_negative() {
            return false;
        }
        self.value.abs == other.value
    }
}

impl PartialOrd<MpInt> for MpUint {
    fn partial_cmp(&self, other: &MpInt) -> Option<Ordering> {
        if other.is_negative() {
            return Some(Ordering::Greater);
        }
        self.value.partial_cmp(&other.value.abs)
    }
}

impl PartialOrd<MpUint> for MpInt {
    fn partial_cmp(&self, other: &MpUint) -> Option<Ordering> {
        if self.is_negative() {
            return Some(Ordering::Less);
        }
        self.value.abs.partial_cmp(&other.value)
    }
}
