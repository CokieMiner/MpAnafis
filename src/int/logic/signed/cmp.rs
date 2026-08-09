//! Signed integer comparison and hashing implementations.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::InternalMpInt;

impl PartialEq for InternalMpInt {
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
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.is_positive.hash(state);
        self.abs.hash(state);
    }
}
