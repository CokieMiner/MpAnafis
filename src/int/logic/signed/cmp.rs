//! Signed integer comparison and hashing implementations.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::InternalArbiInt;

impl PartialEq for InternalArbiInt {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.is_positive == other.is_positive && self.abs == other.abs
    }
}

impl Eq for InternalArbiInt {}

impl PartialOrd for InternalArbiInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalArbiInt {
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

impl Hash for InternalArbiInt {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.is_positive.hash(state);
        self.abs.hash(state);
    }
}
