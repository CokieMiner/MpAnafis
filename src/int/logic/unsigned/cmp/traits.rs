//! Standard comparison and hashing traits.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{InternalArbiUint, Limb};

impl InternalArbiUint {
    /// Compares two normalized limb slices as unsigned integers.
    #[must_use]
    pub fn cmp_limbs(left: &[Limb], right: &[Limb]) -> Ordering {
        match left.len().cmp(&right.len()) {
            Ordering::Equal => {
                // Reverse iteration compares most significant limbs first. The
                // zip of equal-length slices is bounds-check free by
                // construction and unrolls to direct limb compares.
                for (left_limb, right_limb) in left.iter().rev().zip(right.iter().rev()) {
                    let ordering = left_limb.cmp(right_limb);
                    if !ordering.is_eq() {
                        return ordering;
                    }
                }
                Ordering::Equal
            }
            ordering @ (Ordering::Less | Ordering::Greater) => ordering,
        }
    }
}

impl PartialEq for InternalArbiUint {
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "hot path: eq is called repeatedly in tight comparison loops"
    )]
    fn eq(&self, other: &Self) -> bool {
        self.limbs() == other.limbs()
    }
}

impl Eq for InternalArbiUint {}

impl PartialOrd for InternalArbiUint {
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "hot path: partial_cmp is called repeatedly in sorting and comparison contexts"
    )]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalArbiUint {
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "hot path: cmp is called in every sorting/comparison operation"
    )]
    fn cmp(&self, other: &Self) -> Ordering {
        Self::cmp_limbs(self.limbs(), other.limbs())
    }
}

impl Hash for InternalArbiUint {
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "hot path: hash is called during hashing and set operations"
    )]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.limbs().hash(state);
    }
}
