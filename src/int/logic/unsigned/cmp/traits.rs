//! Standard comparison and hashing traits.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{InternalMpUint, Limb};

impl PartialEq for InternalMpUint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Canonical magnitudes have one active limb sequence regardless of
        // inline/heap storage, spare capacity, or inactive inline slots.
        self.limbs() == other.limbs()
    }
}

impl Eq for InternalMpUint {}

impl PartialOrd for InternalMpUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Unsigned integers are totally ordered; this Option is always Some.
        Some(self.cmp(other))
    }
}

impl Ord for InternalMpUint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Self::cmp_limbs(self.limbs(), other.limbs())
    }
}

impl Hash for InternalMpUint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the same canonical sequence used by equality, including its
        // length; allocation metadata and inactive limbs do not participate.
        self.limbs().hash(state);
    }
}

impl InternalMpUint {
    /// Compares little-endian limb slices as unsigned integers.
    ///
    /// Equal-length slices may contain leading zero padding. For unequal
    /// lengths, the longer slice must have a nonzero highest limb; normalized
    /// magnitudes satisfy this contract, including the empty encoding of zero.
    #[must_use]
    pub fn cmp_limbs(left: &[Limb], right: &[Limb]) -> Ordering {
        match left.len().cmp(&right.len()) {
            Ordering::Equal => {
                // In radix B = 2^Limb::BITS, the highest differing digit
                // dominates every lower digit. Equal lengths give both reverse
                // iterators the same trip count, including padded windows;
                // shared or overlapping read-only slices are valid.
                for (left_limb, right_limb) in left.iter().rev().zip(right.iter().rev()) {
                    let ordering = left_limb.cmp(right_limb);
                    if !ordering.is_eq() {
                        return ordering;
                    }
                }
                Ordering::Equal
            }
            ordering @ (Ordering::Less | Ordering::Greater) => {
                let longer = if ordering.is_lt() { right } else { left };
                debug_assert!(
                    longer.last().is_some_and(|&top| top != 0),
                    "unequal-width comparison requires a nonzero highest limb in the longer slice"
                );
                // A nonzero top digit gives the longer value at least
                // B^(longer.len()-1); every shorter value is below that bound.
                ordering
            }
        }
    }
}
