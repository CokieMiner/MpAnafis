//! Unsigned integer capacity management APIs.

use core::mem::swap;

use super::{ArbiUint, DebugVerbose};
impl ArbiUint {
    /// Validates internal invariants. Panics if invariants are violated.
    #[inline]
    #[track_caller]
    pub(crate) fn debug_assert_valid(&self) {
        if cfg!(debug_assertions)
            && let Some(bits) = self.precision.significant_bits()
        {
            assert!(
                self.value.required_unsigned_bits_for_bounded_storage() <= bits,
                "ArbiUint magnitude exceeds its bounded precision of {bits} bits"
            );
        }
    }

    /// Returns a wrapper that displays the value and its precision when
    /// formatted with `Debug`.
    #[must_use]
    pub const fn as_debug_verbose(&self) -> DebugVerbose<'_, Self> {
        DebugVerbose(self)
    }

    /// Reserves capacity for at least `additional` more native-width limbs.
    /// This is useful to avoid repeated allocations during operations that grow the integer.
    pub fn reserve(&mut self, additional: usize) {
        self.value.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more native-width limbs.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.value.reserve_exact(additional);
    }

    /// Shrinks the capacity of the value as much as possible to save memory.
    pub fn shrink_to_fit(&mut self) {
        self.value.shrink_to_fit();
    }

    /// Returns the total number of native-width limbs the value can hold without reallocating.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.value.capacity()
    }

    /// Swaps the value and precision of `self` with `other` in O(1) time.
    pub const fn swap(&mut self, other: &mut Self) {
        self.value.swap(&mut other.value);
        swap(&mut self.precision, &mut other.precision);
    }
}
