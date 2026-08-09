//! Signed integer capacity management APIs.

use core::mem::swap;

use super::{DebugVerbose, MpInt};

impl MpInt {
    /// Validates internal invariants. Panics if invariants are violated.
    #[inline]
    #[track_caller]
    pub(crate) fn debug_assert_valid(&self) {
        if cfg!(debug_assertions) {
            if self.value.abs.is_zero() {
                assert!(
                    self.value.is_positive,
                    "MpInt canonical zero must have a positive sign"
                );
            }

            if let Some(bits) = self.precision.significant_bits() {
                assert!(
                    self.value.required_signed_bits_for_bounded_storage() <= bits,
                    "MpInt magnitude exceeds its bounded precision of {bits} bits"
                );
            }
        }
    }

    /// Reserves capacity for at least `additional` more native-width limbs.
    /// This is useful to avoid repeated allocations during operations that grow the integer.
    pub fn reserve(&mut self, additional: usize) {
        self.value.abs.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more native-width limbs.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.value.abs.reserve_exact(additional);
    }

    /// Shrinks the capacity of the value as much as possible to save memory.
    pub fn shrink_to_fit(&mut self) {
        self.value.abs.shrink_to_fit();
    }

    /// Returns the total number of native-width limbs the value can hold without reallocating.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.value.abs.capacity()
    }

    /// Returns a wrapper that displays the value and its precision when
    /// formatted with `Debug`.
    #[must_use]
    pub const fn as_debug_verbose(&self) -> DebugVerbose<'_, Self> {
        DebugVerbose(self)
    }

    /// Swaps the value and precision of `self` with `other` in O(1) time.
    pub const fn swap(&mut self, other: &mut Self) {
        self.value.abs.swap(&mut other.value.abs);
        swap(&mut self.value.is_positive, &mut other.value.is_positive);
        swap(&mut self.precision, &mut other.precision);
    }
}
