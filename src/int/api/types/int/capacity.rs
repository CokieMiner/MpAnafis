//! Signed integer capacity management APIs.

use core::mem::swap;

use super::{DebugVerbose, MpInt};

impl MpInt {
    /// Returns a wrapper that displays the value and its precision when
    /// formatted with `Debug`.
    #[must_use]
    pub const fn as_debug_verbose(&self) -> DebugVerbose<'_, Self> {
        DebugVerbose(self)
    }

    /// Reserves capacity for at least `additional` native-width limbs.
    pub fn reserve(&mut self, additional: usize) {
        self.value.abs.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` native-width limbs.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.value.abs.reserve_exact(additional);
    }

    /// Shrinks the capacity of the magnitude buffer to match its current limb count.
    pub fn shrink_to_fit(&mut self) {
        self.value.abs.shrink_to_fit();
    }

    /// Returns the total number of native-width limbs the value can hold without reallocating.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.value.abs.capacity()
    }

    /// Swaps the value and precision of `self` with `other` in $\mathcal{O}(1)$ time.
    pub const fn swap(&mut self, other: &mut Self) {
        self.value.abs.swap(&mut other.value.abs);
        swap(&mut self.value.is_positive, &mut other.value.is_positive);
        swap(&mut self.precision, &mut other.precision);
    }
}
