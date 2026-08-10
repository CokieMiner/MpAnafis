//! Shared validation helpers for public integer API operations.

use super::{ArbiInt, ArbiUint};

impl ArbiUint {
    #[inline]
    #[track_caller]
    pub(crate) fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_unsigned_bits_for_bounded_storage() <= bits,
                "ArbiUint {operation} overflow for Bounded({bits})"
            );
        }
    }
}

impl ArbiInt {
    #[inline]
    #[track_caller]
    pub(crate) fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_signed_bits_for_bounded_storage() <= bits,
                "ArbiInt {operation} overflow for Bounded({bits})"
            );
        }
    }
}
