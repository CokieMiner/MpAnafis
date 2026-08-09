//! Shared validation helpers for public integer API operations.

use super::{MpInt, MpUint};

impl MpUint {
    #[inline]
    #[track_caller]
    pub(crate) fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_unsigned_bits_for_bounded_storage() <= bits,
                "MpUint {operation} overflow for Bounded({bits})"
            );
        }
    }
}

impl MpInt {
    #[inline]
    #[track_caller]
    pub(crate) fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_signed_bits_for_bounded_storage() <= bits,
                "MpInt {operation} overflow for Bounded({bits})"
            );
        }
    }
}
