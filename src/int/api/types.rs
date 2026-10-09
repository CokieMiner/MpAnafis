//! Public integer storage and private precision invariants.
//!
//! Constructors and operations maintain normalized magnitudes, positive zero,
//! and values within their declared precision. Implementation descendants
//! access private storage fields and validation methods.

#![expect(
    clippy::self_named_module_files,
    reason = "The dedicated type owner declares descendants that access its private fields; mod.rs remains a structural registry"
)]

use super::{
    AmbientPrecision, BoundedPrecision, InternalMpInt, InternalMpUint, Precision, PrecisionContext,
};

mod cmp;
mod convert;
mod int;
mod iter;
#[cfg(feature = "num-traits")]
mod num_traits;
mod ops;
mod string;
mod uint;

/// Arbitrary precision signed integer.
pub struct MpInt {
    /// Signed magnitude and canonical sign.
    value: InternalMpInt,
    /// Declared precision of the value.
    precision: Precision,
}

/// Arbitrary precision unsigned integer.
pub struct MpUint {
    /// Normalized unsigned magnitude.
    value: InternalMpUint,
    /// Declared precision of the value.
    precision: Precision,
}

/// Formats an integer with its precision metadata.
#[non_exhaustive]
pub struct DebugVerbose<'data, T>(pub &'data T);

impl Clone for MpInt {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            precision: self.precision,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value.abs.clone_from(&source.value.abs);
        self.value.is_positive = source.value.is_positive;
        self.precision = source.precision;
    }
}

impl Clone for MpUint {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            precision: self.precision,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value.clone_from(&source.value);
        self.precision = source.precision;
    }
}

impl Precision {
    /// Combines bounded widths by maximum; any unlimited operand gives unlimited precision.
    #[must_use]
    const fn combine_for_binary_op(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Bounded(a), Self::Bounded(b)) => {
                let max_bits = if a.get() >= b.get() { a } else { b };
                Self::Bounded(max_bits)
            }
            _ => Self::Unlimited,
        }
    }

    /// Uses the greater of the bounded ambient width and the value's required width.
    #[must_use]
    fn for_ambient_construction(required_bits: usize) -> Self {
        match PrecisionContext::active() {
            AmbientPrecision::Bounded(width) => {
                if required_bits <= width.get() {
                    Self::Bounded(width)
                } else {
                    Self::new_bounded(required_bits).unwrap_or(Self::Unlimited)
                }
            }
            AmbientPrecision::Unset | AmbientPrecision::Unlimited => Self::Unlimited,
        }
    }

    /// Preserves bounded ambient precision, rejecting values that exceed it.
    #[must_use]
    fn checked_for_ambient_construction(required_bits: usize) -> Option<Self> {
        match PrecisionContext::active() {
            AmbientPrecision::Unset | AmbientPrecision::Unlimited => Some(Self::Unlimited),
            AmbientPrecision::Bounded(width) => {
                (required_bits <= width.get()).then_some(Self::Bounded(width))
            }
        }
    }
}

impl MpUint {
    /// Validates that the magnitude fits the declared bounded width.
    #[inline]
    #[track_caller]
    fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_unsigned_bits_for_bounded_storage() <= bits,
                "MpUint {operation} overflow for Bounded({bits})"
            );
        }
    }

    /// Checks the declared bounded width in debug builds.
    #[inline]
    #[track_caller]
    fn debug_assert_valid(&self) {
        if cfg!(debug_assertions)
            && let Some(bits) = self.precision.significant_bits()
        {
            assert!(
                self.value.required_unsigned_bits_for_bounded_storage() <= bits,
                "MpUint magnitude exceeds its bounded precision of {bits} bits"
            );
        }
    }
}

impl MpInt {
    /// Rejects the bounded minimum, whose positive magnitude requires another sign bit.
    #[track_caller]
    fn assert_negation_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                !self.value.is_signed_min_for_width(bits),
                "MpInt {operation} overflow for Bounded({bits})"
            );
        }
    }

    /// Validates that the signed value fits the declared bounded width.
    #[inline]
    #[track_caller]
    fn assert_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                self.value.required_signed_bits_for_bounded_storage() <= bits,
                "MpInt {operation} overflow for Bounded({bits})"
            );
        }
    }

    /// Checks canonical zero and the declared bounded width in debug builds.
    #[inline]
    #[track_caller]
    fn debug_assert_valid(&self) {
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
}
