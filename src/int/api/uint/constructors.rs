//! Unsigned integer constructors and precision constructors.

use crate::error::ArbiError;

use super::{ArbiUint, BoundedPrecision, InternalArbiUint, Precision, PrecisionContext};

impl ArbiUint {
    // Constructors
    /// Creates a zero-valued `ArbiUint` with unlimited precision.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn zero() -> Self {
        let result = Self {
            value: InternalArbiUint::zero(),
            precision: Precision::Unlimited,
        };
        result.debug_assert_valid();
        result
    }
    /// Creates a one-valued `ArbiUint` with unlimited precision.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn one() -> Self {
        let result = Self {
            value: InternalArbiUint::one(),
            precision: Precision::Unlimited,
        };
        result.debug_assert_valid();
        result
    }
    /// Creates an `ArbiUint` with the given initial limb capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let result = Self {
            value: InternalArbiUint::with_capacity(capacity),
            precision: Precision::from(PrecisionContext::active()),
        };
        result.debug_assert_valid();
        result
    }

    // Precision-aware constructors
    // ------------------------------------------------------------------

    /// Create a new `ArbiUint` from any value that implements `Into<ArbiUint>`,
    /// applying ambient precision.
    #[must_use]
    pub fn new<T>(value: T) -> Self
    where
        Self: From<T>,
    {
        Self::from(value)
    }

    /// Create an `ArbiUint` with an explicit bounded precision check.
    ///
    /// # Errors
    ///
    /// Returns `ArbiError::PrecisionExceeded` if the magnitude exceeds the
    /// given bit width.
    pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, ArbiError>
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        if v.value.significant_bits() > bits.get() {
            return Err(ArbiError::PrecisionExceeded);
        }
        let result = Self {
            value: v.value,
            precision: Precision::Bounded(bits),
        };
        result.debug_assert_valid();
        Ok(result)
    }

    /// Create an `ArbiUint` with wrapping precision (truncates to fit).
    ///
    /// # Panics
    /// Panics if the bit count is larger than fits in a `usize` index.
    #[must_use]
    pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        let mask_bits = bits.get();
        let wrapped = v.value.apply_wrapping(mask_bits);
        let result = Self {
            value: wrapped,
            precision: Precision::Bounded(bits),
        };
        result.debug_assert_valid();
        result
    }

    /// Create an `ArbiUint` with saturating precision (clamps to max value).
    #[must_use]
    pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        if v.value.significant_bits() <= bits.get() {
            let result = Self {
                value: v.value,
                precision: Precision::Bounded(bits),
            };
            result.debug_assert_valid();
            result
        } else {
            let result = Self {
                value: InternalArbiUint::max_for_bits(bits.get()),
                precision: Precision::Bounded(bits),
            };
            result.debug_assert_valid();
            result
        }
    }

    /// Returns the maximum value representable with the given bit width.
    ///
    /// # Panics
    /// Panics if `bits` is zero or `usize::MAX`.
    #[must_use]
    pub fn max_for_precision(bits: usize) -> Self {
        let precision = Precision::new_bounded(bits)
            .expect("bits must be in the bounded-precision range 1..usize::MAX");
        Self {
            value: InternalArbiUint::max_for_bits(bits),
            precision,
        }
    }

    /// Returns the minimum value representable with the given bit width
    /// (always zero for unsigned).
    ///
    /// # Panics
    ///
    /// Panics if `bits` is zero or `usize::MAX`.
    #[must_use]
    pub const fn min_for_precision(bits: usize) -> Self {
        let width = BoundedPrecision::new(bits)
            .expect("bits must be in the bounded-precision range 1..usize::MAX");
        Self::zero_with_precision(width)
    }

    /// Creates a zero value with the given bounded precision.
    ///
    #[must_use]
    pub const fn zero_with_precision(bits: BoundedPrecision) -> Self {
        Self {
            value: InternalArbiUint::zero(),
            precision: Precision::Bounded(bits),
        }
    }

    // ------------------------------------------------------------------
}
impl Default for ArbiUint {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
