//! Signed integer constructors, including the bounded-precision constructors.
//!
//! Sign operations live in [`sign`](super::sign) and value predicates in
//! [`properties`](super::properties).

use crate::error::ArbiError;

use super::{
    ArbiInt, BoundedPrecision, InternalArbiInt, InternalArbiUint, Precision, PrecisionContext,
};

impl ArbiInt {
    /// Creates a zero-valued `ArbiInt` with unlimited precision.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn zero() -> Self {
        let result = Self {
            value: InternalArbiInt::zero(),
            precision: Precision::Unlimited,
        };
        result.debug_assert_valid();
        result
    }
    /// Creates a one-valued `ArbiInt` with unlimited precision.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn one() -> Self {
        let result = Self {
            value: InternalArbiInt::one(),
            precision: Precision::Unlimited,
        };
        result.debug_assert_valid();
        result
    }
    /// Creates an `ArbiInt` with the given initial limb capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let result = Self {
            value: InternalArbiInt::with_capacity(capacity),
            precision: Precision::from(PrecisionContext::active()),
        };
        result.debug_assert_valid();
        result
    }

    /// Creates a new `ArbiInt` from any value that implements `Into<ArbiInt>`.
    #[must_use]
    pub fn new<T>(value: T) -> Self
    where
        Self: From<T>,
    {
        Self::from(value)
    }

    /// Returns the value -1 (minus one) with unlimited precision.
    #[must_use]
    pub fn minus_one() -> Self {
        let result = Self {
            value: InternalArbiInt {
                abs: InternalArbiUint::one(),
                is_positive: false,
            },
            precision: Precision::Unlimited,
        };
        result.debug_assert_valid();
        result
    }

    /// Returns the maximum signed value representable with the given bit width
    /// (2^(bits-1) - 1).
    ///
    /// # Panics
    /// Panics if `bits` is zero or `usize::MAX`.
    #[must_use]
    pub fn max_for_precision(bits: usize) -> Self {
        let precision = Precision::new_bounded(bits)
            .expect("bits must be in the bounded-precision range 1..usize::MAX");
        let abs = InternalArbiUint::max_for_bits(bits.saturating_sub(1));
        Self {
            value: InternalArbiInt {
                abs,
                is_positive: true,
            },
            precision,
        }
    }

    /// Returns the minimum signed value representable with the given bit width
    /// (-2^(bits-1)).
    ///
    /// # Panics
    /// Panics if `bits` is zero or `usize::MAX`.
    #[must_use]
    pub fn min_for_precision(bits: usize) -> Self {
        let precision = Precision::new_bounded(bits)
            .expect("bits must be in the bounded-precision range 1..usize::MAX");
        let abs = InternalArbiUint::one().shl(bits.saturating_sub(1));
        Self {
            value: InternalArbiInt {
                abs,
                is_positive: false,
            },
            precision,
        }
    }

    /// Creates a zero value with the given bounded precision.
    ///
    #[must_use]
    pub const fn zero_with_precision(bits: BoundedPrecision) -> Self {
        Self {
            value: InternalArbiInt::zero(),
            precision: Precision::Bounded(bits),
        }
    }

    /// Creates an `ArbiInt` with bounded precision, checking the value fits.
    ///
    /// # Errors
    /// Returns `ArbiError::PrecisionExceeded` if the magnitude exceeds the
    /// given signed bit width.
    pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, ArbiError>
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        if v.value.required_signed_bits_for_bounded_storage() > bits.get() {
            return Err(ArbiError::PrecisionExceeded);
        }
        let result = Self {
            value: v.value,
            precision: Precision::Bounded(bits),
        };
        result.debug_assert_valid();
        Ok(result)
    }

    /// Creates an `ArbiInt` with wrapping precision (truncates to fit).
    ///
    /// # Panics
    /// Panics if the bit count is larger than fits in a `usize` index.
    #[must_use]
    pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        let sig = v.value.required_signed_bits_for_bounded_storage();
        if sig <= bits.get() {
            let result = Self {
                value: v.value,
                precision: Precision::Bounded(bits),
            };
            result.debug_assert_valid();
            result
        } else {
            // Truncate to N-bit two's complement
            let wrapped = v.value.apply_wrapping(bits.get());
            let result = Self {
                value: wrapped,
                precision: Precision::Bounded(bits),
            };
            result.debug_assert_valid();
            result
        }
    }

    /// Creates an `ArbiInt` with saturating precision (clamps to min/max).
    #[must_use]
    pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self
    where
        Self: From<T>,
    {
        let v = Self::from(value);
        if v.value.required_signed_bits_for_bounded_storage() <= bits.get() {
            let result = Self {
                value: v.value,
                precision: Precision::Bounded(bits),
            };
            result.debug_assert_valid();
            result
        } else if v.value.is_positive {
            Self::max_for_precision(bits.get())
        } else {
            Self::min_for_precision(bits.get())
        }
    }
}
impl Default for ArbiInt {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
