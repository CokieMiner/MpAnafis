//! Primitive and cross-type conversion trait implementations.

use crate::error::ArbiError;

use super::{ArbiInt, ArbiUint, InternalArbiInt, InternalArbiUint, Precision};

macro_rules! impl_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for ArbiUint {
                fn from(value: $t) -> Self {
                    let internal = InternalArbiUint::from_u128(u128::from(value));
                    let required = internal.required_unsigned_bits_for_bounded_storage();
                    let precision = Precision::for_ambient_construction(required);
                    let result = Self {
                        value: internal,
                        precision,
                    };
                    result.debug_assert_valid();
                    result
                }
            }

            impl From<$t> for ArbiInt {
                fn from(value: $t) -> Self {
                    let internal = InternalArbiInt {
                        abs: InternalArbiUint::from_u128(u128::from(value)),
                        is_positive: true,
                    };
                    let required = internal.required_signed_bits_for_bounded_storage();
                    let precision = Precision::for_ambient_construction(required);
                    let result = Self {
                        value: internal,
                        precision,
                    };
                    result.debug_assert_valid();
                    result
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),*) => {
        $(
            impl TryFrom<$t> for ArbiUint {
                type Error = ArbiError;

                fn try_from(value: $t) -> Result<Self, Self::Error> {
                    if value < 0 {
                        return Err(ArbiError::NegativeInput);
                    }
                    let internal = InternalArbiUint::from_u128(u128::from(value.unsigned_abs()));
                    let required = internal.required_unsigned_bits_for_bounded_storage();
                    let precision = Precision::for_ambient_construction(required);
                    let result = Self {
                        value: internal,
                        precision,
                    };
                    result.debug_assert_valid();
                    Ok(result)
                }
            }

            impl From<$t> for ArbiInt {
                fn from(value: $t) -> Self {
                    let is_negative = value < 0;
                    let abs_value = u128::from(value.unsigned_abs());
                    let internal = InternalArbiInt {
                        abs: InternalArbiUint::from_u128(abs_value),
                        is_positive: !is_negative,
                    };
                    let required = internal.required_signed_bits_for_bounded_storage();
                    let precision = Precision::for_ambient_construction(required);
                    let result = Self {
                        value: internal,
                        precision,
                    };
                    result.debug_assert_valid();
                    result
                }
            }
        )*
    };
}

impl From<usize> for ArbiUint {
    fn from(value: usize) -> Self {
        let internal =
            InternalArbiUint::from_u128(u128::try_from(value).expect("usize always fits in u128"));
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl From<usize> for ArbiInt {
    fn from(value: usize) -> Self {
        let internal = InternalArbiInt {
            abs: InternalArbiUint::from_u128(
                u128::try_from(value).expect("usize always fits in u128"),
            ),
            is_positive: true,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl TryFrom<isize> for ArbiUint {
    type Error = ArbiError;
    fn try_from(value: isize) -> Result<Self, Self::Error> {
        if value < 0 {
            return Err(ArbiError::NegativeInput);
        }
        let internal =
            InternalArbiUint::from_u128(u128::try_from(value.unsigned_abs()).unwrap_or(0));
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}

impl From<isize> for ArbiInt {
    fn from(value: isize) -> Self {
        let is_negative = value < 0;
        let abs_value = u128::try_from(value.unsigned_abs()).unwrap_or(0);
        let internal = InternalArbiInt {
            abs: InternalArbiUint::from_u128(abs_value),
            is_positive: !is_negative,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl From<u128> for ArbiUint {
    fn from(value: u128) -> Self {
        let internal = InternalArbiUint::from_u128(value);
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl From<u128> for ArbiInt {
    fn from(value: u128) -> Self {
        let internal = InternalArbiInt {
            abs: InternalArbiUint::from_u128(value),
            is_positive: true,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl TryFrom<i128> for ArbiUint {
    type Error = ArbiError;
    fn try_from(value: i128) -> Result<Self, Self::Error> {
        if value < 0 {
            return Err(ArbiError::NegativeInput);
        }
        let internal = InternalArbiUint::from_u128(value.unsigned_abs());
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}

impl From<i128> for ArbiInt {
    fn from(value: i128) -> Self {
        let is_negative = value < 0;
        let abs_value = value.unsigned_abs();
        let internal = InternalArbiInt {
            abs: InternalArbiUint::from_u128(abs_value),
            is_positive: !is_negative,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl From<ArbiUint> for ArbiInt {
    fn from(value: ArbiUint) -> Self {
        let result = Self {
            value: InternalArbiInt {
                abs: value.value,
                is_positive: true,
            },
            precision: value.precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl TryFrom<ArbiInt> for ArbiUint {
    type Error = ArbiError;
    fn try_from(value: ArbiInt) -> Result<Self, Self::Error> {
        if value.is_negative() {
            return Err(ArbiError::NegativeInput);
        }
        Ok(Self {
            value: value.value.abs,
            precision: value.precision,
        })
    }
}

macro_rules! impl_try_from_arbi_uint_unsigned {
    ($($prim:ty),*) => {
        $(
            impl TryFrom<ArbiUint> for $prim {
                type Error = ArbiError;
                fn try_from(value: ArbiUint) -> Result<Self, Self::Error> {
                    value.to_u128()
                        .and_then(|v| <$prim>::try_from(v).ok())
                        .ok_or(ArbiError::IntegerConversionLoss)
                }
            }
        )*
    };
}

macro_rules! impl_try_from_arbi_int_unsigned {
    ($($prim:ty),*) => {
        $(
            impl TryFrom<ArbiInt> for $prim {
                type Error = ArbiError;
                fn try_from(value: ArbiInt) -> Result<Self, Self::Error> {
                    if value.is_negative() {
                        return Err(ArbiError::IntegerConversionLoss);
                    }
                    value.value.abs.to_u128()
                        .and_then(|v| <$prim>::try_from(v).ok())
                        .ok_or(ArbiError::IntegerConversionLoss)
                }
            }
        )*
    };
}

macro_rules! impl_try_from_arbi_int_signed {
    ($(($prim:ty, $unsigned:ty)),*) => {
        $(
            impl TryFrom<ArbiInt> for $prim {
                type Error = ArbiError;
                fn try_from(value: ArbiInt) -> Result<Self, Self::Error> {
                    let abs = value.value.abs.to_u128()
                        .ok_or(ArbiError::IntegerConversionLoss)?;
                    let max_abs = 1_u128 << (<$unsigned>::BITS - 1);
                    let is_negative = value.is_negative();
                    if is_negative && abs == max_abs {
                        return Ok(<$prim>::MIN);
                    }
                    let Ok(magnitude) = <$prim>::try_from(abs) else {
                        return Err(ArbiError::IntegerConversionLoss);
                    };
                    if is_negative {
                        Ok(magnitude.wrapping_neg())
                    } else {
                        Ok(magnitude)
                    }
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64);
impl_from_signed!(i8, i16, i32, i64);
impl_try_from_arbi_uint_unsigned!(u8, u16, u32, u64, u128, usize);
impl_try_from_arbi_int_unsigned!(u8, u16, u32, u64, u128, usize);
impl_try_from_arbi_int_signed!(
    (i8, u8),
    (i16, u16),
    (i32, u32),
    (i64, u64),
    (i128, u128),
    (isize, usize)
);
