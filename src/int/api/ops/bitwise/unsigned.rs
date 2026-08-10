//! Unsigned bitwise operator trait implementations.

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use super::ArbiUint;

impl BitAnd<Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<&Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAndAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitand_assign(&mut self, rhs: Self) {
        BitAndAssign::bitand_assign(self, &rhs);
    }
}

impl BitAndAssign<&Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitand_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: self.value.bitand(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl BitOr<Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<&Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOrAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitor_assign(&mut self, rhs: Self) {
        BitOrAssign::bitor_assign(self, &rhs);
    }
}

impl BitOrAssign<&Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitor_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: self.value.bitor(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl BitXor<Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<&Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXorAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitxor_assign(&mut self, rhs: Self) {
        BitXorAssign::bitxor_assign(self, &rhs);
    }
}

impl BitXorAssign<&Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn bitxor_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: self.value.bitxor(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl Not for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let width = self
            .precision
            .significant_bits()
            .expect("`!` on unlimited ArbiUint: use `.not_with_width(width)` or `.try_not()`");
        let result = Self {
            value: self.value.not(width),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl Not for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let width = self
            .precision
            .significant_bits()
            .expect("`!` on &ArbiUint unlimited: use `.not_with_width(width)` or `.try_not()`");
        let result = ArbiUint {
            value: self.value.not(width),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}
