//! Unsigned bitwise operator trait implementations.

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use super::MpUint;

impl BitAnd<Self> for MpUint {
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

impl BitAnd<&Self> for MpUint {
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

impl BitAnd<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAndAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn bitand_assign(&mut self, rhs: Self) {
        BitAndAssign::bitand_assign(self, &rhs);
    }
}

impl BitAndAssign<&Self> for MpUint {
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

impl BitOr<Self> for MpUint {
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

impl BitOr<&Self> for MpUint {
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

impl BitOr<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOrAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn bitor_assign(&mut self, rhs: Self) {
        BitOrAssign::bitor_assign(self, &rhs);
    }
}

impl BitOrAssign<&Self> for MpUint {
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

impl BitXor<Self> for MpUint {
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

impl BitXor<&Self> for MpUint {
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

impl BitXor<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXorAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn bitxor_assign(&mut self, rhs: Self) {
        BitXorAssign::bitxor_assign(self, &rhs);
    }
}

impl BitXorAssign<&Self> for MpUint {
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

impl Not for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let width = self
            .precision
            .significant_bits()
            .expect("`!` on unlimited MpUint: use `.not_with_width(width)` or `.try_not()`");
        let result = Self {
            value: self.value.not(width),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl Not for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let width = self
            .precision
            .significant_bits()
            .expect("`!` on &MpUint unlimited: use `.not_with_width(width)` or `.try_not()`");
        let result = MpUint {
            value: self.value.not(width),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}
