//! Signed bitwise operator trait implementations.

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use super::MpInt;

impl BitAnd<Self> for MpInt {
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

impl BitAnd<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: (&self.value).bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAnd<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitand(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitand(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitAndAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitand_assign(&mut self, rhs: Self) {
        BitAndAssign::bitand_assign(self, &rhs);
    }
}

impl BitAndAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitand_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: (&self.value).bitand(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl BitOr<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: (&self.value).bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: (&self.value).bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOr<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitor(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitOrAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitor_assign(&mut self, rhs: Self) {
        BitOrAssign::bitor_assign(self, &rhs);
    }
}

impl BitOrAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitor_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: (&self.value).bitor(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl BitXor<Self> for MpInt {
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

impl BitXor<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: (&self.value).bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXor<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn bitxor(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: (&self.value).bitxor(&rhs.value),
            precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl BitXorAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitxor_assign(&mut self, rhs: Self) {
        BitXorAssign::bitxor_assign(self, &rhs);
    }
}

impl BitXorAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn bitxor_assign(&mut self, rhs: &Self) {
        let result = Self {
            value: (&self.value).bitxor(&rhs.value),
            precision: self.precision,
        };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        *self = result;
    }
}

impl Not for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let precision = self.precision;
        let value = !self.value;
        let result = Self { value, precision };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}

impl Not for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        let precision = self.precision;
        let value = !&self.value;
        let result = MpInt { value, precision };
        result.assert_fits("bitwise operation");
        result.debug_assert_valid();
        result
    }
}
