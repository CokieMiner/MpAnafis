//! Subtraction and negation operator trait implementations.

use core::ops::{Neg, Sub, SubAssign};

use super::{InternalMpInt, MpInt, MpUint};

impl Sub<Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn sub(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let underflowed = self.value.sub_assign_with_underflow(&rhs.value);
        assert!(!underflowed, "MpUint underflow");
        self.precision = precision;
        // `self - rhs <= self`, and result precision is at least the original
        // left precision, so a successful unsigned subtraction always fits.
        self.debug_assert_valid();
        self
    }
}

impl Sub<&Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let underflowed = self.value.sub_assign_with_underflow(&rhs.value);
        assert!(!underflowed, "MpUint underflow");
        self.precision = precision;
        // `self - rhs <= self`, and result precision is at least the original
        // left precision, so a successful unsigned subtraction always fits.
        self.debug_assert_valid();
        self
    }
}

impl Sub<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn sub(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let (value, underflowed) = self.value.sub_with_underflow(&rhs.value);
        assert!(!underflowed, "MpUint underflow");
        let result = MpUint { value, precision };
        // `self - rhs <= self`, and result precision is at least the original
        // left precision, so a successful unsigned subtraction always fits.
        result.debug_assert_valid();
        result
    }
}

impl Sub<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn sub(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let (value, underflowed) = self.value.sub_with_underflow(&rhs.value);
        assert!(!underflowed, "MpUint underflow");
        let result = MpUint { value, precision };
        // `self - rhs <= self`, and result precision is at least the original
        // left precision, so a successful unsigned subtraction always fits.
        result.debug_assert_valid();
        result
    }
}

impl SubAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn sub_assign(&mut self, rhs: Self) {
        SubAssign::sub_assign(self, &rhs);
    }
}

impl SubAssign<&Self> for MpUint {
    #[inline]
    #[track_caller]
    fn sub_assign(&mut self, rhs: &Self) {
        // The limb kernel writes a wrapping residue on underflow. Check the
        // mathematical precondition first so a caught panic leaves `self`
        // unchanged rather than exposing that internal residue.
        assert!(self.value >= rhs.value, "MpUint underflow");
        self.value.sub_assign(&rhs.value);
        // A successful unsigned subtraction cannot exceed the unchanged
        // receiver precision because its result is at most the old receiver.
        self.debug_assert_valid();
    }
}

impl Sub<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn sub(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        if self.value.abs.limbs().len() >= rhs.value.abs.limbs().len() {
            self.value.sub_assign(&rhs.value);
        } else {
            let mut result_val = rhs.value;
            result_val.sub_assign(&self.value);
            if !result_val.abs.is_zero() {
                result_val.is_positive = !result_val.is_positive;
            }
            self.value = result_val;
        }
        self.precision = precision;
        self.assert_fits("subtraction");
        self.debug_assert_valid();
        self
    }
}

impl Sub<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        self.value.sub_assign(&rhs.value);
        self.precision = precision;
        self.assert_fits("subtraction");
        self.debug_assert_valid();
        self
    }
}

impl Sub<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn sub(self, mut rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.sub_assign(&self.value);
        if !rhs.value.abs.is_zero() {
            rhs.value.is_positive = !rhs.value.is_positive;
        }
        rhs.precision = precision;
        rhs.assert_fits("subtraction");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Sub<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn sub(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: self.value.sub(&rhs.value),
            precision,
        };
        result.assert_fits("subtraction");
        result.debug_assert_valid();
        result
    }
}

impl SubAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn sub_assign(&mut self, mut rhs: Self) {
        if !self.precision.is_unlimited() {
            // Compute `rhs - self` in the owned right-hand buffer, then negate
            // it to obtain `self - rhs`. Validation precedes the commit, so a
            // bounded overflow cannot corrupt the receiver across unwinding.
            rhs.value.sub_assign(&self.value);
            if !rhs.value.abs.is_zero() {
                rhs.value.is_positive = !rhs.value.is_positive;
            }
            let result = Self {
                value: rhs.value,
                precision: self.precision,
            };
            result.assert_fits("subtraction");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        SubAssign::sub_assign(self, &rhs);
    }
}

impl SubAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn sub_assign(&mut self, rhs: &Self) {
        if !self.precision.is_unlimited() {
            let result = Self {
                value: self.value.sub(&rhs.value),
                precision: self.precision,
            };
            result.assert_fits("subtraction");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        self.value.sub_assign(&rhs.value);
        self.assert_fits("subtraction");
        self.debug_assert_valid();
    }
}

impl Neg for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn neg(self) -> Self::Output {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                !self.value.is_signed_min_for_width(bits),
                "MpInt neg overflow for Bounded({bits})"
            );
        }
        let value = if self.value.abs.is_zero() {
            InternalMpInt::zero()
        } else {
            InternalMpInt {
                abs: self.value.abs,
                is_positive: !self.value.is_positive,
            }
        };
        let result = Self {
            value,
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Neg for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn neg(self) -> Self::Output {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                !self.value.is_signed_min_for_width(bits),
                "MpInt neg overflow for Bounded({bits})"
            );
        }
        let value = if self.value.abs.is_zero() {
            InternalMpInt::zero()
        } else {
            InternalMpInt {
                abs: self.value.abs.clone(),
                is_positive: !self.value.is_positive,
            }
        };
        let result = MpInt {
            value,
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }
}
