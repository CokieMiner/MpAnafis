//! Multiplication operator trait implementations.

use core::ops::{Mul, MulAssign};

use super::{MpInt, MpUint};

impl Mul<Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.mul_into(rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl Mul<&Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        self.value.mul_assign(&rhs.value);
        self.precision = precision;
        self.assert_fits("multiplication");
        self.debug_assert_valid();
        self
    }
}

impl Mul<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn mul(self, mut rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.mul_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("multiplication");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Mul<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.mul(&rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl MulAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn mul_assign(&mut self, mut rhs: Self) {
        if !self.precision.is_unlimited() {
            // Multiplication is commutative, so the owned right-hand buffer
            // can hold the candidate until its bounded fit is proven.
            rhs.value.mul_assign(&self.value);
            let result = Self {
                value: rhs.value,
                precision: self.precision,
            };
            result.assert_fits("multiplication");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        MulAssign::mul_assign(self, &rhs);
    }
}

impl MulAssign<&Self> for MpUint {
    #[inline]
    #[track_caller]
    fn mul_assign(&mut self, rhs: &Self) {
        if !self.precision.is_unlimited() {
            let result = Self {
                value: self.value.mul(&rhs.value),
                precision: self.precision,
            };
            result.assert_fits("multiplication");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        self.value.mul_assign(&rhs.value);
        self.assert_fits("multiplication");
        self.debug_assert_valid();
    }
}

impl Mul<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.mul_into(rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl Mul<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        self.value.mul_assign(&rhs.value);
        self.precision = precision;
        self.assert_fits("multiplication");
        self.debug_assert_valid();
        self
    }
}

impl Mul<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn mul(self, mut rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.mul_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("multiplication");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Mul<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: self.value.mul(&rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl MulAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn mul_assign(&mut self, mut rhs: Self) {
        if !self.precision.is_unlimited() {
            // The owned right-hand buffer makes bounded multiplication
            // transactional without forfeiting operand-storage reuse.
            rhs.value.mul_assign(&self.value);
            let result = Self {
                value: rhs.value,
                precision: self.precision,
            };
            result.assert_fits("multiplication");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        MulAssign::mul_assign(self, &rhs);
    }
}

impl MulAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn mul_assign(&mut self, rhs: &Self) {
        if !self.precision.is_unlimited() {
            let result = Self {
                value: self.value.mul(&rhs.value),
                precision: self.precision,
            };
            result.assert_fits("multiplication");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        self.value.mul_assign(&rhs.value);
        self.assert_fits("multiplication");
        self.debug_assert_valid();
    }
}
