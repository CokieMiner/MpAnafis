//! Multiplication operator trait implementations.

use core::ops::{Mul, MulAssign};

use super::{ArbiInt, ArbiUint};

impl Mul<Self> for ArbiUint {
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

impl Mul<&Self> for ArbiUint {
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

impl Mul<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn mul(self, mut rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.mul_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("multiplication");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Mul<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.mul(&rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl MulAssign<Self> for ArbiUint {
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

impl MulAssign<&Self> for ArbiUint {
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

impl Mul<Self> for ArbiInt {
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

impl Mul<&Self> for ArbiInt {
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

impl Mul<ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn mul(self, mut rhs: ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.mul_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("multiplication");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Mul<&ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn mul(self, rhs: &ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiInt {
            value: self.value.mul(&rhs.value),
            precision,
        };
        result.assert_fits("multiplication");
        result.debug_assert_valid();
        result
    }
}

impl MulAssign<Self> for ArbiInt {
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

impl MulAssign<&Self> for ArbiInt {
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
