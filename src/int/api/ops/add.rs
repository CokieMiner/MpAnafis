//! Addition operator trait implementations.

use core::ops::{Add, AddAssign};

use super::{ArbiInt, ArbiUint};

impl Add<Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let self_len = self.value.limbs().len();
        let rhs_len = rhs.value.limbs().len();
        let reuse_rhs = rhs_len > self_len
            || (rhs_len == self_len && rhs.value.capacity() > self.value.capacity());
        // For unequal lengths, keeping the longer magnitude avoids copying its
        // tail. At equal lengths both kernels do identical work, so retaining
        // the larger-capacity buffer weakly minimizes future growth.
        if reuse_rhs {
            let mut result_val = rhs.value;
            result_val.add_assign(&self.value);
            self.value = result_val;
        } else {
            self.value.add_assign(&rhs.value);
        }
        self.precision = precision;
        self.assert_fits("addition");
        self.debug_assert_valid();
        self
    }
}

impl Add<&Self> for ArbiUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        self.value.add_assign(&rhs.value);
        self.precision = precision;
        self.assert_fits("addition");
        self.debug_assert_valid();
        self
    }
}

impl Add<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn add(self, mut rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.add_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("addition");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Add<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn add(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiUint {
            value: self.value.add(&rhs.value),
            precision,
        };
        result.assert_fits("addition");
        result.debug_assert_valid();
        result
    }
}

impl AddAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, mut rhs: Self) {
        if !self.precision.is_unlimited() {
            // Bounded overflow is a panic contract, so the receiver cannot be
            // used as scratch: a caught panic must not expose an out-of-range
            // value. Addition is commutative, which lets the owned right-hand
            // buffer hold the exact candidate until validation succeeds.
            rhs.value.add_assign(&self.value);
            let result = Self {
                value: rhs.value,
                precision: self.precision,
            };
            result.assert_fits("addition");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        let self_len = self.value.limbs().len();
        let rhs_len = rhs.value.limbs().len();
        let self_capacity = self.value.capacity();
        let reuse_rhs = self_capacity < rhs_len
            || (self_len == rhs_len && rhs.value.capacity() > self_capacity);
        // Addition is commutative. If the left buffer cannot hold the right
        // magnitude, computing `rhs + self` in the owned right buffer gives
        // the same exact sum while eliminating the otherwise necessary grow.
        // For equal lengths, preferring a strictly larger right capacity does
        // the same limb work and also avoids a carry-induced reallocation.
        if reuse_rhs {
            rhs.value.add_assign(&self.value);
            self.value = rhs.value;
        } else {
            self.value.add_assign(&rhs.value);
        }
        self.assert_fits("addition");
        self.debug_assert_valid();
    }
}

impl AddAssign<&Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, rhs: &Self) {
        if !self.precision.is_unlimited() {
            let result = Self {
                value: self.value.add(&rhs.value),
                precision: self.precision,
            };
            result.assert_fits("addition");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        self.value.add_assign(&rhs.value);
        self.assert_fits("addition");
        self.debug_assert_valid();
    }
}

// ---- ArbiInt ----

impl Add<Self> for ArbiInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let self_len = self.value.abs.limbs().len();
        let rhs_len = rhs.value.abs.limbs().len();
        let reuse_rhs = rhs_len > self_len
            || (rhs_len == self_len
                && self.value.is_positive == rhs.value.is_positive
                && rhs.value.abs.capacity() > self.value.abs.capacity());
        // The longer magnitude avoids tail copying. At equal lengths, capacity
        // breaks ties only for equal signs: opposite signs use subtraction,
        // where swapping may add a full negation after underflow.
        if reuse_rhs {
            let mut result_val = rhs.value;
            result_val.add_assign(&self.value);
            self.value = result_val;
        } else {
            self.value.add_assign(&rhs.value);
        }
        self.precision = precision;
        self.assert_fits("addition");
        self.debug_assert_valid();
        self
    }
}

impl Add<&Self> for ArbiInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        self.value.add_assign(&rhs.value);
        self.precision = precision;
        self.assert_fits("addition");
        self.debug_assert_valid();
        self
    }
}

impl Add<ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn add(self, mut rhs: ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.add_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("addition");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Add<&ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn add(self, rhs: &ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = ArbiInt {
            value: self.value.add(&rhs.value),
            precision,
        };
        result.assert_fits("addition");
        result.debug_assert_valid();
        result
    }
}

impl AddAssign<Self> for ArbiInt {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, mut rhs: Self) {
        if !self.precision.is_unlimited() {
            // Signed addition is commutative too, so the owned right-hand
            // buffer provides transactional storage for the bounded result.
            rhs.value.add_assign(&self.value);
            let result = Self {
                value: rhs.value,
                precision: self.precision,
            };
            result.assert_fits("addition");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        let self_len = self.value.abs.limbs().len();
        let rhs_len = rhs.value.abs.limbs().len();
        let self_capacity = self.value.abs.capacity();
        let reuse_rhs = self_capacity < rhs_len
            || (self_len == rhs_len
                && self.value.is_positive == rhs.value.is_positive
                && rhs.value.abs.capacity() > self_capacity);
        // Signed addition is also commutative. Reusing the owned right
        // magnitude is therefore exact for every sign combination and avoids
        // growing the left buffer when its capacity is provably insufficient.
        // For equal signs, equal-length addition does identical limb work in
        // either buffer, so the larger capacity is allocation-optimal. With
        // opposite signs, retaining the left side avoids a possible negation.
        if reuse_rhs {
            rhs.value.add_assign(&self.value);
            self.value = rhs.value;
        } else {
            self.value.add_assign(&rhs.value);
        }
        self.assert_fits("addition");
        self.debug_assert_valid();
    }
}

impl AddAssign<&Self> for ArbiInt {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, rhs: &Self) {
        if !self.precision.is_unlimited() {
            let result = Self {
                value: self.value.add(&rhs.value),
                precision: self.precision,
            };
            result.assert_fits("addition");
            result.debug_assert_valid();
            *self = result;
            return;
        }

        self.value.add_assign(&rhs.value);
        self.assert_fits("addition");
        self.debug_assert_valid();
    }
}
