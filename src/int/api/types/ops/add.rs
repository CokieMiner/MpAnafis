//! Addition operator trait implementations.

use core::ops::{Add, AddAssign};

use super::{MpInt, MpUint};

impl Add<Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let self_len = self.value.limbs().len();
        let rhs_len = rhs.value.limbs().len();
        let reuse_rhs = rhs_len > self_len
            || (rhs_len == self_len
                && self_len > 4
                && rhs.value.capacity() > self.value.capacity());
        // The longer magnitude supplies the unchanged tail. Equal lengths
        // perform the same limb work; greater heap capacity admits a carry.
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

impl Add<&Self> for MpUint {
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

impl Add<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn add(self, mut rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.add_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("addition");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Add<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn add(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpUint {
            value: self.value.add(&rhs.value),
            precision,
        };
        result.assert_fits("addition");
        result.debug_assert_valid();
        result
    }
}

impl AddAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, mut rhs: Self) {
        if let Some(bits) = self.precision.significant_bits()
            && self
                .value
                .significant_bits()
                .max(rhs.value.significant_bits())
                >= bits
        {
            // Commutativity permits computing the candidate in rhs.
            // Validate its width before replacing the receiver.
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
        let reuse_rhs = if self_len > 4 || rhs_len > 4 {
            let self_capacity = self.value.capacity();
            self_capacity < rhs_len || (self_len == rhs_len && rhs.value.capacity() > self_capacity)
        } else {
            false
        };
        // Reuse rhs when the receiver cannot hold its magnitude. At equal
        // lengths, greater rhs capacity also admits a carry without growth.
        if reuse_rhs {
            rhs.value.add_assign(&self.value);
            self.value = rhs.value;
        } else {
            self.value.add_assign(&rhs.value);
        }
        // For bounded precision, both operands are below 2^(bits-1), so
        // their sum is at most 2^bits-2 and fits before receiver mutation.
        self.debug_assert_valid();
    }
}

impl AddAssign<&Self> for MpUint {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, rhs: &Self) {
        if let Some(bits) = self.precision.significant_bits()
            && self
                .value
                .significant_bits()
                .max(rhs.value.significant_bits())
                >= bits
        {
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
        // The bounded path proves both significant widths are below bits;
        // adding the two values therefore cannot reach 2^bits.
        self.debug_assert_valid();
    }
}

impl Add<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn add(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let self_len = self.value.abs.limbs().len();
        let rhs_len = rhs.value.abs.limbs().len();
        let reuse_rhs = rhs_len > self_len
            || (rhs_len == self_len
                && self_len > 4
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

impl Add<&Self> for MpInt {
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

impl Add<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn add(self, mut rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        rhs.value.add_assign(&self.value);
        rhs.precision = precision;
        rhs.assert_fits("addition");
        rhs.debug_assert_valid();
        rhs
    }
}

impl Add<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn add(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = MpInt {
            value: self.value.add(&rhs.value),
            precision,
        };
        result.assert_fits("addition");
        result.debug_assert_valid();
        result
    }
}

impl AddAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, mut rhs: Self) {
        if let Some(bits) = self.precision.significant_bits()
            && !self.value.sum_fits_by_width(&rhs.value, bits)
        {
            // Signed addition is commutative; the owned right-hand
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
        let reuse_rhs = if self_len > 4 || rhs_len > 4 {
            let self_capacity = self.value.abs.capacity();
            self_capacity < rhs_len
                || (self_len == rhs_len
                    && self.value.is_positive == rhs.value.is_positive
                    && rhs.value.abs.capacity() > self_capacity)
        } else {
            false
        };
        // Addition permits buffer exchange for every sign combination.
        // Equal signs and lengths perform the same limb work; opposite signs
        // retain the receiver to avoid a subtraction sign adjustment.
        if reuse_rhs {
            rhs.value.add_assign(&self.value);
            self.value = rhs.value;
        } else {
            self.value.add_assign(&rhs.value);
        }
        // Unlimited precision or the operand-width proof permits mutation.
        self.debug_assert_valid();
    }
}

impl AddAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, rhs: &Self) {
        if let Some(bits) = self.precision.significant_bits()
            && !self.value.sum_fits_by_width(&rhs.value, bits)
        {
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
        // Unlimited precision or the operand-width proof permits mutation.
        self.debug_assert_valid();
    }
}
