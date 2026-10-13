//! Value-returning division for canonical magnitudes and nonzero divisors.
//!
//! Native quotients zero and one use a bound and a comparison.
//! Combined outputs for powers of two use a shift and low-bit mask. Other
//! native pairs use one division. Wider operands enter the scalar kernel
//! or the division tower.

#![expect(
    unsafe_code,
    reason = "canonical scalar operands establish positive divisors, exact differences and sub-limb shifts"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Rem},
};

use super::{
    ArchKernels, DivScratch, Division, DivisionAttempt, InternalMpUint, QuotientAndRemainder,
    QuotientOnly, RemainderOnly,
};

impl InternalMpUint {
    /// Computes `ceil(self / rhs)` for a caller-validated nonzero divisor.
    #[inline]
    #[must_use]
    pub fn div_ceil(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal ceiling division requires a nonzero divisor"
        );
        let (mut quotient, remainder) = self.div_rem(rhs);
        if !remainder.is_zero() {
            quotient.increment();
        }
        quotient
    }

    /// Computes quotient and remainder of `self / rhs`.
    ///
    /// `rhs` must be nonzero.
    #[must_use]
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a nonzero divisor"
        );
        let u_limbs = self.limbs();
        let v_limbs = rhs.limbs();
        if u_limbs.len() < v_limbs.len() {
            return (Self::zero(), self.clone());
        }
        if let [divisor] = v_limbs {
            if let [value] = u_limbs {
                if value < divisor {
                    return (Self::zero(), Self::from_limb(*value));
                }
                // SAFETY: value >= divisor makes the difference exact.
                let difference = unsafe { value.unchecked_sub(*divisor) };
                if difference < *divisor {
                    return (Self::one(), Self::from_limb(difference));
                }
                // SAFETY: the canonical scalar divisor is positive.
                let mask = unsafe { divisor.unchecked_sub(1) };
                if divisor & mask == 0 {
                    // SAFETY: a positive power of two has a sub-limb shift.
                    // difference >= divisor proves value >= 2*divisor,
                    // so the shifted quotient is at least two.
                    let quotient = unsafe {
                        NonZeroUsize::new_unchecked(value.unchecked_shr(divisor.trailing_zeros()))
                    };
                    return (
                        Self::from_limb(quotient.get()),
                        Self::from_limb(*value & mask),
                    );
                }
                // SAFETY: a canonical scalar divisor is positive. A zero high
                // limb is below it, so the quotient fits one native limb.
                let (quotient, remainder) =
                    unsafe { ArchKernels::divrem_1_unchecked(*value, 0, *divisor) };
                // SAFETY: difference >= divisor proves value >= 2*divisor;
                // the exact architecture kernel therefore returns quotient >= 2.
                let positive_quotient = unsafe { NonZeroUsize::new_unchecked(quotient) };
                return (
                    Self::from_limb(positive_quotient.get()),
                    Self::from_limb(remainder),
                );
            }
            let mut quotient = Self::zero();
            let remainder = Division::div_rem_1::<true>(u_limbs, *divisor, &mut quotient);
            return (quotient, Self::from_limb(remainder));
        }

        let mut quotient = Self::zero();
        let mut remainder = Self::zero();
        if Division::power_of_two::<QuotientAndRemainder>(self, rhs, &mut quotient, &mut remainder)
        {
            return (quotient, remainder);
        }
        if u_limbs.len() == v_limbs.len()
            && Division::small_quotient_div_rem(self, rhs, &mut quotient, &mut remainder)
        {
            return (quotient, remainder);
        }
        if Division::try_algorithm_d_unscratched::<QuotientAndRemainder, false>(
            self,
            rhs,
            &mut quotient,
            &mut remainder,
        ) == DivisionAttempt::Complete
        {
            return (quotient, remainder);
        }
        let mut scratch = DivScratch::default();
        Division::div_rem_into(self, rhs, &mut quotient, &mut remainder, &mut scratch);
        (quotient, remainder)
    }

    /// Computes only the quotient of `self / rhs`.
    ///
    /// `rhs` must be nonzero. Operand truncation and triangular updates omit
    /// residue products that cannot affect the quotient.
    #[must_use]
    pub fn div(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a nonzero divisor"
        );
        if let [divisor] = rhs.limbs() {
            if *divisor == 1 {
                return self.clone();
            }
            let numerator = self.limbs();
            if numerator.len() <= 1 {
                let value = numerator.first().copied().unwrap_or(0);
                // SAFETY: a canonical scalar divisor is positive.
                let positive_divisor = unsafe { NonZeroUsize::new_unchecked(*divisor) };
                return Self::div_scalar(value, positive_divisor);
            }
            let mut quotient = Self::zero();
            let _ = Division::div_rem_1::<true>(numerator, *divisor, &mut quotient);
            return quotient;
        }
        let mut quotient = Self::zero();
        if Division::truncated_quotient::<false>(self, rhs, &mut quotient) {
            return quotient;
        }
        let mut unused_remainder = Self::zero();
        if Division::try_algorithm_d_unscratched::<QuotientOnly, false>(
            self,
            rhs,
            &mut quotient,
            &mut unused_remainder,
        ) == DivisionAttempt::Complete
        {
            return quotient;
        }
        let mut scratch = DivScratch::default();
        Division::div_into::<false, false>(self, rhs, &mut quotient, &mut scratch);
        quotient
    }

    /// Computes only the remainder of `self % rhs`.
    ///
    /// `rhs` must be nonzero.
    #[must_use]
    pub fn rem(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a nonzero divisor"
        );
        if let [divisor] = rhs.limbs() {
            if *divisor == 1 {
                return Self::zero();
            }
            let numerator = self.limbs();
            if numerator.len() <= 1 {
                let value = numerator.first().copied().unwrap_or(0);
                if value < *divisor {
                    return Self::from_limb(value);
                }
                // SAFETY: value >= divisor makes the difference exact.
                let difference = unsafe { value.unchecked_sub(*divisor) };
                if difference < *divisor {
                    return Self::from_limb(difference);
                }
                // SAFETY: a canonical scalar divisor is positive.
                let positive_divisor = unsafe { NonZeroUsize::new_unchecked(*divisor) };
                return Self::from_limb(Rem::rem(value, positive_divisor));
            }
            let mut unused_quotient = Self::zero();
            let remainder = Division::div_rem_1::<false>(numerator, *divisor, &mut unused_quotient);
            return Self::from_limb(remainder);
        }
        let mut unused_quotient = Self::zero();
        let mut remainder = Self::zero();
        if Division::try_algorithm_d_unscratched::<RemainderOnly, true>(
            self,
            rhs,
            &mut unused_quotient,
            &mut remainder,
        ) == DivisionAttempt::Complete
        {
            return remainder;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(self, rhs, &mut remainder, &mut scratch);
        remainder
    }

    /// Computes a native-limb quotient with a positive divisor.
    ///
    /// `floor(numerator/2) < divisor` is equivalent to `numerator < 2*divisor`
    /// without doubling the divisor. Under this bound the quotient is zero
    /// or one, determined by `numerator >= divisor`.
    #[inline]
    #[must_use]
    pub fn div_scalar(numerator: usize, divisor: NonZeroUsize) -> Self {
        if numerator >> 1 < divisor.get() {
            return Self::from_limb(usize::from(numerator >= divisor.get()));
        }
        let quotient = Div::div(numerator, divisor);
        // SAFETY: floor(numerator/2) >= divisor proves numerator >= 2*divisor,
        // so the exact quotient is at least two and therefore nonzero.
        let positive_quotient = unsafe { NonZeroUsize::new_unchecked(quotient) };
        Self::from_limb(positive_quotient.get())
    }
}
