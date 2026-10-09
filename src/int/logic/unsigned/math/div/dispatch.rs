//! Value, assignment, and reusable-output division dispatch.
//!
//! [`InternalMpUint`] provides quotient, remainder, and rounding operations.
//! [`Division`] selects the algorithm and writes caller-owned outputs using
//! [`DivScratch`] when normalization exceeds the bounded stack workspace.
//!
//! Trivial quotients and powers of two precede quotient truncation and the
//! Algorithm D, Burnikel-Ziegler, and Newton algorithms. Operand geometry and
//! generated thresholds determine admission; kernels receive proved bounds.

#![expect(
    unsafe_code,
    reason = "division dispatch proves normalized operand lengths before accessing scalar limbs and high suffixes"
)]

use core::{cmp::Ordering, mem::replace};

use super::{
    BURNIKEL_ZIEGLER_THRESHOLD, DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DivScratch, InternalMpUint,
    Limb, NEWTON_RAPHSON_THRESHOLD,
};

/// Zero-sized namespace for division dispatch and arithmetic kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Division;

/// One policy for full quotient/remainder and remainder-only division.
#[derive(Clone, Copy)]
enum FullDivisionAlgorithm {
    AlgorithmD,
    Burnikel,
    Newton,
}

impl InternalMpUint {
    /// Computes quotient and remainder of `self / rhs`.
    ///
    /// `rhs` must be non-zero.
    #[must_use]
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let u_limbs = self.limbs();
        let v_limbs = rhs.limbs();
        if u_limbs.len() < v_limbs.len() {
            return (Self::zero(), self.clone());
        }
        if v_limbs.len() == 1 {
            let mut q = Self::zero();
            // SAFETY: `v_limbs.len() == 1` proves index 0 is valid.
            let v0 = unsafe { *v_limbs.get_unchecked(0) };
            let rem = Division::div_rem_1::<true>(u_limbs, v0, &mut q);
            return (
                q,
                if rem == 0 {
                    Self::zero()
                } else {
                    Self::from_limb(rem)
                },
            );
        }

        let mut q = Self::zero();
        let mut r = Self::zero();
        if Division::power_of_two::<true, true>(self, rhs, &mut q, &mut r) {
            return (q, r);
        }
        if u_limbs.len() == v_limbs.len()
            && Division::small_quotient_div_rem(self, rhs, &mut q, &mut r)
        {
            return (q, r);
        }
        if Division::try_algorithm_d_unscratched::<true, true, false>(self, rhs, &mut q, &mut r) {
            return (q, r);
        }
        let mut scratch = DivScratch::default();
        Division::div_rem_into(self, rhs, &mut q, &mut r, &mut scratch);
        (q, r)
    }

    /// Computes `ceil(self / rhs)` for a caller-validated non-zero divisor.
    #[inline]
    #[must_use]
    pub fn div_ceil(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal ceiling division requires a non-zero divisor"
        );
        let (mut quotient, remainder) = self.div_rem(rhs);
        if !remainder.is_zero() {
            quotient.increment();
        }
        quotient
    }

    /// Computes only the quotient of `self / rhs`.
    ///
    /// `rhs` must be nonzero. Operand truncation and triangular updates omit
    /// residue products that cannot affect the quotient.
    #[must_use]
    pub fn div(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if rhs.is_one() {
            return self.clone();
        }
        let mut q = Self::zero();
        if Division::truncated_quotient::<false>(self, rhs, &mut q) {
            return q;
        }
        let mut rem = Self::zero();
        if Division::try_algorithm_d_unscratched::<true, false, false>(self, rhs, &mut q, &mut rem)
        {
            return q;
        }
        let mut scratch = DivScratch::default();
        Division::div_into::<false, false>(self, rhs, &mut q, &mut scratch);
        q
    }

    /// Computes only the remainder of `self % rhs`.
    ///
    /// `rhs` must be non-zero.
    #[must_use]
    pub fn rem(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if rhs.is_one() {
            return Self::zero();
        }
        let mut dummy_quot = Self::zero();
        let mut r = Self::zero();
        if Division::try_algorithm_d_unscratched::<false, true, true>(
            self,
            rhs,
            &mut dummy_quot,
            &mut r,
        ) {
            return r;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(self, rhs, &mut r, &mut scratch);
        r
    }

    /// Replaces `self` with the quotient of `self / rhs`.
    ///
    /// `rhs` must be non-zero.
    pub fn div_assign(&mut self, rhs: &Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if rhs.is_one() {
            return;
        }
        match (*self).cmp(rhs) {
            Ordering::Less => {
                self.clear();
                return;
            }
            Ordering::Equal => {
                self.clone_from(&Self::one());
                return;
            }
            Ordering::Greater => {}
        }
        if let [divisor] = rhs.limbs() {
            let _ = Division::div_rem_1_assign(self, *divisor);
            return;
        }
        let source = replace(self, Self::zero());
        if Division::truncated_quotient::<false>(&source, rhs, self) {
            return;
        }
        let mut rem = Self::zero();
        if Division::try_algorithm_d_unscratched::<true, false, false>(&source, rhs, self, &mut rem)
        {
            return;
        }
        let mut scratch = DivScratch::default();
        Division::div_into::<false, false>(&source, rhs, self, &mut scratch);
    }

    /// Replaces `self` with the remainder of `self % rhs`.
    ///
    /// `rhs` must be non-zero.
    pub fn rem_assign(&mut self, rhs: &Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if rhs.is_one() {
            self.clear();
            return;
        }
        match (*self).cmp(rhs) {
            Ordering::Less => return,
            Ordering::Equal => {
                self.clear();
                return;
            }
            Ordering::Greater => {}
        }
        let mut dummy_quot = Self::zero();
        if let [divisor] = rhs.limbs() {
            let rem = Division::div_rem_1::<false>(self.limbs(), *divisor, &mut dummy_quot);
            self.clone_from_slice(&[rem]);
            return;
        }
        let source = replace(self, Self::zero());
        if Division::power_of_two::<false, true>(&source, rhs, &mut dummy_quot, self) {
            return;
        }
        if Division::try_algorithm_d_unscratched::<false, true, false>(
            &source,
            rhs,
            &mut dummy_quot,
            self,
        ) {
            return;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(&source, rhs, self, &mut scratch);
    }
}

impl Division {
    /// Computes both halves of `num_a / den_b` into caller-owned outputs,
    /// choosing the divider from the divisor length.
    pub fn div_rem_into(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if Self::trivial::<true, true>(num_a, den_b, quotient_out, rem_out) {
            return;
        }
        if Self::power_of_two::<true, true>(num_a, den_b, quotient_out, rem_out) {
            return;
        }
        let v_limbs = den_b.limbs();
        let u_limbs = num_a.limbs();

        if v_limbs.len() == 1 {
            // SAFETY: v_limbs.len() == 1 so index 0 is valid.
            let v0 = unsafe { *v_limbs.get_unchecked(0) };
            let rem = Self::div_rem_1::<true>(u_limbs, v0, quotient_out);
            *rem_out = InternalMpUint::from_limb(rem);
            return;
        }

        // SAFETY: rejecting the trivial cases proves num_a > den_b, hence
        // the canonical numerator has at least the divisor's limb count.
        let extra = unsafe { u_limbs.len().unchecked_sub(v_limbs.len()) };
        match full_output_algorithm(v_limbs.len(), extra) {
            FullDivisionAlgorithm::AlgorithmD => {
                let _ = Self::algorithm_d::<true, true, false, false>(
                    u_limbs,
                    v_limbs,
                    quotient_out,
                    rem_out,
                    scratch,
                );
            }
            FullDivisionAlgorithm::Newton => {
                Self::newton::<true, true, false>(num_a, den_b, quotient_out, rem_out, scratch);
            }
            FullDivisionAlgorithm::Burnikel => {
                Self::burnikel_ziegler::<true>(num_a, den_b, quotient_out, rem_out, scratch);
            }
        }
    }

    /// Computes only `num_a % den_b` using reusable division scratch.
    pub fn rem_into(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let mut dummy_quot = InternalMpUint::zero();
        if !Self::trivial::<false, true>(num_a, den_b, &mut dummy_quot, rem_out) {
            if Self::power_of_two::<false, true>(num_a, den_b, &mut dummy_quot, rem_out) {
                return;
            }
            let v_limbs = den_b.limbs();
            let u_limbs = num_a.limbs();
            // SAFETY: rejecting the trivial cases proves num_a > den_b,
            // so the canonical numerator is at least as wide as the divisor.
            let extra = unsafe { u_limbs.len().unchecked_sub(v_limbs.len()) };
            if v_limbs.len() == 1 {
                // SAFETY: v_limbs.len() == 1 so index 0 is valid.
                let v0 = unsafe { *v_limbs.get_unchecked(0) };
                let rem = Self::div_rem_1::<false>(u_limbs, v0, &mut dummy_quot);
                *rem_out = InternalMpUint::from_limb(rem);
            } else {
                match full_output_algorithm(v_limbs.len(), extra) {
                    FullDivisionAlgorithm::AlgorithmD => {
                        let _ = Self::algorithm_d::<false, true, false, false>(
                            u_limbs,
                            v_limbs,
                            &mut dummy_quot,
                            rem_out,
                            scratch,
                        );
                    }
                    FullDivisionAlgorithm::Newton => Self::newton::<false, true, false>(
                        num_a,
                        den_b,
                        &mut dummy_quot,
                        rem_out,
                        scratch,
                    ),
                    FullDivisionAlgorithm::Burnikel => {
                        // Burnikel writes a temporary quotient. Newton consumes
                        // its estimate in the product buffer, so only Burnikel
                        // transfers this reusable quotient-output storage.
                        dummy_quot = replace(&mut scratch.dummy_quot, InternalMpUint::zero());
                        Self::burnikel_ziegler::<true>(
                            num_a,
                            den_b,
                            &mut dummy_quot,
                            rem_out,
                            scratch,
                        );
                        scratch.dummy_quot = dummy_quot;
                    }
                }
            }
        }
    }

    /// Resolves quotients that are provably `0` or `1`.
    ///
    /// Returns `false` when the operands need the tower.
    pub fn trivial<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        let v_limbs = den_b.limbs();
        debug_assert!(!v_limbs.is_empty(), "division requires a non-zero divisor");
        let u_limbs = num_a.limbs();

        if u_limbs.len() < v_limbs.len() {
            if WRITE_QUOTIENT {
                quotient_out.clear();
            }
            if WRITE_REMAINDER {
                rem_out.clone_from(num_a);
            }
            return true;
        }

        if u_limbs.len() == v_limbs.len() {
            let comparison = if v_limbs.len() > 1 {
                // SAFETY: both equal-length initialized slices have at least
                // two limbs, so their low limbs and high suffixes exist.
                unsafe {
                    let high = InternalMpUint::cmp_limbs(
                        u_limbs.get_unchecked(1..),
                        v_limbs.get_unchecked(1..),
                    );
                    if high == Ordering::Equal {
                        let low_u = *u_limbs.get_unchecked(0);
                        let low_v = *v_limbs.get_unchecked(0);
                        let subtract = low_u >= low_v;
                        if WRITE_QUOTIENT {
                            quotient_out.set_limb(Limb::from(subtract));
                        }
                        if WRITE_REMAINDER {
                            if subtract {
                                // Equal high suffixes and low_u >= low_v
                                // make this the complete nonnegative residue.
                                *rem_out = InternalMpUint::from_limb(low_u.unchecked_sub(low_v));
                            } else {
                                rem_out.clone_from(num_a);
                            }
                        }
                        return true;
                    }
                    high
                }
            } else {
                InternalMpUint::cmp_limbs(u_limbs, v_limbs)
            };
            match comparison {
                Ordering::Less => {
                    if WRITE_QUOTIENT {
                        quotient_out.clear();
                    }
                    if WRITE_REMAINDER {
                        rem_out.clone_from(num_a);
                    }
                    return true;
                }
                Ordering::Equal => {
                    if WRITE_QUOTIENT {
                        quotient_out.set_limb(1);
                    }
                    if WRITE_REMAINDER {
                        rem_out.clear();
                    }
                    return true;
                }
                Ordering::Greater => {
                    if Self::less_than_double(u_limbs, v_limbs) {
                        if WRITE_QUOTIENT {
                            quotient_out.set_limb(1);
                        }
                        if WRITE_REMAINDER {
                            let underflowed = rem_out.assign_difference(num_a, den_b);
                            debug_assert!(
                                !underflowed,
                                "num_a > den_b guaranteed by Ordering::Greater"
                            );
                        }
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// Selects the full-output tier after scalar and trivial cases are excluded.
const fn full_output_algorithm(divisor_len: usize, extra: usize) -> FullDivisionAlgorithm {
    if extra < DIVISION_BASECASE_QUOTIENT_MAX_LIMBS {
        FullDivisionAlgorithm::AlgorithmD
    } else if divisor_len >= NEWTON_RAPHSON_THRESHOLD {
        FullDivisionAlgorithm::Newton
    } else if divisor_len >= BURNIKEL_ZIEGLER_THRESHOLD {
        FullDivisionAlgorithm::Burnikel
    } else {
        FullDivisionAlgorithm::AlgorithmD
    }
}
