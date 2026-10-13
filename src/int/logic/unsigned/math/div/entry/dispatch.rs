//! Assignment and reusable-output division dispatch.
//!
//! [`InternalMpUint`] mutates a magnitude through quotient and remainder operations.
//! [`Division`] selects the algorithm and writes caller-owned outputs using
//! [`DivScratch`] when normalization exceeds the bounded stack workspace.
//!
//! Scalar divisors, trivial quotients and powers of two precede quotient
//! truncation, Algorithm D, Burnikel-Ziegler and Newton. Operand geometry and
//! tuned thresholds select the general kernel after dispatch proves its bounds.

#![expect(
    unsafe_code,
    reason = "division dispatch proves normalized operand lengths before accessing scalar limbs and high suffixes"
)]

use core::{cmp::Ordering, mem::replace};

use super::{
    BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_THRESHOLD,
    CompleteDivisionMode, DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DIVISION_STACK_LIMBS, DivScratch,
    DivisionAttempt, InternalMpUint, Limb, NEWTON_RAPHSON_THRESHOLD, QuotientAndRemainder,
    QuotientOnly, RemainderOnly,
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
        if Division::try_algorithm_d_unscratched::<QuotientOnly, false>(
            &source, rhs, self, &mut rem,
        ) == DivisionAttempt::Complete
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
        if Division::power_of_two::<RemainderOnly>(&source, rhs, &mut dummy_quot, self) {
            return;
        }
        if Division::try_algorithm_d_unscratched::<RemainderOnly, false>(
            &source,
            rhs,
            &mut dummy_quot,
            self,
        ) == DivisionAttempt::Complete
        {
            return;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(&source, rhs, self, &mut scratch);
    }
}

impl Division {
    /// Attempts Algorithm D without constructing heap-backed division scratch.
    ///
    /// Returns `NeedsScratch` when the configured dispatch or normalization size
    /// requires the reusable scratch path. The divisor is nonzero. With
    /// `CHECK_TRIVIAL = false`, the numerator must have at least as many
    /// active limbs as the divisor; the quotient may still be zero or one.
    pub fn try_algorithm_d_unscratched<Mode: CompleteDivisionMode, const CHECK_TRIVIAL: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> DivisionAttempt {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if CHECK_TRIVIAL && Self::trivial::<Mode>(num_a, den_b, quotient_out, rem_out) {
            return DivisionAttempt::Complete;
        }
        if CHECK_TRIVIAL && Self::power_of_two::<Mode>(num_a, den_b, quotient_out, rem_out) {
            return DivisionAttempt::Complete;
        }
        let v_limbs = den_b.limbs();
        let u_limbs = num_a.limbs();
        // SAFETY: trivial division returned above, or CHECK_TRIVIAL=false
        // requires numerator width >= divisor width at the dispatch boundary.
        let extra = unsafe { u_limbs.len().unchecked_sub(v_limbs.len()) };
        let recursive_cutoff = if Mode::WRITE_REMAINDER {
            BURNIKEL_ZIEGLER_THRESHOLD
        } else if extra <= v_limbs.len() {
            BURNIKEL_QUOTIENT_THRESHOLD
        } else {
            BURNIKEL_LONG_QUOTIENT_THRESHOLD
        };
        if (v_limbs.len() >= recursive_cutoff && extra >= DIVISION_BASECASE_QUOTIENT_MAX_LIMBS)
            || (v_limbs.len() > 8 && u_limbs.len() >= DIVISION_STACK_LIMBS)
        {
            return DivisionAttempt::NeedsScratch;
        }
        Self::algorithm_d_stack::<Mode>(u_limbs, v_limbs, quotient_out, rem_out);
        DivisionAttempt::Complete
    }

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
        if Self::trivial::<QuotientAndRemainder>(num_a, den_b, quotient_out, rem_out) {
            return;
        }
        if Self::power_of_two::<QuotientAndRemainder>(num_a, den_b, quotient_out, rem_out) {
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
                Self::algorithm_d::<QuotientAndRemainder>(
                    u_limbs,
                    v_limbs,
                    quotient_out,
                    rem_out,
                    scratch,
                );
            }
            FullDivisionAlgorithm::Newton => {
                Self::newton::<QuotientAndRemainder>(num_a, den_b, quotient_out, rem_out, scratch);
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
        if !Self::trivial::<RemainderOnly>(num_a, den_b, &mut dummy_quot, rem_out) {
            if Self::power_of_two::<RemainderOnly>(num_a, den_b, &mut dummy_quot, rem_out) {
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
                        Self::algorithm_d::<RemainderOnly>(
                            u_limbs,
                            v_limbs,
                            &mut dummy_quot,
                            rem_out,
                            scratch,
                        );
                    }
                    FullDivisionAlgorithm::Newton => Self::newton::<RemainderOnly>(
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
    pub fn trivial<Mode: CompleteDivisionMode>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        let v_limbs = den_b.limbs();
        debug_assert!(!v_limbs.is_empty(), "division requires a non-zero divisor");
        let u_limbs = num_a.limbs();

        if u_limbs.len() < v_limbs.len() {
            if Mode::WRITE_QUOTIENT {
                quotient_out.clear();
            }
            if Mode::WRITE_REMAINDER {
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
                        if Mode::WRITE_QUOTIENT {
                            quotient_out.set_limb(Limb::from(subtract));
                        }
                        if Mode::WRITE_REMAINDER {
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
                    if Mode::WRITE_QUOTIENT {
                        quotient_out.clear();
                    }
                    if Mode::WRITE_REMAINDER {
                        rem_out.clone_from(num_a);
                    }
                    return true;
                }
                Ordering::Equal => {
                    if Mode::WRITE_QUOTIENT {
                        quotient_out.set_limb(1);
                    }
                    if Mode::WRITE_REMAINDER {
                        rem_out.clear();
                    }
                    return true;
                }
                Ordering::Greater => {
                    if Self::less_than_double(u_limbs, v_limbs) {
                        if Mode::WRITE_QUOTIENT {
                            quotient_out.set_limb(1);
                        }
                        if Mode::WRITE_REMAINDER {
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
