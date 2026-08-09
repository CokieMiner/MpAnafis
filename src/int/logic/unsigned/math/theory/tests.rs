//! Properties for number-theoretic operations.

#![allow(
    clippy::unwrap_used,
    reason = "unwraps are acceptable in test fixtures"
)]

use core::cmp::Ordering;

use proptest::prelude::*;

use super::*;

proptest! {
        #[allow(
            clippy::arithmetic_side_effects,
            clippy::cast_possible_truncation,
            clippy::as_conversions,
            reason = "proptest for number theory properties"
        )]
        #[test]
        fn theory_prop(
            a_limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
            b_limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
        ) {
            let a = InternalMpUint::from_limbs(a_limbs);
            let b = InternalMpUint::from_limbs(b_limbs);

            // jacobi_symbol returns None for even/zero n, Some for odd n
            if !b.is_zero() && b.is_odd() {
                prop_assert!(matches!(a.jacobi_symbol(&b), -1..=1));
            }

            // abs_diff symmetry: |a - b| == |b - a|
            let mut out_ab = InternalMpUint::zero();
            let mut out_ba = InternalMpUint::zero();
            compute_abs_diff(&a, &b, &mut out_ab);
            compute_abs_diff(&b, &a, &mut out_ba);
            prop_assert_eq!(out_ab, out_ba);

            // abs_diff with self equals zero
            let mut out_self = InternalMpUint::zero();
            compute_abs_diff(&a, &a, &mut out_self);
            prop_assert!(out_self.is_zero());
        }

}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// `euler_phi` is kept to small inputs on purpose: it factors its
    /// argument, so a random four-limb operand costs a 256-bit
    /// factorization. The previous shape paid exactly that and then
    /// asserted only `is_some()`, which is why this property ran for
    /// minutes while establishing nothing. Counting the coprime residues
    /// directly checks the actual definition, and only stays affordable
    /// because the operand is bounded.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest counter is bounded by the 2000-element loop above it"
    )]
    #[test]
    fn euler_phi_prop(value in 0_u16..=2_000) {
        let n = InternalMpUint::from_limb(Limb::from(value));
        let Some(phi) = n.euler_phi() else {
            prop_assert!(n.is_zero(), "euler_phi is only undefined at zero");
            return Ok(());
        };
        prop_assert!(!n.is_zero());

        if n.is_one() {
            prop_assert!(phi.is_one());
            return Ok(());
        }
        // For every n > 1, `0 < phi(n) < n`, and phi(n) is by definition the
        // number of residues in `1..n` coprime to n.
        prop_assert!(!phi.is_zero());
        prop_assert!(phi.cmp(&n) == Ordering::Less);

        let mut coprime_count: Limb = 0;
        for candidate in 1..value {
            if InternalMpUint::from_limb(Limb::from(candidate)).gcd(&n).is_one() {
                coprime_count = coprime_count.wrapping_add(1);
            }
        }
        prop_assert_eq!(phi, InternalMpUint::from_limb(coprime_count));
    }
}
