//! Cross-family properties for unsigned arithmetic.

#![allow(
    clippy::unwrap_used,
    reason = "unwraps are acceptable in test fixtures"
)]

use proptest::prelude::*;

use super::InternalMpUint;
use crate::int::types::Limb;

proptest! {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "proptest for modular arithmetic"
        )]
        #[test]
        fn modular_prop(
            a_limbs in proptest::collection::vec(any::<Limb>(), 0..=3),
            b_limbs in proptest::collection::vec(any::<Limb>(), 0..=3),
            m_limbs in proptest::collection::vec(any::<Limb>(), 0..=3)
                .prop_filter("modulus must be non-zero", |limbs| limbs.iter().any(|&limb| limb != 0)),
        ) {
            let a = InternalMpUint::from_limbs(a_limbs);
            let b = InternalMpUint::from_limbs(b_limbs);
            let m = InternalMpUint::from_limbs(m_limbs);

            // add_mod
            let add_result = a.add_mod(&b, &m);
            prop_assert!(add_result < m);
            // commutativity
            prop_assert_eq!(add_result, b.add_mod(&a, &m));

            // sub_mod
            let sub_result = a.sub_mod(&b, &m);
            prop_assert!(sub_result < m);
            // self-subtraction yields zero
            let self_sub = a.sub_mod(&a, &m);
            prop_assert!(self_sub.is_zero());

            // mul_mod
            let mul_result = a.mul_mod(&b, &m);
            prop_assert!(mul_result < m);
            // commutativity
            prop_assert_eq!(mul_result, b.mul_mod(&a, &m));

            // pow_mod
            let pow_result = a.pow_mod(&b, &m);
            if m.is_one() {
                // Modulus 1: result is always 0
                prop_assert!(pow_result.is_zero());
            } else {
                prop_assert!(pow_result < m);
                // exponent 0 => result = 1 % m = 1 (for m > 1)
                let zero_exp = InternalMpUint::zero();
                let pow_zero = a.pow_mod(&zero_exp, &m);
                prop_assert!(pow_zero.is_one());
            }

            // invert: exists iff gcd(a, m) == 1
            let g = a.gcd(&m);
            let inv_exists = g.is_one();
            prop_assert_eq!(a.invert(&m).is_some(), inv_exists);
            if inv_exists {
                let inv = a.invert(&m).expect("invert should succeed");
                prop_assert!(inv < m);
                // Verify: a * inv === 1 (mod m).
                let prod = a.mul_mod(&inv, &m);
                prop_assert_eq!(prod, InternalMpUint::one().rem(&m));
            }
        }
}
