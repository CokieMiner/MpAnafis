//! Properties for the GCD family.

use proptest::prelude::*;

use super::*;

proptest! {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "proptest for gcd properties"
        )]
        #[test]
        fn gcd_prop(
            a_limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
            b_limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
        ) {
            let a = InternalMpUint::from_limbs(a_limbs);
            let b = InternalMpUint::from_limbs(b_limbs);

            let g = a.gcd(&b);

            // gcd divides both inputs
            if !g.is_zero() {
                let r_a = a.rem(&g);
                prop_assert!(r_a.is_zero());
                let r_b = b.rem(&g);
                prop_assert!(r_b.is_zero());
            }

            // Commutativity
            prop_assert_eq!(&g, &b.gcd(&a));

            // is_coprime matches gcd == 1
            prop_assert_eq!(a.is_coprime(&b), g.is_one());

            // lcm property: lcm(a,b) * gcd(a,b) == a * b
            let l = a.lcm(&b);
            let lhs = l.mul(&g);
            let rhs = a.mul(&b);
            prop_assert_eq!(lhs, rhs);

            // extended_gcd properties
            if !b.is_zero() {
                let (g_egcd, x, y) = a.extended_gcd(&b);
                prop_assert!(g_egcd == g);
                // a*x === g (mod b)
                if !b.is_zero() {
                    let ax = a.mul(&x);
                    let r_ax = ax.rem(&b);
                    let r_g = g.rem(&b);
                    prop_assert_eq!(r_ax, r_g);
                }
                // b*y === g (mod a)
                if !a.is_zero() {
                    let by = b.mul(&y);
                    let r_by = by.rem(&a);
                    let r_g2 = g.rem(&a);
                    prop_assert_eq!(r_by, r_g2);
                }
            }
        }
}
