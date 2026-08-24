//! Public number-theory API properties.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "property tests verify arithmetic operations and cloning semantics"
)]

use core::cmp::Ordering;

use proptest::prelude::*;

use super::{nz, strategies};
use crate::int::api::{MpInt, MpUint};

proptest! {
    #[test]
    fn prop_theory_phi_prime(val in 2_u64..=1000) {
        let n = MpUint::from(val);
        if n.is_prime() {
            prop_assert_eq!(n.euler_phi(), Some(MpUint::from(val - 1)));
        }
    }
}

proptest! {
    #[test]
    fn prop_next_prime_preserves_bounded_invariants(
        bits in 1_usize..=16,
        unsigned_seed in any::<u16>(),
        signed_seed in any::<i16>(),
    ) {
        let unsigned = MpUint::with_precision_wrapping(unsigned_seed, nz(bits));
        if let Some(next) = unsigned.next_prime() {
            prop_assert_eq!(next.precision, unsigned.precision);
            prop_assert!(next.significant_bits() <= bits);
            prop_assert!(next >= unsigned);
            prop_assert!(next.is_prime());
        }

        let signed = MpInt::with_precision_wrapping(signed_seed, nz(bits));
        if let Some(next) = signed.next_prime() {
            prop_assert_eq!(next.precision, signed.precision);
            prop_assert!(next.value.required_signed_bits_for_bounded_storage() <= bits);
            prop_assert!(next >= signed);
            prop_assert!(next.is_prime());
        }
    }
}

proptest! {
    #[test]
    fn prop_theory_phi_composite(p in 2_u64..=50, q in 2_u64..=50) {
        let np = MpUint::from(p);
        let nq = MpUint::from(q);
        if np.is_prime() && nq.is_prime() && p != q {
            let n = MpUint::from(p * q);
            let expected = MpUint::from((p - 1) * (q - 1));
            prop_assert_eq!(n.euler_phi(), Some(expected));
        }
    }
}

proptest! {
    #[test]
    fn prop_int_pow_mod_negative_exponent(
        base_val in -1000_i64..=1000_i64,
        mod_val in 2_i64..=1000_i64,
        exp_val in 1_i64..=50_i64,
    ) {
        let base = MpInt::from(base_val);
        let modulus = MpInt::from(mod_val);
        let pos_exp = MpInt::from(exp_val);
        let neg_exp = MpInt::from(-exp_val);

        let inv = base.invert(&modulus);
        if let Some(inv_base) = inv {
            let expected = inv_base.pow_mod(&pos_exp, &modulus);
            let actual = base.pow_mod(&neg_exp, &modulus);
            prop_assert_eq!(actual, expected);
        } else {
            let actual = base.pow_mod(&neg_exp, &modulus);
            prop_assert_eq!(actual, None);
        }
    }
}

proptest! {
    #[test]
    fn prop_modular_inverse_unit_fast_paths(m in strategies::uint_nonzero(16)) {
        if m.is_one() {
            return Ok(());
        }

        let one = MpUint::one();
        let predecessor = &m - &one;
        prop_assert_eq!(one.invert(&m), Some(one));
        prop_assert_eq!(predecessor.invert(&m), Some(predecessor));
    }
}

proptest! {
    #[test]
    fn prop_theory_square(
        u in strategies::uint(64),
        i in strategies::int(64),
    ) {
        prop_assert_eq!(u.square(), &u * &u);
        prop_assert_eq!(i.square(), &i * &i);
    }
}

proptest! {
    #[test]
    fn prop_theory_sqrt_rem_contract(u in strategies::uint(8)) {
        let Some((root, rem)) = u.sqrt_rem() else {
            prop_assert!(false, "sqrt_rem should be defined for unsigned integers");
            return Ok(());
        };

        let root_sq = &root * &root;
        let recomposed = &root_sq + &rem;
        prop_assert_eq!(
            recomposed.cmp(&u),
            Ordering::Equal,
            "sqrt_rem decomposition"
        );

        let next = &root + &MpUint::one();
        let next_sq = &next * &next;
        prop_assert!(
            root_sq.cmp(&u) != Ordering::Greater,
            "root is not too large"
        );
        prop_assert!(u.cmp(&next_sq) == Ordering::Less, "root is maximal");

        let two_root_plus_one = &(&root << 1_usize) + &MpUint::one();
        prop_assert!(rem < two_root_plus_one, "remainder bound");
    }
}

proptest! {
    #[test]
    fn prop_theory_isqrt_contract(u in strategies::uint(8)) {
        let Some(root) = u.isqrt() else {
            prop_assert!(false, "isqrt should be defined for unsigned integers");
            return Ok(());
        };

        let root_sq = &root * &root;
        let next = &root + &MpUint::one();
        let next_sq = &next * &next;

        prop_assert!(
            root_sq.cmp(&u) != Ordering::Greater,
            "root is not too large"
        );
        prop_assert!(u.cmp(&next_sq) == Ordering::Less, "root is maximal");
    }
}

proptest! {
    #[test]
    fn prop_theory_barrett_reduce(
        u in strategies::uint(64),
        m in strategies::uint_nonzero(32),
        i in strategies::int(64),
        mut m_int in strategies::int(32),
    ) {
        let expected = Some(&u % &m);
        prop_assert_eq!(u.barrett_reduce(&m), expected);
        if m_int.is_zero() { m_int = MpInt::one(); }
        let expected_int = Some(&i.abs() % &m_int.abs());
        prop_assert_eq!(i.barrett_reduce(&m_int), expected_int);
    }
}

proptest! {
    #[test]
    fn prop_theory_montgomery_mul(
        a in strategies::uint(32),
        b in strategies::uint(32),
        m in strategies::uint(32),
    ) {
        let m_odd = m | MpUint::one();
        let ab = a.montgomery_mul(&b, &m_odd);
        let ba = b.montgomery_mul(&a, &m_odd);
        prop_assert_eq!(ab, ba, "montgomery_mul should be commutative");

        let am_m = a.montgomery_mul(&m_odd, &m_odd);
        prop_assert_eq!(am_m, Some(MpUint::zero()));

        let even_m = m_odd ^ MpUint::one();
        if !even_m.is_zero() {
            prop_assert_eq!(a.montgomery_mul(&b, &even_m), None);
        }
    }
}

proptest! {
    #[test]
    fn prop_theory_int_extended_gcd(
        a in strategies::int(32),
        b in strategies::int(32),
    ) {
        if let Some((g, x, y)) = a.extended_gcd(&b) {
            prop_assert_eq!(&a * &x + &b * &y, g);
        }
    }
}

proptest! {
    #[test]
    fn prop_signed_euler_phi_requires_positive_input(input in -1_000_i64..=1_000) {
        let candidate = MpInt::from(input);
        if input <= 0 {
            prop_assert_eq!(candidate.euler_phi(), None);
        } else {
            let magnitude = MpUint::from(input.unsigned_abs());
            let expected = magnitude.euler_phi().map(MpInt::from);
            prop_assert_eq!(candidate.euler_phi(), expected);
        }
    }
}

proptest! {
    #[test]
    fn prop_signed_jacobi_preserves_numerator_sign(
        numerator in any::<i64>(),
        modulus_seed in any::<u16>(),
    ) {
        let odd_modulus = u64::from(modulus_seed) | 1;
        let argument = MpInt::from(numerator);
        let denominator = MpInt::from(odd_modulus);
        let magnitude = MpUint::from(numerator.unsigned_abs());
        let unsigned_modulus = MpUint::from(odd_modulus);
        let expected = magnitude.jacobi_symbol(&unsigned_modulus).map(|symbol| {
            if numerator < 0 && odd_modulus & 3 == 3 {
                symbol.wrapping_neg()
            } else {
                symbol
            }
        });

        prop_assert_eq!(argument.jacobi_symbol(&denominator), expected);
        prop_assert_eq!(
            argument.jacobi_symbol(&MpInt::from(
                i64::try_from(odd_modulus)
                    .expect("a u16-derived odd modulus fits i64")
                    .wrapping_neg()
            )),
            None,
        );
        let even_modulus = u64::from(modulus_seed).wrapping_add(1).wrapping_mul(2);
        prop_assert_eq!(
            argument.jacobi_symbol(&MpInt::from(even_modulus)),
            None,
        );
    }
}

proptest! {
    #[test]
    fn prop_theory_jacobi_symbol(
        a in strategies::uint(64),
        b in strategies::uint(64),
    ) {
        let n = &b | &MpUint::one();
        if n.is_one() {
            return Ok(());
        }

        let a_rem = &a % &n;
        prop_assert_eq!(a_rem.jacobi_symbol(&n), a.jacobi_symbol(&n), "periodicity");

        if let (Some(j_a), Some(j_b)) = (a.jacobi_symbol(&n), b.jacobi_symbol(&n)) {
            let ab = &a * &b;
            let j_ab = ab.jacobi_symbol(&n);
            prop_assert_eq!(j_ab, Some(j_a * j_b), "multiplicativity");
        }

        let m = &b | &MpUint::one();
        if m.is_one() { return Ok(()); }
        if !m.is_one()
            && a.is_coprime(&m)
            && !(&a & &MpUint::one()).is_zero()
            && let (Some(j_am), Some(j_ma)) = (a.jacobi_symbol(&m), m.jacobi_symbol(&a))
        {
            let a_mod4 = u32::try_from(&a & &MpUint::from(3_u32)).unwrap_or(0);
            let m_mod4 = u32::try_from(&m & &MpUint::from(3_u32)).unwrap_or(0);
            let expected_prod = if a_mod4 == 3 && m_mod4 == 3 { -1 } else { 1 };
            prop_assert_eq!(j_am * j_ma, expected_prod, "quadratic reciprocity");
        }
    }
}
