//! Exact cofactor identities across HGCD admission, scalar tails and asymmetry.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::{
    DivScratch, Division, EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS, EXTENDED_HGCD_CROSSOVER_THRESHOLD,
    Gcd, HgcdMatrix, HgcdWorkspace, InternalMpUint, Limb,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]

    #[test]
    fn scalar_extended_tail_preserves_bezout_for_either_order(
        head in any::<Limb>(),
        tail in any::<Limb>(),
    ) {
        let (gcd, x, y, parity) = Division::extended_gcd_limb(head, tail);
        let a = InternalMpUint::from_limb(head);
        let b = InternalMpUint::from_limb(tail);
        let ax = a.mul(&InternalMpUint::from_limb(x));
        let by = b.mul(&InternalMpUint::from_limb(y));
        let difference = if parity == 0 { ax.sub(&by) } else { by.sub(&ax) };
        prop_assert_eq!(difference, InternalMpUint::from_limb(gcd));
        prop_assert_eq!(gcd, Gcd::gcd_1(head, tail));
    }

    #[test]
    fn close_cofactors_preserve_both_signs_and_nonunits(
        mut high in proptest::collection::vec(any::<Limb>(), 1..=128),
        low_a in prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()],
        low_b in prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *high.last_mut().expect("nonempty high suffix") |= 1;
        let mut left = vec![low_a];
        left.extend_from_slice(&high);
        let mut right = vec![low_b];
        right.extend_from_slice(&high);
        let a = InternalMpUint::from_limbs(left);
        let b = InternalMpUint::from_limbs(right);
        let result = a.extended_gcd(&b);
        let ax = a.mul(&result.x_magnitude);
        let by = b.mul(&result.y_magnitude);
        let identity = if result.x_is_positive { ax.sub(&by) } else { by.sub(&ax) };
        prop_assert_eq!(&identity, &result.gcd);
        prop_assert_eq!(&result.gcd, &a.gcd(&b));
        let inverse = a.invert(&b);
        prop_assert_eq!(inverse.is_some(), result.gcd.is_one());
        if let Some(value) = inverse {
            prop_assert!(value < b);
            prop_assert!(a.mul(&value).rem(&b).is_one());
        }
    }

    #[test]
    fn extended_core_preserves_the_signed_bezout_coefficient(
        left in proptest::collection::vec(any::<Limb>(), 0..=257),
        right in proptest::collection::vec(any::<Limb>(), 1..=257),
    ) {
        let a = InternalMpUint::from_limbs(left);
        let b = InternalMpUint::from_limbs(right);
        prop_assume!(!b.is_zero());
        let (gcd, coefficient, parity) = Division::compute_extended_euclid_core(&a, &b);
        prop_assert_eq!(&gcd, &a.gcd(&b));
        prop_assert!(coefficient <= b);
        if gcd.is_one() && !b.is_one() {
            prop_assert!(!coefficient.is_zero() && coefficient < b);
        }
        let mut numerator = a.mul(&coefficient);
        if parity & 1 == 0 {
            prop_assert!(numerator >= gcd);
            numerator.sub_assign(&gcd);
        } else {
            numerator.add_assign(&gcd);
        }
        let mut residue = InternalMpUint::zero();
        Division::rem_into(&numerator, &b, &mut residue, &mut DivScratch::default());
        prop_assert!(residue.is_zero());
    }

    #[test]
    fn retained_hgcd_matrix_reconstructs_both_original_operands(
        limbs in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 63..=129),
    ) {
        let (left, right): (Vec<_>, Vec<_>) = limbs.into_iter().unzip();
        let a = InternalMpUint::from_limbs(left);
        let b = InternalMpUint::from_limbs(right);
        let (mut u, mut v) = if a >= b { (a, b) } else { (b, a) };
        let original_u = u.clone();
        let original_v = v.clone();
        let mut matrix = HgcdMatrix::default();
        if Gcd::hgcd_block_matrix(&mut u, &mut v, &mut matrix,
            &mut DivScratch::default(), &mut HgcdWorkspace::default()) {
            prop_assert_eq!(matrix.m00.mul(&u).add(&matrix.m01.mul(&v)), original_u);
            prop_assert_eq!(matrix.m10.mul(&u).add(&matrix.m11.mul(&v)), original_v);
            let diagonal = matrix.m00.mul(&matrix.m11);
            let off_diagonal = matrix.m01.mul(&matrix.m10);
            if matrix.positive_det {
                prop_assert_eq!(diagonal, off_diagonal.add(&InternalMpUint::one()));
            } else {
                prop_assert_eq!(off_diagonal, diagonal.add(&InternalMpUint::one()));
            }
        } else {
            prop_assert_eq!((u, v), (original_u, original_v));
        }
    }
}

#[test]
fn scalar_extended_fibonacci_tail_reaches_the_limb_boundary() {
    let (mut a, mut b) = (1_usize, 1_usize);
    loop {
        for (head, tail) in [(a, b), (b, a)] {
            let (gcd, x, y, parity) = Division::extended_gcd_limb(head, tail);
            let ax = InternalMpUint::from_limb(head).mul(&InternalMpUint::from_limb(x));
            let by = InternalMpUint::from_limb(tail).mul(&InternalMpUint::from_limb(y));
            assert_eq!(
                if parity == 0 {
                    ax.sub(&by)
                } else {
                    by.sub(&ax)
                },
                InternalMpUint::one()
            );
            assert_eq!(gcd, 1);
        }
        let Some(next) = a.checked_add(b) else { break };
        (a, b) = (b, next);
    }
}

#[test]
fn initial_extended_reduction_preserves_asymmetric_coefficients() {
    for width in [
        1,
        4,
        5,
        EXTENDED_HGCD_CROSSOVER_THRESHOLD.saturating_sub(1),
        EXTENDED_HGCD_CROSSOVER_THRESHOLD,
        EXTENDED_HGCD_CROSSOVER_THRESHOLD
            .checked_add(1)
            .expect("test width fits"),
    ] {
        let divisor = InternalMpUint::from_limbs(vec![Limb::MAX; width]);
        for quotient in [
            InternalMpUint::from_limb(3),
            InternalMpUint::from_limb(Limb::MAX),
            InternalMpUint::from_limbs(vec![0, 1]),
        ] {
            for residue in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                divisor.sub(&InternalMpUint::one()),
            ] {
                let dividend = divisor.mul(&quotient).add(&residue);
                let expected_gcd = divisor.gcd(&residue);
                for (left, right) in [(&dividend, &divisor), (&divisor, &dividend)] {
                    let result = left.extended_gcd(right);
                    assert_eq!(result.gcd, expected_gcd);
                    let left_product = left.mul(&result.x_magnitude);
                    let right_product = right.mul(&result.y_magnitude);
                    let identity = if result.x_is_positive {
                        left_product.sub(&right_product)
                    } else {
                        right_product.sub(&left_product)
                    };
                    assert_eq!(identity, expected_gcd);
                    let inverse = left.invert(right);
                    assert_eq!(inverse.is_some(), expected_gcd.is_one());
                    if let Some(value) = inverse {
                        assert!(value < *right);
                        assert_eq!(
                            left.mul(&value).rem(right),
                            InternalMpUint::one().rem(right)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cofactor_completion_crosses_its_width_boundary() {
    let boundary = EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS;
    for width in [
        boundary.checked_sub(1).expect("positive batch minimum"),
        boundary,
        boundary.checked_add(1).expect("test width fits"),
    ] {
        // Let B be the limb radix, t=2B+1, u=B+1, b=Q*t+u,
        // a=b+t. Then -(2Q+1)*a + (2Q+3)*b = 2u-t = 1.
        // After the two leading quotients 1 and Q, two-limb
        // remainders coexist with a cofactor of exactly `width` limbs.
        let mut limbs = vec![0; width];
        *limbs.last_mut().expect("positive cofactor width") = 1;
        let quotient = InternalMpUint::from_limbs(limbs);
        let head_remainder = InternalMpUint::from_limbs(vec![1, 2]);
        let tail_remainder = InternalMpUint::from_limbs(vec![1, 1]);
        let divisor = quotient.mul(&head_remainder).add(&tail_remainder);
        let dividend = divisor.add(&head_remainder);
        let expected = quotient
            .mul(&InternalMpUint::from_limb(2))
            .add(&InternalMpUint::one());
        let (gcd, coefficient, parity) =
            Division::compute_extended_euclid_core(&dividend, &divisor);
        assert!(gcd.is_one());
        assert_eq!(coefficient, expected);
        assert_eq!(parity, 1);
    }
}

#[test]
fn extended_euclid_covers_crossover_and_exact_scalar_tails() {
    for width in [
        2,
        3,
        4,
        5,
        EXTENDED_HGCD_CROSSOVER_THRESHOLD.saturating_sub(1),
        EXTENDED_HGCD_CROSSOVER_THRESHOLD,
        EXTENDED_HGCD_CROSSOVER_THRESHOLD
            .checked_add(1)
            .expect("test width fits"),
    ] {
        // Consecutive integers have GCD one. Odd common factors exercise
        // noninvertible operands and preserve the same signed coefficients.
        let a = InternalMpUint::from_limbs(vec![Limb::MAX; width]);
        let b = a.add(&InternalMpUint::one());
        for factor in [1, 3, Limb::MAX] {
            let common = InternalMpUint::from_limb(factor);
            let left = a.mul(&common);
            let right = b.mul(&common);
            for (first, second) in [(&left, &right), (&right, &left), (&right, &common)] {
                let result = first.extended_gcd(second);
                assert_eq!(result.gcd, common);
                let first_product = first.mul(&result.x_magnitude);
                let second_product = second.mul(&result.y_magnitude);
                let identity = if result.x_is_positive {
                    first_product.sub(&second_product)
                } else {
                    second_product.sub(&first_product)
                };
                assert_eq!(identity, common);
            }
        }
    }
}
