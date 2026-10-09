//! Independent arithmetic oracles for GCD transitions and storage reuse.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::int::logic::unsigned::math::{
    KARATSUBA_THRESHOLD, LEHMER_BRANCHLESS_THRESHOLD,
    gcd::{lehmer::lehmer_update, lehmer_simulation::lehmer_simulate_wide},
};

use super::{
    DivScratch, DoubleLimb, Gcd, HGCD_CROSSOVER_THRESHOLD, HgcdMatrix, HgcdWorkspace,
    InternalMpUint, LIMB_BITS, Limb,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 2_048 }))]

    #[test]
    fn hybrid_windows_preserve_exact_matrix_action_at_truncation_extremes(
        first in any::<DoubleLimb>(),
        second in any::<DoubleLimb>(),
        near_switch in any::<bool>(),
    ) {
        let limit: DoubleLimb = (1 << LIMB_BITS) << (LIMB_BITS >> 1);
        let head = if near_switch { first % limit } else { first };
        let tail = if near_switch { second % limit } else { second };
        let mut quotients = Vec::new();
        let (u0, v0, u1, v1, even) =
            lehmer_simulate_wide::<false>(head, tail, |q| quotients.push(q));
        // Each output is affine in the omitted tails, so the four corners
        // include its minimum and maximum over every possible omitted limb.
        for (head_tail, divisor_tail) in [(0, 0), (0, Limb::MAX), (Limb::MAX, 0), (Limb::MAX, Limb::MAX)] {
            let mut left = InternalMpUint::from_le_bytes(&head.to_le_bytes());
            left.shl_assign(LIMB_BITS);
            left.add_assign(&InternalMpUint::from_limb(head_tail));
            let mut right = InternalMpUint::from_le_bytes(&tail.to_le_bytes());
            right.shl_assign(LIMB_BITS);
            right.add_assign(&InternalMpUint::from_limb(divisor_tail));
            let mut exact_left = left.clone();
            let mut exact_right = right.clone();
            for &q in &quotients {
                let product = exact_right.mul(&InternalMpUint::from_limb(q));
                prop_assert!(exact_left >= product, "accepted quotient makes the full remainder negative");
                let remainder = exact_left.sub(&product);
                exact_left = exact_right;
                exact_right = remainder;
            }
            let mut next_left = InternalMpUint::zero();
            let mut next_right = InternalMpUint::zero();
            prop_assert!(lehmer_update::<false>(
                &mut left, &mut right, &mut next_left, &mut next_right,
                u0, v0, u1, v1, even,
            ));
            prop_assert_eq!((left, right), (exact_left, exact_right));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 1_024 }))]

    #[test]
    fn independent_product_carries_match_signed_carry_stream(
        pairs in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 1..=if cfg!(miri) { 5 } else { 128 }),
        coefficients in proptest::array::uniform4(prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()]),
        even in any::<bool>(),
    ) {
        let (mut left_limbs, mut right_limbs): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        *left_limbs.last_mut().expect("nonempty source") |= 1;
        *right_limbs.last_mut().expect("nonempty source") |= 1;
        let mut left = InternalMpUint::from_limbs(left_limbs);
        let mut right = InternalMpUint::from_limbs(right_limbs);
        let mut reference_left = left.clone();
        let mut reference_right = right.clone();
        let mut next_left = InternalMpUint::zero();
        let mut next_right = InternalMpUint::zero();
        let mut reference_next_left = InternalMpUint::zero();
        let mut reference_next_right = InternalMpUint::zero();
        let [u0, v0, u1, v1] = coefficients;
        let actual = lehmer_update::<false>(
            &mut left, &mut right, &mut next_left, &mut next_right,
            u0, v0, u1, v1, even,
        );
        let expected = lehmer_update::<true>(
            &mut reference_left, &mut reference_right,
            &mut reference_next_left, &mut reference_next_right,
            u0, v0, u1, v1, even,
        );
        prop_assert_eq!(actual, expected);
        prop_assert_eq!((left, right), (reference_left, reference_right));
        prop_assert_eq!((next_left, next_right), (reference_next_left, reference_next_right));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]

    #[test]
    fn paired_cofactor_products_match_independent_products(
        left_limbs in proptest::collection::vec(prop_oneof![Just(Limb::MAX), any::<Limb>()], 0..=if cfg!(miri) { 5 } else { 80 }),
        right_limbs in proptest::collection::vec(prop_oneof![Just(Limb::MAX), any::<Limb>()], 0..=if cfg!(miri) { 5 } else { 80 }),
        scalars in proptest::array::uniform4(prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()]),
    ) {
        let left = InternalMpUint::from_limbs(left_limbs);
        let right = InternalMpUint::from_limbs(right_limbs);
        let [a, b, c, d] = scalars;
        let expected_first = left.mul(&InternalMpUint::from_limb(a))
            .add(&right.mul(&InternalMpUint::from_limb(b)));
        let expected_second = left.mul(&InternalMpUint::from_limb(c))
            .add(&right.mul(&InternalMpUint::from_limb(d)));
        let mut first = InternalMpUint::zero();
        let mut second = InternalMpUint::zero();
        for reused in [false, true] {
            if reused {
                first = InternalMpUint::from_limbs(vec![Limb::MAX; 90]);
                second = first.clone();
            }
            Gcd::assign_linear_combinations(&mut first, &mut second, &left, &right, a, b, c, d);
            prop_assert_eq!(&first, &expected_first);
            prop_assert_eq!(&second, &expected_second);
        }
    }

    #[test]
    fn matrix_product_matches_four_independent_dot_products(
        entries in proptest::collection::vec(
            proptest::collection::vec(any::<Limb>(), 0..=65), 8),
        width in prop_oneof![Just(0), Just(1), Just(4), Just(5),
            Just(KARATSUBA_THRESHOLD.checked_sub(1).expect("nonzero threshold")),
            Just(KARATSUBA_THRESHOLD),
            Just(KARATSUBA_THRESHOLD.checked_add(1).expect("test size fits")),
            Just(64)],
    ) {
        let mut values = entries.into_iter().map(|mut limbs| {
            limbs.resize(width, Limb::MAX);
            if let Some(top) = limbs.last_mut() { *top |= 1; }
            limbs
        });
        let mut left = matrix_from_entries(values.by_ref().take(4).collect(), true);
        let right = matrix_from_entries(values.collect(), false);
        let expected = [
            left.m00.mul(&right.m00).add(&left.m01.mul(&right.m10)),
            left.m00.mul(&right.m01).add(&left.m01.mul(&right.m11)),
            left.m10.mul(&right.m00).add(&left.m11.mul(&right.m10)),
            left.m10.mul(&right.m01).add(&left.m11.mul(&right.m11)),
        ];
        // Poison every reusable destination to expose incomplete initialization.
        let mut frame = super::HgcdFrame {
            next_matrix: matrix_from_entries(vec![vec![Limb::MAX; 129]; 4], true),
            ..super::HgcdFrame::default()
        };
        left.multiply_right(&right, &mut frame.next_matrix,
            &mut frame.product_a, &mut frame.product_b,
            &mut DivScratch::default());
        prop_assert_eq!([left.m00, left.m01, left.m10, left.m11], expected);
        prop_assert!(!left.positive_det);
    }

    #[test]
    fn scalar_matrix_composition_matches_full_products(
        entries in proptest::collection::vec(
            proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 24 }), 4),
        coefficients in proptest::array::uniform4(prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()]),
        positive_det in any::<bool>(),
        even in any::<bool>(),
    ) {
        let mut matrix = matrix_from_entries(entries, positive_det);
        let [u0, v0, u1, v1] = coefficients;
        let expected = [
            matrix.m00.mul(&InternalMpUint::from_limb(v1)).add(&matrix.m01.mul(&InternalMpUint::from_limb(u1))),
            matrix.m00.mul(&InternalMpUint::from_limb(v0)).add(&matrix.m01.mul(&InternalMpUint::from_limb(u0))),
            matrix.m10.mul(&InternalMpUint::from_limb(v1)).add(&matrix.m11.mul(&InternalMpUint::from_limb(u1))),
            matrix.m10.mul(&InternalMpUint::from_limb(v0)).add(&matrix.m11.mul(&InternalMpUint::from_limb(u0))),
        ];
        let mut next = HgcdMatrix::default();
        matrix.update_small(&mut next, u0, v0, u1, v1, even);
        prop_assert_eq!([matrix.m00, matrix.m01, matrix.m10, matrix.m11], expected);
        prop_assert_eq!(matrix.positive_det, positive_det == even);
    }

    #[test]
    fn sparse_matrix_transitions_match_full_products(
        entries in proptest::collection::vec(
            proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 24 }), 4),
        quotient_limbs in proptest::collection::vec(any::<Limb>(), 1..=8),
        positive_det in any::<bool>(),
    ) {
        let mut matrix = matrix_from_entries(entries, positive_det);
        let mut quotient = InternalMpUint::from_limbs(quotient_limbs);
        if quotient.is_zero() {
            quotient.increment();
        }
        let expected = [
            matrix.m00.mul(&quotient).add(&matrix.m01),
            matrix.m00.clone(),
            matrix.m10.mul(&quotient).add(&matrix.m11),
            matrix.m10.clone(),
        ];
        let mut next = HgcdMatrix::default();
        matrix.update_quotient(&mut next, &quotient, &mut DivScratch::default());
        prop_assert_eq!([matrix.m00, matrix.m01, matrix.m10, matrix.m11], expected);
        prop_assert_eq!(matrix.positive_det, !positive_det);
    }

    #[test]
    fn full_width_lehmer_batches_match_exact_matrix_action(
        mut left_limbs in proptest::collection::vec(any::<Limb>(), 3..=if cfg!(miri) { 5 } else { 80 }),
        mut right_limbs in proptest::collection::vec(any::<Limb>(), 3..=if cfg!(miri) { 5 } else { 80 }),
        coefficients in proptest::array::uniform4(prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()]),
        equal_width in any::<bool>(),
        even in any::<bool>(),
    ) {
        if equal_width {
            right_limbs.resize(left_limbs.len(), 0);
            *left_limbs.last_mut().expect("nonempty source") |= 1;
            *right_limbs.last_mut().expect("equal nonempty source") |= 1;
        }
        let left = InternalMpUint::from_limbs(left_limbs);
        let right = InternalMpUint::from_limbs(right_limbs);
        let [u0, v0, u1, v1] = coefficients;
        let a = left.mul(&InternalMpUint::from_limb(u0));
        let b = right.mul(&InternalMpUint::from_limb(v0));
        let c = right.mul(&InternalMpUint::from_limb(v1));
        let d = left.mul(&InternalMpUint::from_limb(u1));
        let (positive_u, negative_u, positive_v, negative_v) = if even {
            (a, b, c, d)
        } else {
            (b, a, d, c)
        };
        let nonnegative = positive_u >= negative_u && positive_v >= negative_v;
        let max_len = left.limbs().len().max(right.limbs().len());
        let expected = nonnegative.then(|| (positive_u.sub(&negative_u), positive_v.sub(&negative_v)));
        let expected_valid = expected.as_ref().is_some_and(|(u, v)| {
            u.limbs().len() <= max_len && v.limbs().len() <= max_len
        });
        let mut u = left.clone();
        let mut v = right.clone();
        let mut next_u = InternalMpUint::zero();
        let mut next_v = InternalMpUint::zero();
        let valid = lehmer_update::<false>(
            &mut u, &mut v, &mut next_u, &mut next_v, u0, v0, u1, v1, even);
        prop_assert_eq!(valid, expected_valid);
        if valid {
            prop_assert_eq!((u, v), expected.expect("valid reconstruction exists"));
        } else {
            prop_assert_eq!((u, v), (left, right));
        }
    }
}

#[test]
fn two_limb_equality_restores_the_implicit_high_bit() {
    for low in [1, 3, Limb::MAX] {
        assert_eq!(Gcd::gcd_2([low, 1], [low, 1]), [low, 1]);
        let value = InternalMpUint::from_limbs_2(low, 1);
        assert_eq!(value.gcd(&value), value);
    }
    // An unequal pair can reach the same equality exit after a reduction.
    let value = InternalMpUint::from_limbs_2(3, 1);
    let triple = value.mul(&InternalMpUint::from_limb(3));
    assert_eq!(value.gcd(&triple), value);
}

#[test]
fn paired_cofactor_carries_initialize_both_guards() {
    let mut first = InternalMpUint::zero();
    let mut second = InternalMpUint::zero();
    for width in [0_usize, 1, 4, 5, 32, 5, 1, 0] {
        let left = InternalMpUint::from_limbs(vec![Limb::MAX; width]);
        let right = left.clone();
        let expected = left.mul(&InternalMpUint::from_limb(Limb::MAX));
        Gcd::assign_linear_combinations(
            &mut first,
            &mut second,
            &left,
            &right,
            Limb::MAX,
            Limb::MAX,
            Limb::MAX,
            0,
        );
        assert_eq!(first, expected.add(&expected));
        assert_eq!(second, expected);
    }
}

#[test]
fn matrix_products_initialize_fresh_and_reused_destinations() {
    let mut first = InternalMpUint::zero();
    let mut second = InternalMpUint::zero();
    let mut scratch = DivScratch::default();
    // Cross inline/heap boundaries in both directions, and retain dirty
    // capacity when either matrix entry becomes zero.
    for width in [4, 5, 17, 1, 0, 5] {
        let left = InternalMpUint::from_limbs(vec![Limb::MAX; width]);
        let right = left.add(&InternalMpUint::from_limb(2));
        let low = InternalMpUint::from_limbs(vec![Limb::MAX; 5]);
        let zero = InternalMpUint::zero();
        for (a, b) in [
            (&left, &right),
            (&zero, &right),
            (&left, &zero),
            (&zero, &zero),
        ] {
            HgcdMatrix::assign_product_slice_two_by_one(
                &mut first,
                &mut second,
                a,
                b,
                low.limbs(),
                &mut scratch,
            );
            assert_eq!(first, a.mul(&low));
            assert_eq!(second, b.mul(&low));
            Gcd::assign_linear_combination(&mut first, a, Limb::MAX, b, Limb::MAX);
            assert_eq!(first, a.add(b).mul(&InternalMpUint::from_limb(Limb::MAX)));
        }
    }
}

#[test]
fn matrix_reset_retains_heap_storage() {
    let mut matrix = matrix_from_entries(vec![vec![Limb::MAX; 17]; 4], false);
    let pointers = [
        matrix.m00.limbs().as_ptr(),
        matrix.m01.limbs().as_ptr(),
        matrix.m10.limbs().as_ptr(),
        matrix.m11.limbs().as_ptr(),
    ];
    matrix.reset();
    assert!(matrix.is_identity());
    assert_eq!(
        pointers,
        [
            matrix.m00.limbs().as_ptr(),
            matrix.m01.limbs().as_ptr(),
            matrix.m10.limbs().as_ptr(),
            matrix.m11.limbs().as_ptr(),
        ]
    );
}

#[test]
fn unit_quotient_scalar_update_matches_reference() {
    // [[3,5],[7,11]] * [[1,1],[1,0]] = [[8,3],[18,7]] with flipped sign.
    let mut matrix = matrix_from_entries(vec![vec![3], vec![5], vec![7], vec![11]], true);
    let mut next = HgcdMatrix::default();
    matrix.update_quotient_scalar(&mut next, 1);
    assert_eq!(matrix.m00, InternalMpUint::from_limb(8));
    assert_eq!(matrix.m01, InternalMpUint::from_limb(3));
    assert_eq!(matrix.m10, InternalMpUint::from_limb(18));
    assert_eq!(matrix.m11, InternalMpUint::from_limb(7));
    assert!(!matrix.positive_det);
}

#[test]
fn common_zero_limbs_and_inline_handoffs_preserve_gcd() {
    let mut workspace = HgcdWorkspace::default();
    for width in [1_usize, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129] {
        if cfg!(miri) && width > 5 {
            continue;
        }
        let mut left = InternalMpUint::one();
        left.shl_assign(width.checked_mul(LIMB_BITS).expect("test width fits"));
        left.decrement();
        let mut right = left.clone();
        right.add_assign(&InternalMpUint::from_limb(2));
        // Consecutive odd cofactors have gcd one. The large common power of
        // two tests whole-limb elision and every residual shift endpoint.
        for residual in [0, 1, LIMB_BITS.checked_sub(1).expect("nonzero limb width")] {
            let shift = (if cfg!(miri) { 5_usize } else { 257 })
                .checked_mul(LIMB_BITS)
                .and_then(|bits| bits.checked_add(residual))
                .expect("test shift fits");
            let mut a = left.clone();
            let mut b = right.clone();
            a.shl_assign(shift);
            b.shl_assign(shift);
            let mut expected = InternalMpUint::one();
            expected.shl_assign(shift);
            assert_eq!(a.gcd(&b), expected);
            assert_eq!(Gcd::compute_half_gcd(&b, &a, &mut workspace), expected);
            assert_eq!(Gcd::compute_lehmer(&a, &a), a);
        }
    }
}

#[test]
fn close_pairs_reduce_to_their_small_difference() {
    for width in [3, 4, 5, 63, 64, 65, 127, 128, 129] {
        if cfg!(miri) && width > 5 {
            continue;
        }
        for low in [0, 1, 2, 3, 4, Limb::MAX] {
            let mut limbs = vec![Limb::MAX; width];
            *limbs.first_mut().expect("nonempty fixture") = low;
            *limbs.get_mut(1).expect("second fixture limb") = 1;
            let a = InternalMpUint::from_limbs(limbs);
            for step in [
                1,
                2,
                1 << LIMB_BITS.checked_sub(1).expect("positive limb width"),
            ] {
                let b = a.add(&InternalMpUint::from_limb(step));
                // gcd(a, a+2^k) = gcd(a, 2^k), including carry. The even
                // fixtures have valuations 1, 2, or at least LIMB_BITS.
                let expected = InternalMpUint::from_limb(match low {
                    0 => step,
                    2 => step.min(2),
                    4 => step.min(4),
                    _ => 1,
                });
                assert_eq!(a.gcd(&b), expected);
                assert_eq!(b.gcd(&a), expected);
            }
            assert_eq!(a.gcd(&a), a);
        }
    }
}

#[cfg_attr(
    miri,
    ignore = "Production crossover matrices require long multi-limb Euclidean reductions."
)]
#[test]
fn euclidean_oracle_covers_each_crossover_and_asymmetric_tail() {
    let mut workspace = HgcdWorkspace::default();
    for width in [
        1_usize,
        2,
        3,
        4,
        5,
        63,
        64,
        65,
        127,
        128,
        129,
        LEHMER_BRANCHLESS_THRESHOLD
            .checked_sub(1)
            .expect("positive crossover"),
        LEHMER_BRANCHLESS_THRESHOLD,
        LEHMER_BRANCHLESS_THRESHOLD
            .checked_add(1)
            .expect("crossover fits"),
        HGCD_CROSSOVER_THRESHOLD
            .checked_sub(1)
            .expect("positive crossover"),
        HGCD_CROSSOVER_THRESHOLD,
        HGCD_CROSSOVER_THRESHOLD
            .checked_add(1)
            .expect("crossover fits"),
    ] {
        // The recurrence is explicitly modulo the limb base; the odd
        // multiplier and increment exercise all low-bit carry positions.
        let mut state = 17_usize;
        let left = InternalMpUint::from_limbs(
            (0..width)
                .map(|_| {
                    state = state.wrapping_mul(40_503).wrapping_add(1);
                    state | 1
                })
                .collect(),
        );
        let right = InternalMpUint::from_limbs(
            (0..width)
                .map(|_| {
                    state = state.wrapping_mul(34_283).wrapping_add(1);
                    state | 1
                })
                .collect(),
        );
        assert_eq!(left.limbs().len(), width);
        assert_eq!(right.limbs().len(), width);
        for divisor in [
            right,
            InternalMpUint::one(),
            InternalMpUint::from_limb(Limb::MAX),
        ] {
            let expected = euclidean_reference(left.clone(), divisor.clone());
            assert_eq!(left.gcd(&divisor), expected);
            assert_eq!(Gcd::compute_lehmer(&divisor, &left), expected);
            assert_eq!(
                Gcd::compute_half_gcd(&left, &divisor, &mut workspace),
                expected
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 24 }))]

    #[test]
    fn shared_high_limbs_match_euclidean_reduction(
        high in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 5 } else { 96 }),
        first in proptest::array::uniform2(any::<Limb>()),
        second in proptest::array::uniform2(any::<Limb>()),
    ) {
        let mut first_limbs = first.to_vec();
        first_limbs.extend_from_slice(&high);
        let mut second_limbs = second.to_vec();
        second_limbs.extend_from_slice(&high);
        let left = InternalMpUint::from_limbs(first_limbs);
        let right = InternalMpUint::from_limbs(second_limbs);
        let expected = euclidean_reference(left.clone(), right.clone());
        prop_assert_eq!(&left.gcd(&right), &expected);
        prop_assert_eq!(&right.gcd(&left), &expected);
        prop_assert_eq!(left.gcd(&left), left);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Random operands at production HGCD crossovers require long Euclidean reductions.")]
    fn gcd_routes_match_euclidean_reduction_at_crossovers(
        (left_limbs, right_limbs) in prop::sample::select(vec![
            1_usize, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129,
            HGCD_CROSSOVER_THRESHOLD.checked_sub(1).expect("positive crossover"),
            HGCD_CROSSOVER_THRESHOLD,
            HGCD_CROSSOVER_THRESHOLD.checked_add(1).expect("crossover fits"),
        ]).prop_flat_map(|width| (
            proptest::collection::vec(any::<Limb>(), width),
            proptest::collection::vec(any::<Limb>(), width),
        )),
        common in 1_usize..=255,
    ) {
        let factor = InternalMpUint::from_limb(common);
        let left = InternalMpUint::from_limbs(left_limbs).mul(&factor);
        let right = InternalMpUint::from_limbs(right_limbs).mul(&factor);
        let expected = euclidean_reference(left.clone(), right.clone());
        let mut workspace = HgcdWorkspace::default();
        prop_assert_eq!(&left.gcd(&right), &expected);
        prop_assert_eq!(&Gcd::compute_lehmer(&left, &right), &expected);
        prop_assert_eq!(&Gcd::compute_half_gcd(&left, &right, &mut workspace), &expected);
    }
}

fn euclidean_reference(mut left: InternalMpUint, mut right: InternalMpUint) -> InternalMpUint {
    while !right.is_zero() {
        let remainder = left.rem(&right);
        left = right;
        right = remainder;
    }
    left
}

fn matrix_from_entries(entries: Vec<Vec<Limb>>, positive_det: bool) -> HgcdMatrix {
    let mut values = entries.into_iter().map(InternalMpUint::from_limbs);
    HgcdMatrix {
        m00: values.next().expect("four matrix entries"),
        m01: values.next().expect("four matrix entries"),
        m10: values.next().expect("four matrix entries"),
        m11: values.next().expect("four matrix entries"),
        positive_det,
    }
}
