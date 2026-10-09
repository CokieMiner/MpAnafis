//! Slice and Fermat negation against ordinary subtraction chains.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded initialized test spans retain explicit guards and sentinel limbs around kernel windows"
)]

use alloc::vec;

use proptest::prelude::*;

use super::{LIMB_BITS, Limb, SsaRing, oracle_negation};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn slice_negation_matches_limb_subtraction(data in prop::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 129 })) {
        check_slice(&data);
    }

    #[test]
    fn ring_negation_matches_modulus_subtraction(data in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 129 }), guard in 0_usize..=1) {
        check_ring(&data, guard);
    }
}

#[test]
fn zero_prefixes_cover_every_vector_boundary_and_final_borrow() {
    for width in [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
    ] {
        if cfg!(miri) && width > 8 {
            continue;
        }
        let mut data = vec![0; width];
        check_slice(&data);
        for first_nonzero in 0..width {
            for first_digit in [1, Limb::MAX] {
                data[..first_nonzero].fill(0);
                data[first_nonzero] = first_digit;
                for (offset, digit) in data[first_nonzero + 1..].iter_mut().enumerate() {
                    *digit = [0, Limb::MAX, 1, Limb::MAX - 1][offset.rem_euclid(4)];
                }
                check_slice(&data);
            }
        }
    }
}

#[test]
fn modular_negation_covers_every_carry_prefix_and_special_residue() {
    for ml in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129] {
        if cfg!(miri) && ml > 8 {
            continue;
        }
        for guard in [0, 1] {
            for low in [0, 1, 2, 3, Limb::MAX - 1, Limb::MAX] {
                let mut data = vec![0; ml];
                data[0] = low;
                check_ring(&data, guard);
                for stop in 1..ml {
                    data[stop] = 1;
                    check_ring(&data, guard);
                    data[stop] = 0;
                }
            }
            check_ring(&vec![Limb::MAX; ml], guard);
        }
    }
}

fn check_slice(data: &[Limb]) {
    let width = data.len();
    let mut expected = vec![0_usize; width];
    let mut expected_borrow = false;
    // Ordinary subtraction visits every digit without a zero-prefix shortcut.
    for (digit, &source) in expected.iter_mut().zip(data) {
        let (difference, first) = digit.overflowing_sub(source);
        let (value, second) = difference.overflowing_sub(Limb::from(expected_borrow));
        *digit = value;
        expected_borrow = first || second;
    }
    let mut actual = vec![37; width + 2];
    let actual_borrow = SsaRing::neg_slice_into(&mut actual[1..=width], data);
    assert_eq!(&actual[1..=width], expected);
    assert_eq!(actual_borrow, expected_borrow);
    assert_eq!((actual[0], actual[width + 1]), (37, 37));
}

fn check_ring(data: &[Limb], guard: Limb) {
    let ml = data.len();
    let expected = oracle_negation(data, guard);
    let mut actual = data.to_vec();
    actual.extend_from_slice(&[guard, 37, Limb::MAX]);
    // SAFETY: the nonempty initialized data and one-bit guard form a complete
    // coefficient. Both suffix sentinels lie beyond the ring span.
    unsafe {
        SsaRing::negate(&mut actual, ml * LIMB_BITS);
    }
    assert_eq!(&actual[..=ml], expected);
    assert_eq!(&actual[ml + 1..], &[37, Limb::MAX]);
    assert!(actual[ml] <= 1);
    assert!(actual[ml] == 0 || actual[..ml].iter().all(|&limb| limb == 0));
}
