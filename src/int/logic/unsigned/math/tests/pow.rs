//! Integer-power bounds, arithmetic, and sliding windows.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Exponent widths are at most 4096 bits and fixture powers at most 129; u128 bounds cover every supported pointer width."
)]

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert_eq, proptest,
};

use crate::int::logic::unsigned::math::pow::limb_bound;

use super::super::{Exponentiation, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn powers_match_repeated_products_and_primitive_results() {
    for native in [0_u128, 1, 2, 3, 7, 255, u128::MAX] {
        let base = InternalMpUint::from_u128(native);
        let mut reference = InternalMpUint::one();
        for exponent in 0..=8 {
            assert_eq!(base.pow(exponent), reference);
            if let Some(expected) = native.checked_pow(exponent) {
                assert_eq!(reference.to_u128(), Some(expected));
            }
            reference = reference.mul(&base);
        }
    }
    for bit in [
        1,
        LIMB_BITS - 1,
        LIMB_BITS,
        4 * LIMB_BITS,
        5 * LIMB_BITS + 1,
    ] {
        let base = InternalMpUint::power_of_two(bit);
        for exponent in [0, 1, 2, 3, 4, 7, 31, 64, 129] {
            let degree = usize::try_from(exponent).expect("fixture degree fits usize");
            assert_eq!(
                base.pow(exponent),
                InternalMpUint::power_of_two(bit * degree)
            );
        }
    }
}

#[test]
fn capacity_bounds_and_window_selection_cover_width_transitions() {
    for bits in [0, 1, 2, LIMB_BITS - 1, LIMB_BITS, LIMB_BITS + 1, usize::MAX] {
        for exponent in [0, 1, 2, 127, 128, 129, u32::MAX] {
            let expected = if bits == 0 || exponent == 0 {
                Some(0)
            } else if usize::try_from(exponent).is_err() {
                None
            } else {
                let bound = (u128::try_from(bits).expect("usize fits u128") * u128::from(exponent))
                    .div_ceil(u128::try_from(LIMB_BITS).expect("limb width fits u128"));
                usize::try_from(bound).ok()
            };
            assert_eq!(limb_bound(bits, exponent), expected);
        }
    }
    for bits in [1, 2, 6, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2048] {
        let sparse = InternalMpUint::power_of_two(bits - 1);
        let paired = sparse.add(&InternalMpUint::one());
        let dense = InternalMpUint::power_of_two(bits).sub(&InternalMpUint::one());
        assert_eq!(Exponentiation::window_plan::<false>(&sparse, bits).width, 1);
        assert_eq!(
            Exponentiation::window_plan::<false>(&paired, paired.significant_bits()).width,
            1
        );
        assert_minimum_window_work(&dense);
    }
}

#[test]
fn isolated_set_bits_use_binary_without_an_odd_power_table() {
    for bits in [7, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2048] {
        let exponent = InternalMpUint::power_of_two(bits - 1)
            .add(&InternalMpUint::power_of_two(bits.div_euclid(2)))
            .add(&InternalMpUint::one());
        let plan = Exponentiation::window_plan::<false>(&exponent, bits);
        assert_eq!((plan.width, plan.powers, plan.initial), (1, 1, (0, 1)));
        assert_minimum_window_work(&exponent);
    }
}

#[test]
fn exhaustive_short_exponents_minimize_arithmetic_and_initialize_every_digit() {
    for exponent in 1..=if cfg!(miri) { 127 } else { 4095 } {
        let value = InternalMpUint::from_u64(exponent);
        assert_minimum_window_work(&value);
        assert_fixed_window_bounds(&value);
    }
}

#[test]
fn window_plans_cover_zero_limbs_and_partial_native_boundaries() {
    for bit in [LIMB_BITS - 1, LIMB_BITS, LIMB_BITS + 1, 4 * LIMB_BITS, 2047] {
        for trailing in [1, 3, 5, 7, 15, 31, 63] {
            let exponent =
                InternalMpUint::power_of_two(bit).add(&InternalMpUint::from_limb(trailing));
            assert_minimum_window_work(&exponent);
        }
    }
}

#[test]
fn fixed_window_admissions_match_rational_scores_at_population_boundaries() {
    for ones in [
        3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 32, 33, 47, 48, 49, 159, 160, 161, 479, 480,
        481,
    ] {
        for gap in 1..Exponentiation::MAX_WINDOW {
            let mut exponent = InternalMpUint::zero();
            for position in (0..gap * ones).step_by(gap) {
                exponent.add_assign(&InternalMpUint::power_of_two(position));
            }
            assert_fixed_window_bounds(&exponent);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn windows_match_bit_extraction_across_limb_boundaries(
        words in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 2 } else { 8 }),
    ) {
        let exponent = InternalMpUint::from_limbs(words);
        for bits in 1..=exponent.significant_bits() {
            if !exponent.get_bit(bits - 1) { continue; }
            for maximum in 1..=6 {
                let width = maximum.min(bits);
                let mut digit = 0_usize;
                for offset in 0..width {
                    digit |= usize::from(exponent.get_bit(bits - width + offset)) << offset;
                }
                let zeros = usize::try_from(digit.trailing_zeros()).expect("at most five zero bits");
                prop_assert_eq!(Exponentiation::window(exponent.limbs(), bits, maximum), (digit >> (zeros + 1), width - zeros));
            }
        }
    }

    #[test]
    fn window_plans_match_independent_set_bit_grouping(
        words in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 2 } else { 32 }),
    ) {
        let exponent = InternalMpUint::from_limbs(words);
        if !exponent.is_zero() {
            assert_minimum_window_work(&exponent);
            assert_fixed_window_bounds(&exponent);
        }
    }
}

fn assert_minimum_window_work(exponent: &InternalMpUint) {
    let bits = exponent.significant_bits();
    let plan = Exponentiation::window_plan::<false>(exponent, bits);
    let mut expected = None;
    for width in 1..=Exponentiation::MAX_WINDOW.min(bits) {
        let (squares, products, powers, initial) = reference_window_work(exponent, width);
        let candidate = (squares + products, width, powers, initial);
        if expected.is_none_or(|incumbent| candidate < incumbent) {
            expected = Some(candidate);
        }
    }
    let (_, width, powers, initial) = expected.expect("positive exponent has a binary plan");
    assert_eq!(
        (plan.width, plan.powers, plan.initial),
        (width, powers, initial),
        "exponent={exponent:?}"
    );
    assert!((1..=Exponentiation::ODD_POWER_CAPACITY).contains(&plan.powers));
}

fn assert_fixed_window_bounds(exponent: &InternalMpUint) {
    let bits = exponent.significant_bits();
    let positions: alloc::vec::Vec<_> = (0..bits)
        .filter(|&position| exponent.get_bit(position))
        .collect();
    let gap = positions
        .windows(2)
        .map(|pair| pair.last().expect("two positions") - pair.first().expect("two positions"))
        .min()
        .unwrap_or(Exponentiation::MAX_WINDOW)
        .min(Exponentiation::MAX_WINDOW);
    let ones = positions.len();
    let mut expected_width = 1;
    let mut expected_powers = 1;
    let mut numerator = ones - 1;
    let mut denominator = 1;
    if ones > 3 {
        for width in 2..=Exponentiation::MAX_WINDOW.min(bits) {
            let mut largest = 0;
            let mut population = 1;
            for digit in (1_usize..(1 << width)).step_by(2) {
                let set_bits: alloc::vec::Vec<_> =
                    (0..width).filter(|&bit| digit & (1 << bit) != 0).collect();
                if set_bits.windows(2).any(|pair| {
                    pair.last().expect("two positions") - pair.first().expect("two positions") < gap
                }) {
                    continue;
                }
                largest = largest.max(digit >> 1);
                population = population.max(set_bits.len());
            }
            if population == 1 {
                continue;
            }
            let predicted = ones + (largest - 1) * population;
            if predicted * denominator < numerator * population {
                numerator = predicted;
                denominator = population;
                expected_width = width;
                expected_powers = largest + 1;
            }
        }
    }
    let plan = Exponentiation::window_plan::<true>(exponent, bits);
    let (_, _, needed_powers, initial) = reference_window_work(exponent, plan.width);
    assert_eq!((plan.width, plan.powers), (expected_width, expected_powers));
    assert_eq!(plan.initial, initial);
    assert!(
        needed_powers <= plan.powers,
        "every referenced digit is prepared"
    );
    assert!(plan.powers <= Exponentiation::ODD_POWER_CAPACITY);
}

/// Groups descending set-bit positions; no production extraction or zero scan.
fn reference_window_work(
    exponent: &InternalMpUint,
    width: usize,
) -> (usize, usize, usize, (usize, usize)) {
    let bits = exponent.significant_bits();
    let mut positions = (0..bits)
        .rev()
        .filter(|&position| exponent.get_bit(position))
        .peekable();
    let mut groups = 0;
    let mut largest = 0;
    let mut initial = (0, 0);
    while let Some(start) = positions.next() {
        let mut last = start;
        let mut digit = 1;
        while let Some(&position) = positions.peek() {
            if start - position >= width {
                break;
            }
            digit = (digit << (last - position)) | 1;
            last = positions.next().expect("peeked set bit exists");
        }
        let index = digit >> 1;
        if groups == 0 {
            initial = (index, start - last + 1);
        }
        largest = largest.max(index);
        groups += 1;
    }
    (
        bits - initial.1 + usize::from(largest != 0),
        groups - 1 + largest,
        largest + 1,
        initial,
    )
}
