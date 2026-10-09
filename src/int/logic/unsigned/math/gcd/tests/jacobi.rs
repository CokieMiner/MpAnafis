//! Jacobi signs against full-precision binary reciprocity.

use core::mem::swap;

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert_eq, proptest,
};

use crate::int::logic::unsigned::math::gcd::jacobi::jacobi_2;

use super::{InternalMpUint, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 192 }))]

    #[test]
    fn scalar_two_limb_and_wide_batches_match_binary_reciprocity(
        first_low in any::<Limb>(), first_high in 1_usize..=Limb::MAX,
        second_low in any::<Limb>(), second_high in 1_usize..=Limb::MAX,
        first in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 80 }),
        second in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 5 } else { 80 }),
        words in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 5 } else { 129 }),
    ) {
        let scalar = InternalMpUint::from_limb(first_low);
        let odd_scalar = InternalMpUint::from_limb(second_low | 1);
        prop_assert_eq!(scalar.jacobi_symbol(&odd_scalar), binary_reference(&scalar, &odd_scalar));

        let first_pair = InternalMpUint::from_limbs_2(first_low, first_high);
        let second_pair = InternalMpUint::from_limbs_2(second_low | 1, second_high);
        let pair_sign = binary_reference(&first_pair, &second_pair);
        prop_assert_eq!(first_pair.jacobi_symbol(&second_pair), pair_sign);
        prop_assert_eq!(jacobi_2([first_low, first_high], [second_low | 1, second_high]), pair_sign);

        let numerator = InternalMpUint::from_limbs(first);
        let mut denominator = InternalMpUint::from_limbs(second);
        if denominator.is_even() {
            denominator.increment();
        }
        prop_assert_eq!(numerator.jacobi_symbol(&denominator), binary_reference(&numerator, &denominator));
        if numerator.is_odd() {
            prop_assert_eq!(denominator.jacobi_symbol(&numerator), binary_reference(&denominator, &numerator));
        }

        let wide = InternalMpUint::from_limbs(words);
        prop_assert_eq!(wide.jacobi_symbol(&odd_scalar), binary_reference(&wide, &odd_scalar));
        let mut odd_wide = wide.clone();
        if odd_wide.is_even() {
            odd_wide.increment();
        }
        prop_assert_eq!(scalar.jacobi_symbol(&odd_wide), binary_reference(&scalar, &odd_wide));
        prop_assert_eq!(wide.mul(&odd_scalar).jacobi_symbol(&odd_scalar), i8::from(odd_scalar.is_one()));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    #[cfg_attr(miri, ignore = "Recursive 120-to-520-limb Jacobi blocks require native execution; scalar and wide batch properties run under Miri.")]
    fn recursive_blocks_and_rejected_windows_match_binary_reciprocity(
        first in proptest::collection::vec(any::<Limb>(), 120..=520),
        second in proptest::collection::vec(any::<Limb>(), 120..=520),
    ) {
        let numerator = InternalMpUint::from_limbs(first);
        let mut denominator = InternalMpUint::from_limbs(second);
        if denominator.is_even() {
            denominator.increment();
        }
        prop_assert_eq!(numerator.jacobi_symbol(&denominator), binary_reference(&numerator, &denominator));
        let near = denominator.add(&InternalMpUint::from_limb(6));
        prop_assert_eq!(near.jacobi_symbol(&denominator), binary_reference(&near, &denominator));
    }
}

#[test]
fn scalar_tails_cover_all_small_pairs() {
    let limit = if cfg!(miri) { 16 } else { 256 };
    for numerator in 0..limit {
        for denominator in (1..limit).step_by(2) {
            let a = InternalMpUint::from_limb(numerator);
            let b = InternalMpUint::from_limb(denominator);
            assert_eq!(a.jacobi_symbol(&b), binary_reference(&a, &b));
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Exact quotient and shared-prefix matrices through 513 limbs require native execution; the broad reciprocity property covers small batches under Miri."
)]
fn shared_prefixes_and_exact_quotient_boundaries_preserve_signs() {
    for width in [2_usize, 5, 64, 123] {
        for denominator_low in [1, 3, 5, (Limb::MAX >> 1) | 1, Limb::MAX] {
            let denominator = InternalMpUint::from_limbs(
                core::iter::once(denominator_low)
                    .chain(core::iter::repeat_n(Limb::MAX, width - 1))
                    .collect(),
            );
            for numerator_low in [0, 1, 2, 4, 6, 16, Limb::MAX] {
                let numerator = InternalMpUint::from_limbs(
                    core::iter::once(numerator_low)
                        .chain(core::iter::repeat_n(Limb::MAX, width - 1))
                        .collect(),
                );
                assert_eq!(
                    numerator.jacobi_symbol(&denominator),
                    binary_reference(&numerator, &denominator)
                );
            }
        }
    }
    for width in [
        2, 3, 4, 5, 23, 24, 25, 63, 64, 65, 122, 123, 124, 177, 178, 179, 511, 512, 513,
    ] {
        let denominator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for quotient in [1, 2, 3, 4, 5, Limb::MAX] {
            let product = denominator.mul(&InternalMpUint::from_limb(quotient));
            for offset in [0, 1, 2, 3, 4, 8, 16] {
                let numerator = product.add(&InternalMpUint::from_limb(offset));
                assert_eq!(
                    numerator.jacobi_symbol(&denominator),
                    binary_reference(&numerator, &denominator)
                );
            }
        }
        for shift in [0, 1, 2, 7, 15, 31, 63, 64, 127] {
            let numerator = InternalMpUint::from_limb(3).shl(shift);
            assert_eq!(
                numerator.jacobi_symbol(&denominator),
                binary_reference(&numerator, &denominator)
            );
        }
    }
}

/// Binary reciprocity using exact Euclidean remainders.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "The accumulated Jacobi sign is always -1 or 1, so negation fits i8."
)]
fn binary_reference(value: &InternalMpUint, modulus: &InternalMpUint) -> i8 {
    let mut numerator = value.rem(modulus);
    let mut denominator = modulus.clone();
    let mut sign = 1_i8;
    while !numerator.is_zero() {
        let twos = numerator.trailing_zeros();
        numerator.shr_assign(twos);
        if twos & 1 != 0 && denominator.get_bit(1) != denominator.get_bit(2) {
            sign = -sign;
        }
        if numerator.get_bit(1) && denominator.get_bit(1) {
            sign = -sign;
        }
        swap(&mut numerator, &mut denominator);
        numerator.rem_assign(&denominator);
    }
    if denominator.is_one() { sign } else { 0 }
}
