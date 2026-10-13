//! Reciprocal reuse and quotient repair over normalized division windows.

use proptest::prelude::*;

use super::{Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb, PreparedDivisor};

#[test]
fn reciprocal_two_by_one_covers_wrap_and_equality_boundaries() {
    let base = DoubleLimb::from(1_u8) << LIMB_BITS;
    let top: Limb = 1 << Limb::BITS.wrapping_sub(1);
    for divisor in [
        top,
        top.checked_add(1).expect("sub-maximum limb"),
        Limb::MAX,
        Limb::MAX.wrapping_sub(1),
    ] {
        let wide_divisor = DoubleLimb::try_from(divisor).expect("limb embeds in double limb");
        let reciprocal = Limb::try_from(
            DoubleLimb::MAX
                .div_euclid(wide_divisor)
                .checked_sub(base)
                .expect("normalized reciprocal"),
        )
        .expect("single-limb reciprocal");
        for high in [0, 1, divisor >> 1, divisor.wrapping_sub(1)] {
            for low in [0, 1, top, Limb::MAX.wrapping_sub(1), Limb::MAX] {
                let numerator = (DoubleLimb::try_from(high).expect("high limb") << LIMB_BITS)
                    | DoubleLimb::try_from(low).expect("low limb");
                let (quotient, remainder) =
                    Division::divrem_2by1_reciprocal(high, low, divisor, reciprocal);
                assert_eq!(
                    DoubleLimb::try_from(quotient).expect("quotient limb"),
                    numerator.div_euclid(wide_divisor)
                );
                assert_eq!(
                    DoubleLimb::try_from(remainder).expect("remainder limb"),
                    numerator.rem_euclid(wide_divisor)
                );
            }
        }
    }
}

#[test]
fn prepared_division_preserves_overflow_and_add_back() {
    let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX, 0, 1 << (Limb::BITS - 1)]);
    // D*B-1 overflows the two-limb quotient estimate. For 2D-1, the
    // leading estimate is two and the omitted low product forces an add-back.
    for digit in [1, Limb::MAX] {
        let residue = divisor.sub(&InternalMpUint::one());
        let numerator = divisor.mul(&InternalMpUint::from_limb(digit)).add(&residue);
        let mut window = numerator.limbs().to_vec();
        window.resize(4, 0);
        let mut quotient = [0];
        Division::knuth_d_divide_slice(&mut window, divisor.limbs(), &mut quotient, &mut []);
        assert_eq!(quotient, [digit]);
        assert_eq!(
            InternalMpUint::from_limbs(window.get(..3).expect("three remainder limbs").to_vec()),
            residue
        );
    }
}

#[test]
fn saturated_digit_preserves_leading_carry_and_prefix_borrow() {
    let top = 1 << (Limb::BITS - 1);
    let base = InternalMpUint::one().shl(LIMB_BITS);
    for width in [3_usize, 4, 5, 16, 17, 63, 64, 65, 127, 128, 129] {
        let middle_index = width.checked_sub(2).expect("leading divisor pair");
        for first in [1, 2, top, Limb::MAX] {
            for middle in [0, 1, top, Limb::MAX] {
                for high in [top, Limb::MAX] {
                    let mut limbs = alloc::vec![Limb::MAX; width];
                    *limbs.first_mut().expect("nonempty divisor") = first;
                    *limbs.get_mut(middle_index).expect("leading pair") = middle;
                    *limbs.last_mut().expect("leading divisor limb") = high;
                    let divisor = InternalMpUint::from_limbs(limbs);
                    let prepared = PreparedDivisor::new(divisor.limbs());
                    for low in [0, 1, top, Limb::MAX] {
                        // R=D-1 retains D's leading pair because D[0]>0.
                        // U=R*B+low gives Q=B-1 and residue D-B+low.
                        let addend = InternalMpUint::from_limb(low);
                        let numerator = divisor
                            .sub(&InternalMpUint::one())
                            .shl(LIMB_BITS)
                            .add(&addend);
                        let expected = divisor.sub(&base).add(&addend);
                        for write_quotient in [false, true] {
                            let mut window = numerator.limbs().to_vec();
                            window.resize(width.checked_add(1).expect("guard width"), 0);
                            let mut output = [7; 3];
                            let end = 1_usize
                                .checked_add(usize::from(write_quotient))
                                .expect("optional output limb");
                            prepared.divide(
                                &mut window,
                                divisor.limbs(),
                                output.get_mut(1..end).expect("guarded output span"),
                            );
                            assert_eq!(output.first(), Some(&7));
                            assert_eq!(output.last(), Some(&7));
                            assert_eq!(
                                output.get(1),
                                Some(&if write_quotient { Limb::MAX } else { 7 })
                            );
                            assert_eq!(
                                InternalMpUint::from_limbs_slice(
                                    window.get(..width).expect("complete remainder")
                                ),
                                expected
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn normalized_division_initializes_full_quotients_and_omits_unrequested_stores() {
    for width in [3_usize, 4, 5, 16, 17, 127, 128, 129] {
        let mut limbs = alloc::vec![Limb::MAX; width];
        *limbs.last_mut().expect("nonempty divisor") = 1 << (Limb::BITS - 1);
        let divisor = InternalMpUint::from_limbs(limbs);
        let residue = divisor.sub(&InternalMpUint::one());
        for digits in [alloc::vec![1, 1, 1], alloc::vec![Limb::MAX; 3]] {
            let expected = InternalMpUint::from_limbs(digits.clone());
            let numerator = divisor.mul(&expected).add(&residue);
            for write_quotient in [false, true] {
                let mut window = numerator.limbs().to_vec();
                window.resize(width.checked_add(4).expect("bounded guard span"), 0);
                let mut output = if write_quotient {
                    alloc::vec![7; 4]
                } else {
                    alloc::vec![]
                };
                Division::knuth_d_divide_slice(&mut window, divisor.limbs(), &mut output, &mut []);
                if write_quotient {
                    let mut complete = digits.clone();
                    complete.push(0);
                    assert_eq!(output, complete);
                }
                assert_eq!(
                    InternalMpUint::from_limbs(window.get(..width).expect("remainder").to_vec()),
                    residue,
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]

    #[test]
    fn reciprocal_two_by_one_matches_native_division(
        denominator in any::<Limb>(),
        high_seed in any::<Limb>(),
        low in any::<Limb>(),
    ) {
        let divisor = denominator | (1 << Limb::BITS.wrapping_sub(1));
        let high = high_seed.rem_euclid(divisor);
        let wide_divisor = DoubleLimb::try_from(divisor).expect("limb embeds in double limb");
        let base = DoubleLimb::from(1_u8) << LIMB_BITS;
        let reciprocal = Limb::try_from(DoubleLimb::MAX.div_euclid(wide_divisor).checked_sub(base).expect("normalized reciprocal")).expect("single-limb reciprocal");
        let numerator = (DoubleLimb::try_from(high).expect("high limb") << LIMB_BITS)
            | DoubleLimb::try_from(low).expect("low limb");
        let (quotient, remainder) = Division::divrem_2by1_reciprocal(high, low, divisor, reciprocal);
        prop_assert_eq!(DoubleLimb::try_from(quotient).expect("quotient limb"), numerator.div_euclid(wide_divisor));
        prop_assert_eq!(DoubleLimb::try_from(remainder).expect("remainder limb"), numerator.rem_euclid(wide_divisor));
    }

    #[test]
    fn one_prepared_reciprocal_divides_every_high_suffix(
        mut limbs in proptest::collection::vec(any::<Limb>(), 2..=129),
        digit in any::<Limb>(),
        exact in any::<bool>(),
    ) {
        *limbs.last_mut().expect("nonempty divisor") |= 1 << (Limb::BITS - 1);
        let prepared = PreparedDivisor::new(&limbs);
        for length in [2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
            if length > limbs.len() {
                continue;
            }
            let start = limbs.len().checked_sub(length).expect("suffix fits the divisor");
            let divisor = InternalMpUint::from_limbs(limbs.get(start..).expect("initialized suffix").to_vec());
            let residue = if exact { InternalMpUint::zero() } else { divisor.sub(&InternalMpUint::one()) };
            let numerator = divisor.mul(&InternalMpUint::from_limb(digit)).add(&residue);
            let mut window = numerator.limbs().to_vec();
            window.resize(length.checked_add(1).expect("test guard fits usize"), 0);
            let mut quotient = [0];
            prepared.divide(&mut window, divisor.limbs(), &mut quotient);
            prop_assert_eq!(quotient, [digit]);
            prop_assert_eq!(InternalMpUint::from_limbs(window.get(..length).expect("initialized remainder").to_vec()), residue);
        }
    }
}
