//! Upper quotient error, saturation and guarded-prefix correction.

use core::mem::MaybeUninit;

use proptest::prelude::*;

use crate::int::logic::unsigned::math::div::quotient::truncated_quotient_into;

use super::{BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, Limb};

proptest! {
    #[test]
    fn recursive_upper_quotients_obey_error_and_buffer_bounds(
        mut denominator in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 3..=if cfg!(miri) { 9 } else { 321 },
        ),
        digits in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 19 } else { 1608 },
        ),
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << (Limb::BITS - 1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let n = divisor.limbs().len();
        let quotient_width = digits.len();
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let limit = expected.add(&InternalMpUint::from_limb(
            usize::try_from(Limb::BITS).expect("supported width")
                .checked_mul(2).expect("bounded recursive error"),
        ));
        let input_width = n.checked_add(quotient_width)
            .expect("bounded input width");
        let input_end = input_width.checked_add(1).expect("input sentinel offset");
        let input_capacity = input_width.checked_add(2).expect("input sentinels");
        let output_end = quotient_width.checked_add(1).expect("output sentinel offset");
        let output_capacity = quotient_width.checked_add(2).expect("output sentinels");
        let product_capacity = n.checked_add(2).expect("product sentinels");
        let mut scratch = DivScratch::default();
        for residue in [InternalMpUint::zero(), divisor.shr(1), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            let mut input = alloc::vec![19; input_capacity];
            let window = input.get_mut(1..input_end).expect("n+quotient_width input limbs");
            window.fill(0);
            window.get_mut(..numerator.limbs().len()).expect("fitting numerator")
                .copy_from_slice(numerator.limbs());
            let mut output = alloc::vec![23; output_capacity];
            let quotient = output.get_mut(1..output_end).expect("complete output limbs");
            let mut products = alloc::vec![MaybeUninit::uninit(); product_capacity];
            let (low, upper) = products.split_at_mut(1);
            let (work, high) = upper.split_at_mut(n);
            let low_guard = low.first_mut().expect("low product sentinel").write(29);
            let high_guard = high.first_mut().expect("high product sentinel").write(29);
            let error_bound = Division::burnikel_div_approximate(
                window, divisor.limbs(), quotient, work, &mut scratch.mul_scratch,
            );
            let result = InternalMpUint::from_limbs(quotient.to_vec());
            prop_assert!(result >= expected);
            prop_assert!(result < limit);
            if error_bound == 0 {
                prop_assert_eq!(&result, &expected);
            } else {
                let certified_limit = expected.add(&InternalMpUint::from_limb(error_bound));
                prop_assert!(result < certified_limit);
            }
            prop_assert_eq!((input.first(), input.last()), (Some(&19), Some(&19)));
            prop_assert_eq!((output.first(), output.last()), (Some(&23), Some(&23)));
            prop_assert_eq!((*low_guard, *high_guard), (29, 29));
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The full recursive crossover matrix uses up to 2558 quotient limbs; bounded quotient and guard properties run under Miri"
)]
fn guarded_full_quotients_handle_block_boundaries_and_exact_products() {
    let mut scratch = DivScratch::default();
    for n in [
        47, 48, 49, 63, 64, 65, 127, 128, 129, 143, 144, 145, 255, 256, 257, 511,
    ] {
        for leading in [1, Limb::MAX] {
            let mut denominator = alloc::vec![Limb::MAX; n];
            *denominator.last_mut().expect("nonempty divisor") = leading;
            let divisor = InternalMpUint::from_limbs(denominator);
            for quotient_width in [
                1,
                2,
                n - 2,
                n - 1,
                n,
                n + 1,
                n + 2,
                2 * n - 1,
                2 * n,
                2 * n + 1,
                3 * n,
                5 * n + 3,
            ] {
                let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; quotient_width]);
                let product = divisor.mul(&expected);
                let mut quotient = InternalMpUint::from_limb(7);
                Division::guarded_quotient::<true>(
                    product.limbs(),
                    divisor.limbs(),
                    &mut quotient,
                    0,
                    &mut scratch,
                );
                assert_eq!(quotient, expected);
                for residue in [
                    InternalMpUint::zero(),
                    InternalMpUint::one(),
                    divisor.shr(1),
                    divisor.sub(&InternalMpUint::one()),
                ] {
                    let numerator = product.add(&residue);
                    Division::guarded_quotient::<false>(
                        numerator.limbs(),
                        divisor.limbs(),
                        &mut quotient,
                        0,
                        &mut scratch,
                    );
                    assert_eq!(quotient, expected);
                    Division::div_into::<true, false>(
                        &numerator,
                        &divisor,
                        &mut quotient,
                        &mut scratch,
                    );
                    assert_eq!(quotient, expected);
                }
                Division::guarded_quotient::<false>(
                    product.sub(&InternalMpUint::one()).limbs(),
                    divisor.limbs(),
                    &mut quotient,
                    0,
                    &mut scratch,
                );
                assert_eq!(quotient, expected.sub(&InternalMpUint::one()));
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Production Burnikel crossovers require native execution; bounded truncated-prefix properties run under Miri"
)]
fn guarded_prefixes_correct_exact_products_and_radix_saturation() {
    let crossover = BURNIKEL_ZIEGLER_THRESHOLD * 3;
    let mut scratch = DivScratch::default();
    for n in [crossover - 1, crossover, crossover + 1, 255, 256, 257, 511] {
        for leading in [1, Limb::MAX] {
            let mut denominator = alloc::vec![Limb::MAX; n * 2 + 8];
            *denominator.last_mut().expect("nonempty divisor") = leading;
            let divisor = InternalMpUint::from_limbs(denominator);
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n]);
            let product = divisor.mul(&expected);
            for residue in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                divisor.sub(&InternalMpUint::one()),
            ] {
                let numerator = product.add(&residue);
                let mut quotient = InternalMpUint::from_limb(7);
                assert!(truncated_quotient_into::<false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut scratch,
                ));
                assert_eq!(quotient, expected);
            }
            let mut quotient = InternalMpUint::zero();
            assert!(truncated_quotient_into::<false>(
                &product.sub(&InternalMpUint::one()),
                &divisor,
                &mut quotient,
                &mut scratch,
            ));
            assert_eq!(quotient, expected.sub(&InternalMpUint::one()));
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Recursive guarded quotients reach 3000 limbs; bounded guard and prefix properties run under Miri"
)]
fn guarded_quotients_cover_one_discarded_limb_and_known_exact_prefixes() {
    let mut scratch = DivScratch::default();
    let mut quotient = InternalMpUint::zero();
    for width in [143_usize, 144, 145, 511, 1500, 3000] {
        for leading in [1, Limb::MAX] {
            let mut limbs = alloc::vec![Limb::MAX; width + 3];
            *limbs.last_mut().expect("nonempty divisor") = leading;
            let divisor = InternalMpUint::from_limbs(limbs);
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
            let product = divisor.mul(&expected);
            let span = product.limbs().len() - divisor.limbs().len() + 1;
            let split = divisor.limbs().len() - span - 1;
            assert!(split == 1 || split == 2);
            Division::guarded_quotient::<true>(
                product.limbs(),
                divisor.limbs(),
                &mut quotient,
                split,
                &mut scratch,
            );
            assert_eq!(quotient, expected);
            for numerator in [
                product.sub(&InternalMpUint::one()),
                product.add(&divisor.sub(&InternalMpUint::one())),
            ] {
                Division::div_into::<true, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut scratch,
                );
                assert_eq!(quotient, numerator.div_rem(&divisor).0);
            }
            Division::div_into::<true, true>(&product, &divisor, &mut quotient, &mut scratch);
            assert_eq!(quotient, expected);
        }
    }
}
