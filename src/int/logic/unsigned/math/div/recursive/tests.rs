//! Recursive upper quotient error and untouched input, output, and product guards.

use core::mem::MaybeUninit;

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, Limb, QuotientAccuracy};

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
            let accuracy = Division::burnikel_div_approximate(
                window, divisor.limbs(), quotient, work, &mut scratch.mul_scratch,
            );
            let result = InternalMpUint::from_limbs(quotient.to_vec());
            prop_assert!(result >= expected);
            prop_assert!(result < limit);
            if accuracy == QuotientAccuracy::Exact {
                prop_assert_eq!(&result, &expected);
            }
            prop_assert_eq!((input.first(), input.last()), (Some(&19), Some(&19)));
            prop_assert_eq!((output.first(), output.last()), (Some(&23), Some(&23)));
            prop_assert_eq!((*low_guard, *high_guard), (29, 29));
        }
    }
}
