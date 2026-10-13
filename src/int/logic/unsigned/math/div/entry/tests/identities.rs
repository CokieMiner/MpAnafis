//! Value and assignment identities with retained destination storage.

use super::{Division, InternalMpUint, Limb};

#[test]
fn single_limb_outputs_and_assignments_preserve_values_and_storage() {
    for width in [0, 1, 2, 4, 5, 33, 64] {
        let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for shift in 0..Limb::BITS {
            for divisor in [1_usize << shift, Limb::MAX >> shift] {
                let denominator = InternalMpUint::from_limb(divisor);
                let mut expected = InternalMpUint::zero();
                let remainder =
                    Division::div_rem_1::<true>(numerator.limbs(), divisor, &mut expected);
                let (full_quotient, full_remainder) = numerator.div_rem(&denominator);
                assert_eq!(full_quotient, expected);
                assert_eq!(full_remainder, InternalMpUint::from_limb(remainder));
                let mut destination = InternalMpUint::with_capacity(128);
                destination.clone_from(&numerator);
                let capacity = destination.capacity();
                let pointer = destination.limbs().as_ptr();
                destination.div_assign(&denominator);
                assert_eq!(destination, expected);
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), pointer);
                destination.clone_from(&numerator);
                destination.rem_assign(&denominator);
                assert_eq!(destination, InternalMpUint::from_limb(remainder));
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), pointer);
            }
        }
    }
}

#[test]
fn value_and_assignment_identities_preserve_retained_storage() {
    let one = InternalMpUint::one();
    let zero = InternalMpUint::zero();
    let larger = InternalMpUint::from_limbs((1..=80).collect());
    for width in [0, 1, 2, 4, 5, 64] {
        let value = InternalMpUint::from_limbs((1..=width).collect());
        for (divisor, quotient, remainder) in [
            (&one, &value, &zero),
            (&larger, &zero, &value),
            (&value, &one, &zero),
        ] {
            if divisor.is_zero() {
                continue;
            }
            assert_eq!(&value.div(divisor), quotient);
            assert_eq!(&value.rem(divisor), remainder);
            let mut destination = InternalMpUint::with_capacity(128);
            destination.clone_from(&value);
            let capacity = destination.capacity();
            let pointer = destination.limbs().as_ptr();
            destination.div_assign(divisor);
            assert_eq!(&destination, quotient);
            assert_eq!(destination.capacity(), capacity);
            assert_eq!(destination.limbs().as_ptr(), pointer);
            destination.clone_from(&value);
            destination.rem_assign(divisor);
            assert_eq!(&destination, remainder);
            assert_eq!(destination.capacity(), capacity);
            assert_eq!(destination.limbs().as_ptr(), pointer);
        }
    }
}
