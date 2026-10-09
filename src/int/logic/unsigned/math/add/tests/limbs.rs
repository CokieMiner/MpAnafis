//! Complete initialization of raw carry and borrow spans.

#![expect(
    unsafe_code,
    reason = "Each raw test supplies aligned, disjoint spans and exposes only the prefix completely initialized by the kernel."
)]

use alloc::vec::Vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::super::{Addition, Limb};

#[test]
fn raw_kernels_match_column_recurrences_for_empty_and_initialized_spans() {
    let check = |source: &[Limb]| {
        for flag in [0, 1] {
            let mut added = Vec::with_capacity(source.len());
            let mut subtracted = Vec::with_capacity(source.len());
            let mut negated = Vec::with_capacity(source.len());
            let mut carry = flag;
            let mut borrow = flag;
            let mut negative_borrow = flag;
            for &limb in source {
                let (sum, overflow) = limb.overflowing_add(carry);
                added.push(sum);
                carry = Limb::from(overflow);
                let (difference, underflow) = limb.overflowing_sub(borrow);
                subtracted.push(difference);
                borrow = Limb::from(underflow);
                let (negative, first) = 0_usize.overflowing_sub(limb);
                let (residue, second) = negative.overflowing_sub(negative_borrow);
                negated.push(residue);
                negative_borrow = Limb::from(first || second);
            }
            let mut output = Vec::<Limb>::with_capacity(source.len());
            // SAFETY: source is initialized and disjoint from the aligned
            // allocation; its capacity covers the complete write-only span.
            let actual_carry = unsafe {
                let result = Addition::copy_tail_with_carry(
                    output.as_mut_ptr(),
                    source.as_ptr(),
                    source.len(),
                    flag,
                );
                output.set_len(source.len());
                result
            };
            assert_eq!(actual_carry, carry);
            assert_eq!(output, added);
            // SAFETY: both disjoint aligned spans cover source.len() limbs;
            // flag is binary and the kernel initializes every output element.
            let actual_borrow = unsafe {
                Addition::copy_tail_with_borrow(
                    output.as_mut_ptr(),
                    source.as_ptr(),
                    source.len(),
                    flag,
                )
            };
            assert_eq!(actual_borrow, borrow);
            assert_eq!(output, subtracted);
            output.clear();
            // SAFETY: clearing retains source.len() writable slots; the
            // disjoint source is initialized and flag is zero or one.
            let actual_negative_borrow = unsafe {
                let result = Addition::negate_with_borrow(
                    output.as_mut_ptr(),
                    source.as_ptr(),
                    source.len(),
                    flag,
                );
                output.set_len(source.len());
                result
            };
            assert_eq!(actual_negative_borrow, negative_borrow);
            assert_eq!(output, negated);
            let mut propagation = source.to_vec();
            assert_eq!(Addition::propagate_carry(&mut propagation, flag), carry);
            assert_eq!(propagation, added);
            propagation.clone_from_slice(source);
            assert_eq!(Addition::propagate_borrow(&mut propagation, flag), borrow);
            assert_eq!(propagation, subtracted);
        }
        check_slice_wrappers(source);
    };
    for width in [0, 1, 2, 4, 5, 7, 8, 9, 17] {
        for fill in [0, 1, Limb::MAX] {
            check(&alloc::vec![fill; width]);
        }
        for first in 0..width {
            for value in [1, Limb::MAX] {
                let mut words = alloc::vec![0; width];
                *words.get_mut(first).expect("position below width") = value;
                check(&words);
            }
        }
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(
            &collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 80 }),
            |source| {
                check(&source);
                Ok(())
            },
        )
        .expect("raw arithmetic property");
}

fn check_slice_wrappers(source: &[Limb]) {
    for fill in [0, Limb::MAX] {
        let mut sum = Vec::with_capacity(source.len());
        let mut difference = Vec::with_capacity(source.len());
        let (mut carry, mut borrow) = (0, 0);
        for &digit in source {
            let (partial_sum, first_carry) = fill.overflowing_add(digit);
            let (sum_digit, second_carry) = partial_sum.overflowing_add(carry);
            sum.push(sum_digit);
            carry = Limb::from(first_carry || second_carry);
            let (partial_difference, first_borrow) = fill.overflowing_sub(digit);
            let (difference_digit, second_borrow) = partial_difference.overflowing_sub(borrow);
            difference.push(difference_digit);
            borrow = Limb::from(first_borrow || second_borrow);
        }
        sum.push(7);
        difference.push(7);
        let mut destination = alloc::vec![fill; source.len()];
        destination.push(7);
        assert_eq!(
            Addition::add_slice_in_place(&mut destination, source),
            carry
        );
        assert_eq!(destination, sum);
        destination.fill(fill);
        *destination.last_mut().expect("guard slot exists") = 7;
        assert_eq!(
            Addition::sub_slice_in_place(&mut destination, source),
            borrow
        );
        assert_eq!(destination, difference);
    }
}
