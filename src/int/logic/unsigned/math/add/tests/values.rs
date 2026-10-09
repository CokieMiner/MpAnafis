//! Owned, in-place, and fused arithmetic against independent limb recurrences.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Fixtures contain at most 257 limbs; capacities, indices, and u128 sums remain representable on every supported target."
)]

use alloc::vec::Vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
fn arithmetic_variants_match_column_recurrences_and_preserve_storage() {
    let widths: &[usize] = if cfg!(miri) {
        &[0, 1, 4, 5]
    } else {
        &[0, 1, 3, 4, 5, 7, 8, 9, 17, 65]
    };
    for &left_width in widths {
        for &right_width in widths {
            for fill in [1, Limb::MAX] {
                check_arithmetic(
                    &InternalMpUint::from_limbs(alloc::vec![fill; left_width]),
                    &InternalMpUint::from_limbs(alloc::vec![fill; right_width]),
                );
            }
        }
    }
    for &width in if cfg!(miri) {
        &[1, 4, 5][..]
    } else {
        &[1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 257][..]
    } {
        let all_ones = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        check_arithmetic(&all_ones, &InternalMpUint::one());
        let power = InternalMpUint::power_of_two(width * LIMB_BITS);
        check_arithmetic(&power, &InternalMpUint::one());
        for stop in 0..width {
            for absorber in [1, 2, Limb::MAX] {
                let mut words = alloc::vec![0; width];
                *words.last_mut().expect("nonempty fixture") = 1;
                *words.get_mut(stop).expect("position below width") = absorber;
                let value = InternalMpUint::from_limbs(words);
                let expected = value.sub(&InternalMpUint::one());
                for capacity in [width, width + 8] {
                    let mut output = InternalMpUint::with_capacity(capacity);
                    output.clone_from(&value);
                    let allocation = output.limbs().as_ptr();
                    let retained_capacity = output.capacity();
                    output.decrement();
                    assert_eq!(output, expected);
                    assert_eq!(output.capacity(), retained_capacity);
                    assert_eq!(output.limbs().as_ptr(), allocation);
                }
            }
        }
    }
    let limit = if cfg!(miri) { 5 } else { 80 };
    let operands = (
        collection::vec(any::<Limb>(), 0..=limit),
        collection::vec(any::<Limb>(), 0..=limit),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&operands, |(left, right)| {
            check_arithmetic(
                &InternalMpUint::from_limbs(left),
                &InternalMpUint::from_limbs(right),
            );
            Ok(())
        })
        .expect("arithmetic property");
}

pub fn check_arithmetic(a: &InternalMpUint, b: &InternalMpUint) {
    let width = a.limbs().len().max(b.limbs().len());
    let capacity = width + 8;
    let mask = u128::try_from(Limb::MAX).expect("limb fits u128");
    let mut sum_words = Vec::with_capacity(width + 1);
    let mut difference_words = Vec::with_capacity(width);
    let mut carry = 0_u128;
    let mut borrow = false;
    for index in 0..width {
        let left = a.limbs().get(index).copied().unwrap_or(0);
        let right = b.limbs().get(index).copied().unwrap_or(0);
        let total = u128::try_from(left).expect("limb fits")
            + u128::try_from(right).expect("limb fits")
            + carry;
        sum_words.push(Limb::try_from(total & mask).expect("masked limb"));
        carry = total >> LIMB_BITS;
        let (difference, first) = left.overflowing_sub(right);
        let (residue, second) = difference.overflowing_sub(Limb::from(borrow));
        difference_words.push(residue);
        borrow = first || second;
    }
    if carry != 0 {
        sum_words.push(1);
    }
    let sum = InternalMpUint::from_limbs(sum_words);
    let difference = InternalMpUint::from_limbs(difference_words);
    assert_eq!(a.add(b), sum);
    assert_eq!(b.add(a), sum);
    assert_eq!(a.sub_with_underflow(b), (difference.clone(), borrow));
    check_outputs(a, b, &sum, &difference, borrow, capacity);
    assert!(sum.limbs().last().is_none_or(|&top| top != 0));
    assert!(difference.limbs().last().is_none_or(|&top| top != 0));
}

fn check_outputs(
    a: &InternalMpUint,
    b: &InternalMpUint,
    sum: &InternalMpUint,
    difference: &InternalMpUint,
    borrow: bool,
    capacity: usize,
) {
    let width = a.limbs().len().max(b.limbs().len());
    for old_width in [0, 1, 4, 5, width + 2] {
        let dirty = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; old_width]);
        for mut output in [dirty.clone(), InternalMpUint::with_capacity(capacity)] {
            output.clone_from(&dirty);
            output.assign_sum(a, b);
            assert_eq!(&output, sum);
            assert_eq!(output.assign_difference(a, b), borrow);
            assert_eq!(&output, difference);
            assert!(!output.assign_difference(a, a));
            assert!(output.is_zero());
        }
    }
    for mut output in [a.clone(), InternalMpUint::with_capacity(capacity)] {
        output.clone_from(a);
        output.add_assign(b);
        assert_eq!(&output, sum);
        output.clone_from(a);
        assert_eq!(output.sub_assign_with_underflow(b), borrow);
        assert_eq!(&output, difference);
        if !borrow {
            output.clone_from(a);
            let allocation = output.limbs().as_ptr();
            let retained_capacity = output.capacity();
            output.sub_assign(b);
            assert_eq!(&output, difference);
            assert_eq!(output.capacity(), retained_capacity);
            assert_eq!(output.limbs().as_ptr(), allocation);
            assert_eq!(&a.sub(b), difference);
        }
    }
    check_retained_storage(a, b, sum, difference, borrow, capacity);
}

fn check_retained_storage(
    a: &InternalMpUint,
    b: &InternalMpUint,
    sum: &InternalMpUint,
    difference: &InternalMpUint,
    borrow: bool,
    capacity: usize,
) {
    let mut retained = InternalMpUint::with_capacity(capacity);
    retained.clone_from(a);
    let allocation = retained.limbs().as_ptr();
    let retained_capacity = retained.capacity();
    retained.add_assign(b);
    assert_eq!(&retained, sum);
    assert_eq!(retained.limbs().as_ptr(), allocation);
    assert_eq!(retained.capacity(), retained_capacity);
    assert_eq!(retained.assign_difference(a, b), borrow);
    assert_eq!(&retained, difference);
    assert_eq!(retained.limbs().as_ptr(), allocation);
    assert_eq!(retained.capacity(), retained_capacity);
    if !a.is_zero() {
        retained.clone_from(a);
        retained.decrement();
        assert_eq!(retained, a.sub(&InternalMpUint::one()));
        retained.increment();
        assert_eq!(&retained, a);
    }
}
