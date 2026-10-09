//! Fused and in-place assignment, canonical signs, and retained destination storage.

use alloc::vec;

use proptest::test_runner::{Config, TestRunner};

use crate::int::{InternalMpInt, InternalMpUint, LIMB_BITS, Limb};

use super::strategies::{public, signed, small_signed};

#[test]
fn fused_and_in_place_assignments_match_borrowed_arithmetic() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(small_signed(), small_signed()),
            |((a, left), (b, right))| {
                check_assignments(&left, &right);
                let mut output = InternalMpInt::zero();
                output.assign_add(&left, &right);
                assert_eq!(
                    public(&output).to_i128(),
                    i128::from(a).checked_add(i128::from(b))
                );
                output.assign_sub(&left, &right);
                assert_eq!(
                    public(&output).to_i128(),
                    i128::from(a).checked_sub(i128::from(b))
                );
                Ok(())
            },
        )
        .expect("primitive assignments agree");
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
            ),
            |(left, right)| {
                check_assignments(&left, &right);
                check_assignments(&left, &left);
                Ok(())
            },
        )
        .expect("wide and shared-input assignments agree");
}

#[test]
fn underflow_restores_normalized_high_limbs_in_retained_heap_storage() {
    for width in [5_usize, 8, 64] {
        let bits = width.checked_mul(LIMB_BITS).expect("small limb width");
        let capacity = width
            .checked_mul(2)
            .and_then(|size| size.checked_add(4))
            .expect("small reserve");
        for delta in [1_usize, 2, 7] {
            let mut limbs = vec![Limb::MAX; width];
            *limbs.first_mut().expect("nonempty magnitude") =
                Limb::MAX.wrapping_sub(delta.checked_sub(1).expect("positive delta"));
            for positive in [false, true] {
                let value = InternalMpInt {
                    abs: InternalMpUint::from_limbs(limbs.clone()),
                    is_positive: positive,
                };
                let mut output = InternalMpInt::with_capacity(capacity);
                let pointer = output.abs.limbs().as_ptr();
                let reserved = output.abs.capacity();
                output.assign_sub(&InternalMpInt::zero(), &value);
                assert_eq!(output.abs, value.abs);
                assert_eq!(output.is_positive, !positive);
                assert_eq!(output.abs.limbs().as_ptr(), pointer);
                assert_eq!(output.abs.capacity(), reserved);
                output.assign_add(
                    &value,
                    &InternalMpInt {
                        abs: value.abs.clone(),
                        is_positive: !positive,
                    },
                );
                assert_eq!(output, InternalMpInt::zero());
                assert_eq!(output.abs.limbs().as_ptr(), pointer);
                output.assign_mul(&value, &InternalMpInt::one());
                assert_eq!(output, value);
                output.assign_square(&InternalMpInt::min_for_bits(bits));
                assert!(output.is_positive);
                assert_eq!(output.abs.limbs().as_ptr(), pointer);
                assert_eq!(output.abs.capacity(), reserved);
            }
        }
    }
}

fn check_assignments(left: &InternalMpInt, right: &InternalMpInt) {
    let expected_sum = left.add(right);
    let expected_difference = left.sub(right);
    let expected_product = left.mul(right);
    let mut output = InternalMpInt::with_capacity(16);
    output.assign_add(left, right);
    assert_eq!(output, expected_sum);
    output.assign_sub(left, right);
    assert_eq!(output, expected_difference);
    output.assign_mul(left, right);
    assert_eq!(output, expected_product);
    output.assign_square(left);
    assert_eq!(output, left.square());
    let mut in_place = left.clone();
    in_place.add_assign(right);
    assert_eq!(in_place, expected_sum);
    in_place.clone_from(left);
    in_place.sub_assign(right);
    assert_eq!(in_place, expected_difference);
    in_place.clone_from(left);
    in_place.mul_assign(right);
    assert_eq!(in_place, expected_product);
    assert!(output.is_positive || !output.abs.is_zero());
    assert!(in_place.is_positive || !in_place.abs.is_zero());
}
