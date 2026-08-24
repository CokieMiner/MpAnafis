//! Bounded overflow and signed division-domain properties.

use core::{
    fmt::Debug,
    ops::{
        Add, AddAssign, BitOrAssign, BitXorAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign,
        Sub, SubAssign,
    },
};

extern crate std;

use proptest::prelude::*;

use self::std::panic::{AssertUnwindSafe, catch_unwind};
use super::{MpInt, MpUint, bounded_int, bounded_uint, signed_max, signed_min, unsigned_max};

fn assert_panics(operation: impl FnOnce()) {
    assert!(
        catch_unwind(AssertUnwindSafe(operation)).is_err(),
        "bounded operation should panic"
    );
}

fn assert_panics_without_mutation<T>(initial: &T, operation: impl FnOnce(&mut T))
where
    T: Clone + Debug + PartialEq,
{
    let mut receiver = initial.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| operation(&mut receiver))).is_err(),
        "bounded assignment should panic"
    );
    assert_eq!(
        &receiver, initial,
        "caught panic must leave the assignment receiver unchanged"
    );
}

proptest! {
    #[test]
    fn ordinary_bounded_arithmetic_panics_on_overflow(bits in 2_usize..=63) {
        let uint_max = bounded_uint(unsigned_max(bits), bits);
        let uint_one = bounded_uint(1, bits);
        let uint_two = bounded_uint(2, bits);
        assert_panics(|| drop(Add::add(&uint_max, &uint_one)));
        assert_panics(|| drop(Mul::mul(&uint_max, &uint_two)));
        assert_panics(|| drop(Sub::sub(&uint_one, &uint_two)));

        let int_max = bounded_int(signed_max(bits), bits);
        let int_min = bounded_int(signed_min(bits), bits);
        let int_one = bounded_int(1, bits);
        let int_neg_one = bounded_int(-1, bits);
        assert_panics(|| drop(Add::add(&int_max, &int_one)));
        assert_panics(|| drop(Sub::sub(&int_min, &int_one)));
        assert_panics(|| drop(Mul::mul(&int_min, &int_neg_one)));
        assert_panics(|| drop(Div::div(&int_min, &int_neg_one)));
        assert_panics(|| drop(Rem::rem(&int_min, &int_neg_one)));
    }

    #[test]
    fn bounded_assignment_panics_instead_of_widening(bits in 2_usize..=63) {
        let uint_max = bounded_uint(unsigned_max(bits), bits);
        let uint_shift = u32::try_from(bits).expect("width fits u32");
        let uint_high_bit = MpUint::from(
            1_u128
                .checked_shl(uint_shift)
                .expect("property width is below 128")
        );
        assert_panics(|| {
            let mut value = uint_max.clone();
            AddAssign::add_assign(&mut value, MpUint::from(1_u8));
        });
        assert_panics(|| {
            let mut value = uint_max.clone();
            MulAssign::mul_assign(&mut value, MpUint::from(2_u8));
        });
        assert_panics(|| {
            let mut value = bounded_uint(0, bits);
            BitOrAssign::bitor_assign(&mut value, &uint_high_bit);
        });
        assert_panics(|| {
            let mut value = bounded_uint(0, bits);
            BitXorAssign::bitxor_assign(&mut value, &uint_high_bit);
        });

        let int_min = bounded_int(signed_min(bits), bits);
        let int_max = bounded_int(signed_max(bits), bits);
        let int_shift = u32::try_from(bits.wrapping_sub(1)).expect("width fits u32");
        let int_high_bit = MpInt::from(
            1_i128
                .checked_shl(int_shift)
                .expect("property width is below 128")
        );
        let int_neg_one = MpInt::from(-1_i8);
        assert_panics(|| {
            let mut value = int_max.clone();
            AddAssign::add_assign(&mut value, MpInt::from(1_i8));
        });
        assert_panics(|| {
            let mut value = int_min.clone();
            SubAssign::sub_assign(&mut value, MpInt::from(1_i8));
        });
        assert_panics(|| {
            let mut value = int_max.clone();
            MulAssign::mul_assign(&mut value, MpInt::from(2_i8));
        });
        assert_panics(|| {
            let mut value = bounded_int(0, bits);
            BitOrAssign::bitor_assign(&mut value, &int_high_bit);
        });
        assert_panics(|| {
            let mut value = bounded_int(0, bits);
            BitXorAssign::bitxor_assign(&mut value, &int_high_bit);
        });
        assert_panics(|| {
            let mut value = int_min.clone();
            DivAssign::div_assign(&mut value, &int_neg_one);
        });
        assert_panics(|| {
            let mut value = int_min;
            RemAssign::rem_assign(&mut value, int_neg_one);
        });
    }

    #[test]
    fn bounded_arithmetic_assignment_failure_is_transactional(bits in 2_usize..=63) {
        let uint_max = bounded_uint(unsigned_max(bits), bits);
        let uint_zero = bounded_uint(0, bits);
        let uint_one = MpUint::from(1_u8);
        let uint_two = MpUint::from(2_u8);
        let uint_zero_divisor = MpUint::from(0_u8);

        assert_panics_without_mutation(&uint_max, |receiver| {
            AddAssign::add_assign(receiver, uint_one.clone());
        });
        assert_panics_without_mutation(&uint_max, |receiver| {
            AddAssign::add_assign(receiver, &uint_one);
        });
        assert_panics_without_mutation(&uint_zero, |receiver| {
            SubAssign::sub_assign(receiver, uint_one.clone());
        });
        assert_panics_without_mutation(&uint_zero, |receiver| {
            SubAssign::sub_assign(receiver, &uint_one);
        });
        assert_panics_without_mutation(&uint_max, |receiver| {
            MulAssign::mul_assign(receiver, uint_two.clone());
        });
        assert_panics_without_mutation(&uint_max, |receiver| {
            MulAssign::mul_assign(receiver, &uint_two);
        });
        assert_panics_without_mutation(&uint_max, |receiver| {
            DivAssign::div_assign(receiver, uint_zero_divisor.clone());
        });
        assert_panics_without_mutation(&uint_max, |receiver| {
            RemAssign::rem_assign(receiver, &uint_zero_divisor);
        });

        let int_max = bounded_int(signed_max(bits), bits);
        let int_min = bounded_int(signed_min(bits), bits);
        let int_one = MpInt::from(1_i8);
        let int_two = MpInt::from(2_i8);
        let int_zero_divisor = MpInt::from(0_i8);
        let int_neg_one = MpInt::from(-1_i8);

        assert_panics_without_mutation(&int_max, |receiver| {
            AddAssign::add_assign(receiver, int_one.clone());
        });
        assert_panics_without_mutation(&int_max, |receiver| {
            AddAssign::add_assign(receiver, &int_one);
        });
        assert_panics_without_mutation(&int_min, |receiver| {
            SubAssign::sub_assign(receiver, int_one.clone());
        });
        assert_panics_without_mutation(&int_min, |receiver| {
            SubAssign::sub_assign(receiver, &int_one);
        });
        assert_panics_without_mutation(&int_max, |receiver| {
            MulAssign::mul_assign(receiver, int_two.clone());
        });
        assert_panics_without_mutation(&int_max, |receiver| {
            MulAssign::mul_assign(receiver, &int_two);
        });
        assert_panics_without_mutation(&int_max, |receiver| {
            DivAssign::div_assign(receiver, &int_zero_divisor);
        });
        assert_panics_without_mutation(&int_max, |receiver| {
            RemAssign::rem_assign(receiver, int_zero_divisor.clone());
        });
        assert_panics_without_mutation(&int_min, |receiver| {
            DivAssign::div_assign(receiver, int_neg_one.clone());
        });
        assert_panics_without_mutation(&int_min, |receiver| {
            RemAssign::rem_assign(receiver, &int_neg_one);
        });
    }
}
