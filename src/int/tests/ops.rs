//! Operator ownership and assignment contracts across inline and heap values.

extern crate std;

use core::{
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    panic::AssertUnwindSafe,
};
use std::panic::catch_unwind;

use proptest::prelude::{prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

use super::{strategies, support::nz};

macro_rules! check_binary_paths {
    ($left:expr, $right:expr, $expected:expr, $binary:ident, $assign:ident, $method:ident, $assign_method:ident) => {{
        let left = &$left;
        let right = &$right;
        let expected = $expected;
        prop_assert_eq!($binary::$method(left, right), expected.clone());
        prop_assert_eq!($binary::$method(left.clone(), right), expected.clone());
        prop_assert_eq!($binary::$method(left, right.clone()), expected.clone());
        prop_assert_eq!(
            $binary::$method(left.clone(), right.clone()),
            expected.clone()
        );
        let mut borrowed = left.clone();
        $assign::$assign_method(&mut borrowed, right);
        prop_assert_eq!(&borrowed, &expected);
        prop_assert_eq!(borrowed.precision(), left.precision());
        let mut owned = left.clone();
        $assign::$assign_method(&mut owned, right.clone());
        prop_assert_eq!(&owned, &expected);
        prop_assert_eq!(owned.precision(), left.precision());
    }};
}

macro_rules! check_shift_paths {
    ($value:expr, $shift:expr, $($count_type:ty),+) => {$(
        for count in <$count_type>::try_from($shift).into_iter() {
            let value = &$value;
            let left = value << $shift;
            let right = value >> $shift;
            prop_assert_eq!(Shl::shl(value, count), left.clone());
            prop_assert_eq!(Shl::shl(value.clone(), count), left.clone());
            prop_assert_eq!(Shr::shr(value, count), right.clone());
            prop_assert_eq!(Shr::shr(value.clone(), count), right.clone());
            let mut assigned_left = value.clone();
            let mut assigned_right = value.clone();
            ShlAssign::shl_assign(&mut assigned_left, count);
            ShrAssign::shr_assign(&mut assigned_right, count);
            prop_assert_eq!(&assigned_left, &left);
            prop_assert_eq!(&assigned_right, &right);
            prop_assert_eq!(assigned_left.precision(), value.precision());
            prop_assert_eq!(assigned_right.precision(), value.precision());
        }
    )+};
}

proptest! {
    #[test]
    fn unsigned_operator_paths_preserve_results(
        a in strategies::uint(32), b in strategies::uint(32),
        shift in 0_usize..=266, additional in 1_usize..=8,
    ) {
        check_binary_paths!(a, b, a.checked_add(&b).expect("unlimited addition"), Add, AddAssign, add, add_assign);
        check_binary_paths!(a, b, a.checked_mul(&b).expect("unlimited multiplication"), Mul, MulAssign, mul, mul_assign);
        check_binary_paths!(a, b, &a & &b, BitAnd, BitAndAssign, bitand, bitand_assign);
        check_binary_paths!(a, b, &a | &b, BitOr, BitOrAssign, bitor, bitor_assign);
        check_binary_paths!(a, b, &a ^ &b, BitXor, BitXorAssign, bitxor, bitxor_assign);
        if let Some(difference) = a.checked_sub(&b) {
            check_binary_paths!(a, b, difference, Sub, SubAssign, sub, sub_assign);
        }
        if !b.is_zero() {
            check_binary_paths!(a, b, a.checked_div(&b).expect("nonzero divisor"), Div, DivAssign, div, div_assign);
            check_binary_paths!(a, b, a.checked_rem(&b).expect("nonzero divisor"), Rem, RemAssign, rem, rem_assign);
        }
        let mut reserved_rhs = b.clone();
        reserved_rhs.reserve(additional);
        prop_assert_eq!(Add::add(a.clone(), reserved_rhs.clone()), &a + &b);
        let mut destination = a.clone();
        destination += reserved_rhs;
        prop_assert_eq!(destination, &a + &b);
        check_shift_paths!(a, shift, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
    }

    #[test]
    fn signed_operator_paths_preserve_results(
        a in strategies::int(32), b in strategies::int(32),
        shift in 0_usize..=266, additional in 1_usize..=8,
    ) {
        check_binary_paths!(a, b, a.checked_add(&b).expect("unlimited addition"), Add, AddAssign, add, add_assign);
        check_binary_paths!(a, b, a.checked_sub(&b).expect("unlimited subtraction"), Sub, SubAssign, sub, sub_assign);
        check_binary_paths!(a, b, a.checked_mul(&b).expect("unlimited multiplication"), Mul, MulAssign, mul, mul_assign);
        check_binary_paths!(a, b, &a & &b, BitAnd, BitAndAssign, bitand, bitand_assign);
        check_binary_paths!(a, b, &a | &b, BitOr, BitOrAssign, bitor, bitor_assign);
        check_binary_paths!(a, b, &a ^ &b, BitXor, BitXorAssign, bitxor, bitxor_assign);
        if !b.is_zero() {
            check_binary_paths!(a, b, a.checked_div(&b).expect("nonzero divisor"), Div, DivAssign, div, div_assign);
            check_binary_paths!(a, b, a.checked_rem(&b).expect("nonzero divisor"), Rem, RemAssign, rem, rem_assign);
        }
        let mut reserved_rhs = b.clone();
        reserved_rhs.reserve(additional);
        prop_assert_eq!(Add::add(a.clone(), reserved_rhs.clone()), &a + &b);
        let mut destination = a.clone();
        destination += reserved_rhs;
        prop_assert_eq!(destination, &a + &b);
        check_shift_paths!(a, shift, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
    }
}

#[test]
fn primitive_shift_counts_reject_negative_and_unrepresentable_values_before_zero_shortcuts() {
    assert_invalid_shift_count(-1_i8);
    assert_invalid_shift_count(-1_i16);
    assert_invalid_shift_count(-1_i32);
    assert_invalid_shift_count(-1_i64);
    assert_invalid_shift_count(-1_i128);
    assert_invalid_shift_count(-1_isize);
    assert_invalid_shift_count(u128::MAX);
    assert_invalid_shift_count(i128::MAX);
    if usize::try_from(u64::MAX).is_err() {
        assert_invalid_shift_count(u64::MAX);
    }
    if usize::try_from(i64::MAX).is_err() {
        assert_invalid_shift_count(i64::MAX);
    }
    if usize::try_from(u32::MAX).is_err() {
        assert_invalid_shift_count(u32::MAX);
    }
    if usize::try_from(i32::MAX).is_err() {
        assert_invalid_shift_count(i32::MAX);
    }
}

#[test]
fn division_failures_preserve_receivers_for_owned_and_borrowed_assignments() {
    for owned in [false, true] {
        let unsigned_original = MpUint::from(37_u8);
        let signed_original = MpInt::from(-37_i8);
        for remainder in [false, true] {
            let mut u = unsigned_original.clone();
            let mut i = signed_original.clone();
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    if remainder {
                        if owned {
                            u %= MpUint::zero();
                        } else {
                            u %= &MpUint::zero();
                        }
                    } else if owned {
                        u /= MpUint::zero();
                    } else {
                        u /= &MpUint::zero();
                    }
                }))
                .is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    if remainder {
                        if owned {
                            i %= MpInt::zero();
                        } else {
                            i %= &MpInt::zero();
                        }
                    } else if owned {
                        i /= MpInt::zero();
                    } else {
                        i /= &MpInt::zero();
                    }
                }))
                .is_err()
            );
            assert_eq!(u, unsigned_original);
            assert_eq!(i, signed_original);
            assert_eq!(u.precision(), unsigned_original.precision());
            assert_eq!(i.precision(), signed_original.precision());
        }
        for bits in [1, 2, 8, 64, 65, 128, 256] {
            let minimum = MpInt::min_for_precision(bits);
            let minus_one = MpInt::with_precision_checked(-1_i8, nz(bits)).expect("minus one fits");
            for remainder in [false, true] {
                let mut value = minimum.clone();
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        if remainder {
                            if owned {
                                value %= minus_one.clone();
                            } else {
                                value %= &minus_one;
                            }
                        } else if owned {
                            value /= minus_one.clone();
                        } else {
                            value /= &minus_one;
                        }
                    }))
                    .is_err()
                );
                assert_eq!(value, minimum);
                assert_eq!(value.precision(), minimum.precision());
            }
        }
    }
    let value = MpInt::from(-37_i8);
    assert!(catch_unwind(|| &value / MpInt::zero()).is_err());
    assert!(catch_unwind(|| &value / &MpInt::zero()).is_err());
    assert!(catch_unwind(|| &value % MpInt::zero()).is_err());
    assert!(catch_unwind(|| &value % &MpInt::zero()).is_err());
}

fn assert_invalid_shift_count<Count: Copy>(count: Count)
where
    MpUint: Shl<Count, Output = MpUint>
        + Shr<Count, Output = MpUint>
        + ShlAssign<Count>
        + ShrAssign<Count>,
    MpInt: Shl<Count, Output = MpInt>
        + Shr<Count, Output = MpInt>
        + ShlAssign<Count>
        + ShrAssign<Count>,
    for<'operand> &'operand MpUint: Shl<Count, Output = MpUint> + Shr<Count, Output = MpUint>,
    for<'operand> &'operand MpInt: Shl<Count, Output = MpInt> + Shr<Count, Output = MpInt>,
{
    let unsigned = MpUint::zero_with_precision(nz(8));
    let signed = MpInt::zero_with_precision(nz(8));
    let operations: [&dyn Fn(); 8] = [
        &|| drop(&unsigned << count),
        &|| drop(unsigned.clone() << count),
        &|| drop(&unsigned >> count),
        &|| drop(unsigned.clone() >> count),
        &|| drop(&signed << count),
        &|| drop(signed.clone() << count),
        &|| drop(&signed >> count),
        &|| drop(signed.clone() >> count),
    ];
    for operation in operations {
        assert!(catch_unwind(AssertUnwindSafe(operation)).is_err());
    }
    for left in [false, true] {
        let mut unsigned_destination = unsigned.clone();
        let mut signed_destination = signed.clone();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                if left {
                    unsigned_destination <<= count;
                } else {
                    unsigned_destination >>= count;
                }
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                if left {
                    signed_destination <<= count;
                } else {
                    signed_destination >>= count;
                }
            }))
            .is_err()
        );
        assert_eq!(unsigned_destination, unsigned);
        assert_eq!(signed_destination, signed);
        assert_eq!(unsigned_destination.precision(), unsigned.precision());
        assert_eq!(signed_destination.precision(), signed.precision());
    }
}
