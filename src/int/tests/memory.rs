//! Capacity changes, storage reuse, and assignment rollback contracts.

extern crate std;

use core::{
    ops::{AddAssign, Shl, Shr},
    panic::AssertUnwindSafe,
};
use std::panic::catch_unwind;

use alloc::vec;

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, Precision};

use super::{
    strategies,
    support::{nz, uint_from_words},
};

proptest! {
    #[test]
    fn reserve_and_shrink_preserve_integer_contracts(
        unsigned_seed in strategies::uint(8), signed_seed in strategies::int(8),
        extra in 0_usize..=20, bounded in any::<bool>(),
        left in any::<usize>(), right in any::<usize>(),
    ) {
        let unsigned = if bounded {
            MpUint::with_precision_checked(unsigned_seed, nz(1024)).expect("test magnitude fits")
        } else { unsigned_seed };
        let signed = if bounded {
            MpInt::with_precision_checked(signed_seed, nz(1024)).expect("test magnitude fits")
        } else { signed_seed };
        let mut u = unsigned.clone();
        let mut i = signed.clone();
        let unsigned_len = u.significant_bits().div_ceil(usize::BITS as usize);
        let signed_len = i.significant_bits().div_ceil(usize::BITS as usize);
        for exact in [false, true] {
            let old_unsigned_capacity = u.capacity();
            let old_signed_capacity = i.capacity();
            if exact {
                u.reserve_exact(extra);
                i.reserve_exact(extra);
            } else {
                u.reserve(extra);
                i.reserve(extra);
            }
            prop_assert!(u.capacity() >= unsigned_len + extra && u.capacity() >= old_unsigned_capacity);
            prop_assert!(i.capacity() >= signed_len + extra && i.capacity() >= old_signed_capacity);
            prop_assert_eq!(&u, &unsigned);
            prop_assert_eq!(&i, &signed);
            prop_assert_eq!(u.precision(), unsigned.precision());
            prop_assert_eq!(i.precision(), signed.precision());
            let before_unsigned_shrink = u.capacity();
            let before_signed_shrink = i.capacity();
            u.shrink_to_fit();
            i.shrink_to_fit();
            prop_assert!((unsigned_len..=before_unsigned_shrink).contains(&u.capacity()));
            prop_assert!((signed_len..=before_signed_shrink).contains(&i.capacity()));
            prop_assert_eq!(&u, &unsigned);
            prop_assert_eq!(&i, &signed);
            prop_assert_eq!(u.precision(), unsigned.precision());
            prop_assert_eq!(i.precision(), signed.precision());
        }

        let width = nz(usize::BITS as usize + 1);
        let mut borrowed = MpUint::with_precision_checked(left, width).expect("native word fits");
        let mut owned = borrowed.clone();
        let source = MpUint::from(right);
        AddAssign::add_assign(&mut borrowed, &source);
        AddAssign::add_assign(&mut owned, source);
        let expected = (left as u128) + (right as u128);
        prop_assert_eq!(borrowed.to_u128(), Some(expected));
        prop_assert_eq!(owned.to_u128(), Some(expected));
        prop_assert_eq!(borrowed.precision(), Precision::Bounded(width));
        prop_assert_eq!(owned.precision(), Precision::Bounded(width));
    }
}

#[test]
fn heap_assignments_reuse_capacity_and_bounded_shift_failures_preserve_receivers() {
    for length in [5_usize, 8, 64] {
        let width = nz(length * usize::BITS as usize);
        let expected = uint_from_words(&vec![3_usize; length]);
        for owned in [false, true] {
            let mut destination =
                MpUint::with_precision_checked(uint_from_words(&vec![1_usize; length]), width)
                    .expect("value fits");
            let source = uint_from_words(&vec![2_usize; length]);
            let capacity = destination.capacity();
            if owned {
                destination += source;
            } else {
                destination += &source;
            }
            assert_eq!(destination, expected);
            assert_eq!(destination.precision(), Precision::Bounded(width));
            assert_eq!(destination.capacity(), capacity);
        }
    }

    let width = nz(8192);
    let magnitude = uint_from_words(&vec![1_usize; 64]);
    let mut unsigned =
        MpUint::with_precision_checked(magnitude.clone(), width).expect("value fits");
    let mut signed =
        MpInt::with_precision_checked(-MpInt::from(magnitude), width).expect("value fits");
    unsigned.reserve_exact(32);
    signed.reserve_exact(32);
    let original_unsigned = unsigned.clone();
    let original_signed = signed.clone();
    unsigned = Shl::shl(unsigned, 3_u8);
    unsigned = Shr::shr(unsigned, 3_u8);
    unsigned <<= 3_u8;
    unsigned >>= 3_u8;
    signed = Shl::shl(signed, 3_u8);
    signed = Shr::shr(signed, 3_u8);
    signed <<= 3_u8;
    signed >>= 3_u8;
    assert_eq!(unsigned, original_unsigned);
    assert_eq!(signed, original_signed);
    assert_eq!(unsigned.capacity(), 96);
    assert_eq!(signed.capacity(), 96);
    assert_eq!(unsigned.precision(), Precision::Bounded(width));
    assert_eq!(signed.precision(), Precision::Bounded(width));

    for bits in [1, 2, 8, 64, 65, 256] {
        for shift in [1_usize, usize::MAX] {
            let mut unsigned_maximum = MpUint::max_for_precision(bits);
            let original_maximum = unsigned_maximum.clone();
            assert!(catch_unwind(AssertUnwindSafe(|| unsigned_maximum <<= shift)).is_err());
            assert_eq!(unsigned_maximum, original_maximum);
            assert_eq!(unsigned_maximum.precision(), original_maximum.precision());
            let mut signed_minimum = MpInt::min_for_precision(bits);
            let original_minimum = signed_minimum.clone();
            assert!(catch_unwind(AssertUnwindSafe(|| signed_minimum <<= shift)).is_err());
            assert_eq!(signed_minimum, original_minimum);
            assert_eq!(signed_minimum.precision(), original_minimum.precision());
        }
        let zero_width = nz(bits);
        let mut unsigned_zero = MpUint::zero_with_precision(zero_width);
        unsigned_zero <<= usize::MAX;
        assert!(unsigned_zero.is_zero());
        assert_eq!(unsigned_zero.precision(), Precision::Bounded(zero_width));
        let mut signed_zero = MpInt::zero_with_precision(zero_width);
        signed_zero <<= usize::MAX;
        assert!(signed_zero.is_zero());
        assert_eq!(signed_zero.precision(), Precision::Bounded(zero_width));
    }
}
