//! Property tests for write-complete basecase multiplication and squaring.

#![expect(
    unsafe_code,
    reason = "The raw-initializer tests reserve exact spare capacity and expose only completely written limbs"
)]

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::super::{LIMB_BITS, Limb, Schoolbook};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn prop_schoolbook_square_overwrites_dirty_destination(
        operand in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 64 }),
        dirty_seed in any::<Limb>(),
        native_left in any::<Limb>(), native_right in any::<Limb>(),
    ) {
        let native_product = u128::try_from(native_left).expect("all pointer widths fit").checked_mul(u128::try_from(native_right).expect("all pointer widths fit")).expect("two native limbs fit u128 on every target");
        let mask = u128::try_from(Limb::MAX).expect("all pointer widths fit");
        let native_expected = [Limb::try_from(native_product & mask).expect("masked limb fits"), Limb::try_from(native_product >> LIMB_BITS).expect("upper limb fits")];
        let mut native_output = [dirty_seed; 2];
        Schoolbook::mul(&mut native_output, &[native_left], &[native_right]);
        prop_assert_eq!(native_output, native_expected);
        let result_len = operand.len().checked_mul(2).expect("test product fits");
        let separate_rhs: Vec<Limb> = operand.clone();
        let mut expected = vec![dirty_seed; result_len];
        Schoolbook::mul(&mut expected, &operand, &separate_rhs);

        let dirty_limb = dirty_seed | 1;
        let mut direct_square = vec![dirty_limb; result_len];
        Schoolbook::sqr(&mut direct_square, &operand);
        prop_assert_eq!(&direct_square, &expected);

        let mut aliased_multiply = vec![dirty_limb; result_len];
        Schoolbook::mul(&mut aliased_multiply, &operand, &operand);
        prop_assert_eq!(&aliased_multiply, &expected);
    }
}

#[test]
fn raw_low_product_initializes_spare_capacity() {
    for len in [0_usize, 1, 2, 4, 5, 17, 18, 19] {
        let left = vec![Limb::MAX; len];
        let right = vec![Limb::MAX; len];
        let mut full = vec![0; len * 2];
        Schoolbook::mul(&mut full, &left, &right);
        let mut actual = Vec::<Limb>::with_capacity(len);
        // SAFETY: the allocation reserves len writable limbs, disjoint from
        // the initialized len-limb inputs. Initialization mode writes every
        // output limb before set_len exposes it; zero length accesses neither pointer.
        unsafe {
            Schoolbook::mullo_basecase_unchecked(
                actual.as_mut_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                len,
            );
            actual.set_len(len);
        }
        assert_eq!(
            actual.as_slice(),
            full.get(..len).expect("low prefix exists")
        );
    }
}

#[test]
fn raw_scalar_initialization_and_exact_aliasing() {
    for len in [0_usize, 1, 2, 4, 5, 17] {
        for scalar in [0, 1, 2, Limb::MAX] {
            let input = vec![Limb::MAX; len];
            let mut expected = vec![0; len + 1];
            Schoolbook::mul(&mut expected, &input, &[scalar]);
            let mut actual = Vec::<Limb>::with_capacity(len + 1);
            // SAFETY: the fresh allocation reserves len+1 writable limbs and
            // input contains len initialized limbs. The scalar kernel writes
            // the complete prefix and final carry before set_len exposes them.
            unsafe {
                Schoolbook::mul_limb_unchecked(actual.as_mut_ptr(), input.as_ptr(), len, scalar);
                actual.set_len(len + 1);
            }
            assert_eq!(actual, expected);
            let mut in_place = Vec::with_capacity(len + 1);
            in_place.extend_from_slice(&input);
            // SAFETY: the old len-limb prefix is initialized, capacity covers
            // len+1 limbs, and exact aliasing reads each source before overwrite.
            // The final carry initializes the sole spare limb before set_len.
            unsafe {
                let pointer = in_place.as_mut_ptr();
                Schoolbook::mul_limb_unchecked(pointer, pointer, len, scalar);
                in_place.set_len(len + 1);
            }
            assert_eq!(in_place, expected);
        }
    }
}
