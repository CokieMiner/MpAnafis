//! Properties for carry and borrow propagation kernels.

#![allow(
    unsafe_code,
    reason = "The properties call unsafe kernels with vector lengths passed unchanged"
)]

use alloc::vec;

use proptest::prelude::*;

use super::super::ArchKernels;
use crate::int::types::Limb;

fn reference_propagate_carry(dst: &mut [Limb], mut carry: Limb) -> Limb {
    for limb in dst {
        let (value, overflow) = limb.overflowing_add(carry);
        *limb = value;
        carry = Limb::from(overflow);
        if carry == 0 {
            break;
        }
    }
    carry
}

fn reference_propagate_borrow(dst: &mut [Limb], mut borrow: Limb) -> Limb {
    for limb in dst {
        let (value, underflow) = limb.overflowing_sub(borrow);
        *limb = value;
        borrow = Limb::from(underflow);
        if borrow == 0 {
            break;
        }
    }
    borrow
}

proptest! {
    #[test]
    fn prop_propagate_carry_all_max_returns_carry_out(
        len in 0_usize..=20,
        carry_in in 0_u8..=1,
    ) {
        let mut actual_dst = vec![Limb::MAX; len];
        let carry_limb = Limb::from(carry_in);
        let mut expected_dst = actual_dst.clone();

        let expected_carry = reference_propagate_carry(&mut expected_dst, carry_limb);
        // SAFETY: `actual_dst` contains exactly `len` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::propagate_carry_unchecked(actual_dst.as_mut_ptr(), len, carry_limb)
        };

        prop_assert_eq!(
            expected_carry,
            actual_carry,
            "propagate_carry: all-{:#x}, len={}, carry_in={}",
            Limb::MAX,
            len,
            carry_in,
        );
        prop_assert_eq!(
            expected_dst,
            actual_dst,
            "propagate_carry dst mismatch: len={}",
            len,
        );
    }

    #[test]
    fn prop_propagate_carry_zeros_no_effect(
        len in 0_usize..=20,
        carry_in in 0_u8..=1,
    ) {
        let mut actual_dst = vec![0; len];
        let carry_limb = Limb::from(carry_in);
        let mut expected_dst = actual_dst.clone();

        let expected_carry = reference_propagate_carry(&mut expected_dst, carry_limb);
        // SAFETY: `actual_dst` contains exactly `len` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::propagate_carry_unchecked(actual_dst.as_mut_ptr(), len, carry_limb)
        };

        prop_assert_eq!(
            expected_carry,
            actual_carry,
            "propagate_carry: all-0, len={}, carry_in={}",
            len,
            carry_in,
        );
        prop_assert_eq!(
            expected_dst,
            actual_dst,
            "propagate_carry dst mismatch: len={}",
            len,
        );
    }

    #[test]
    fn prop_propagate_borrow_all_zero_returns_borrow_out(
        len in 0_usize..=20,
        borrow_in in 0_u8..=1,
    ) {
        let mut actual_dst = vec![0; len];
        let borrow_limb = Limb::from(borrow_in);
        let mut expected_dst = actual_dst.clone();

        let expected_borrow = reference_propagate_borrow(&mut expected_dst, borrow_limb);
        // SAFETY: `actual_dst` contains exactly `len` initialized limbs.
        let actual_borrow = unsafe {
            ArchKernels::propagate_borrow_unchecked(actual_dst.as_mut_ptr(), len, borrow_limb)
        };

        prop_assert_eq!(
            expected_borrow,
            actual_borrow,
            "propagate_borrow: all-0, len={}, borrow_in={}",
            len,
            borrow_in,
        );
        prop_assert_eq!(
            expected_dst,
            actual_dst,
            "propagate_borrow dst mismatch: len={}",
            len,
        );
    }

    #[test]
    fn prop_propagate_borrow_all_max_no_effect(
        len in 0_usize..=20,
        borrow_in in 0_u8..=1,
    ) {
        let mut actual_dst = vec![Limb::MAX; len];
        let borrow_limb = Limb::from(borrow_in);
        let mut expected_dst = actual_dst.clone();

        let expected_borrow = reference_propagate_borrow(&mut expected_dst, borrow_limb);
        // SAFETY: `actual_dst` contains exactly `len` initialized limbs.
        let actual_borrow = unsafe {
            ArchKernels::propagate_borrow_unchecked(actual_dst.as_mut_ptr(), len, borrow_limb)
        };

        prop_assert_eq!(
            expected_borrow,
            actual_borrow,
            "propagate_borrow: all-{:#x}, len={}, borrow_in={}",
            Limb::MAX,
            len,
            borrow_in,
        );
        prop_assert_eq!(
            expected_dst,
            actual_dst,
            "propagate_borrow dst mismatch: len={}",
            len,
        );
    }
}
