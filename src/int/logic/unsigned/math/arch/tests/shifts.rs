//! Reference properties for selected in-place limb-shift kernels.

#![allow(
    unsafe_code,
    reason = "The properties call raw-pointer kernels with vector lengths passed unchanged"
)]

use alloc::vec;

use proptest::prelude::*;

use super::super::ArchKernels;
use crate::int::types::Limb;

fn reference_left_shift(limbs: &mut [Limb], shift: u32) -> Limb {
    let complementary_shift = Limb::BITS.wrapping_sub(shift);
    let mut carry = 0;
    for limb in limbs {
        let source = *limb;
        *limb = source.wrapping_shl(shift) | carry;
        carry = source.wrapping_shr(complementary_shift);
    }
    carry
}

fn reference_right_shift(limbs: &mut [Limb], shift: u32) -> Limb {
    let complementary_shift = Limb::BITS.wrapping_sub(shift);
    let mut carry = 0;
    for limb in limbs.iter_mut().rev() {
        let source = *limb;
        *limb = source.wrapping_shr(shift) | carry;
        carry = source.wrapping_shl(complementary_shift);
    }
    carry
}

#[allow(
    clippy::indexing_slicing,
    reason = "the test constructs equal-length spans and the loop is bounded by dst.len()"
)]
fn reference_sub_shifted_high(
    dst: &mut [Limb],
    src: &[Limb],
    shift: u32,
    mut borrow: bool,
) -> Limb {
    let right_shift = Limb::BITS.wrapping_sub(shift);
    for index in 0..dst.len() {
        let low = src[index].wrapping_shr(right_shift);
        let high = src
            .get(index.wrapping_add(1))
            .map_or(0, |next| next.wrapping_shl(shift));
        let (partial, underflow_a) = dst[index].overflowing_sub(low | high);
        let (result, underflow_b) = partial.overflowing_sub(Limb::from(borrow));
        dst[index] = result;
        borrow = underflow_a | underflow_b;
    }
    Limb::from(borrow)
}

#[allow(
    clippy::indexing_slicing,
    reason = "the test constructs equal-length spans and the loop is bounded by src.len()"
)]
fn reference_left_shift_into(dst: &mut [Limb], src: &[Limb], shift: u32) -> Limb {
    let complementary_shift = Limb::BITS.wrapping_sub(shift);
    let mut carry = 0;
    for index in 0..src.len() {
        let source = src[index];
        dst[index] = source.wrapping_shl(shift) | carry;
        carry = source.wrapping_shr(complementary_shift);
    }
    carry
}

#[allow(
    clippy::indexing_slicing,
    reason = "the test constructs equal-length spans and the loop is bounded by src.len()"
)]
fn reference_right_shift_into(dst: &mut [Limb], src: &[Limb], shift: u32) -> Limb {
    let complementary_shift = Limb::BITS.wrapping_sub(shift);
    let mut carry = 0;
    for index in (0..src.len()).rev() {
        let source = src[index];
        dst[index] = source.wrapping_shr(shift) | carry;
        carry = source.wrapping_shl(complementary_shift);
    }
    carry
}

proptest! {
    #[test]
    fn prop_left_shift_matches_reference(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        let mut expected = initial.clone();
        let expected_carry = reference_left_shift(&mut expected, shift);
        let mut actual = initial;

        // SAFETY: actual contains exactly actual.len() initialized writable limbs,
        // and the strategy proves 0 < shift < Limb::BITS.
        let actual_carry = unsafe {
            ArchKernels::lshift_unchecked(actual.as_mut_ptr(), actual.len(), shift)
        };

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_right_shift_matches_reference(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        let mut expected = initial.clone();
        let expected_carry = reference_right_shift(&mut expected, shift);
        let mut actual = initial;

        // SAFETY: actual contains exactly actual.len() initialized writable limbs,
        // and the strategy proves 0 < shift < Limb::BITS.
        let actual_carry = unsafe {
            ArchKernels::rshift_unchecked(actual.as_mut_ptr(), actual.len(), shift)
        };

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_left_shift_into_matches_reference(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        let mut expected = vec![0; initial.len()];
        let expected_carry = reference_left_shift_into(&mut expected, &initial, shift);
        let mut actual = vec![0; initial.len()];

        // SAFETY: actual contains exactly initial.len() initialized writable
        // limbs, initial holds that many readable limbs, the spans are
        // disjoint, and the strategy proves 0 < shift < Limb::BITS.
        let actual_carry = unsafe {
            ArchKernels::lshift_into_unchecked(
                actual.as_mut_ptr(),
                initial.as_ptr(),
                initial.len(),
                shift,
            )
        };

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_right_shift_into_matches_reference(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        let mut expected = vec![0; initial.len()];
        let expected_carry = reference_right_shift_into(&mut expected, &initial, shift);
        let mut actual = vec![0; initial.len()];

        // SAFETY: actual contains exactly initial.len() initialized writable
        // limbs, initial holds that many readable limbs, the spans are
        // disjoint, and the strategy proves 0 < shift < Limb::BITS.
        let actual_carry = unsafe {
            ArchKernels::rshift_into_unchecked(
                actual.as_mut_ptr(),
                initial.as_ptr(),
                initial.len(),
                shift,
            )
        };

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_sub_shifted_high_matches_reference(
        (initial, source) in super::equal_length_limb_vecs(0..=129),
        shift in 1_u32..Limb::BITS,
        borrow in any::<bool>(),
    ) {
        let mut expected = initial.clone();
        let expected_borrow = reference_sub_shifted_high(&mut expected, &source, shift, borrow);
        let mut actual = initial;
        let kernel = ArchKernels::selected_sub_shifted_high_limbs_unchecked();

        // SAFETY: both vectors contain exactly source.len() limbs, are disjoint,
        // the strategy proves the shift range, and bool conversion is 0 or 1.
        let actual_borrow = unsafe {
            kernel(
                actual.as_mut_ptr(),
                source.as_ptr(),
                source.len(),
                shift,
                Limb::from(borrow),
            )
        };

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(actual_borrow, expected_borrow);
    }

    #[test]
    fn prop_zero_length_shifts_touch_no_memory(shift in 1_u32..Limb::BITS) {
        // SAFETY: len == 0 requires both kernels to return before dereferencing
        // the pointer; the strategy establishes the shift-count precondition.
        let left = unsafe { ArchKernels::lshift_unchecked(core::ptr::null_mut(), 0, shift) };
        // SAFETY: the identical zero-length and shift proof applies here.
        let right = unsafe { ArchKernels::rshift_unchecked(core::ptr::null_mut(), 0, shift) };
        prop_assert_eq!(left, 0);
        prop_assert_eq!(right, 0);
    }
}
