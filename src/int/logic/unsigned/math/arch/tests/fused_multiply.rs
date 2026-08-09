//! Oracle and adversarial properties for fused add-multiply and subtract-multiply kernels.

#![allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "The kernel properties call unsafe backends, index length-proven vectors, and deliberately split DoubleLimb products"
)]

use alloc::{vec, vec::Vec};

use proptest::{prelude::*, sample::select};

use super::{
    equal_length_limb_vecs, limb_vec, reference_add_multiply_limb, reference_add_multiply_two,
    reference_multiply_two,
};
use crate::int::{
    logic::math::arch::ArchKernels,
    types::{DoubleLimb, Limb},
};

const ADVERSARIAL_LENS: [usize; 11] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16];

fn reference_subtract_multiply_limb(dst: &mut [Limb], src: &[Limb], scalar: Limb) -> (Limb, Limb) {
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        let product = (*src_limb as DoubleLimb).wrapping_mul(scalar as DoubleLimb);
        let low_product = product as Limb;
        let high_product = product.wrapping_shr(Limb::BITS) as Limb;

        let (low_with_carry, carry_from_product) = low_product.overflowing_add(carry);
        carry = high_product.wrapping_add(Limb::from(carry_from_product));

        let (partial_difference, borrow_from_product) = dst_limb.overflowing_sub(low_with_carry);
        let (difference, borrow_from_prior_limb) = partial_difference.overflowing_sub(borrow);
        *dst_limb = difference;
        borrow = Limb::from(borrow_from_product).wrapping_add(Limb::from(borrow_from_prior_limb));
    }
    (carry, borrow)
}

fn check_add_multiply_identity(
    dst_before: &[Limb],
    src: &[Limb],
    scalar: Limb,
    carry_out: Limb,
    dst_after: &[Limb],
) {
    if src.len() > 1 {
        return;
    }
    let value_before = if src.is_empty() {
        0
    } else {
        dst_before[0] as DoubleLimb
    };
    let source_value = if src.is_empty() {
        0
    } else {
        src[0] as DoubleLimb
    };
    let value_after = if src.is_empty() {
        0
    } else {
        dst_after[0] as DoubleLimb
    };
    let carry_value = carry_out as DoubleLimb;
    let reconstructed = value_after.wrapping_add(carry_value.wrapping_shl(Limb::BITS));
    let expected = value_before.wrapping_add(source_value.wrapping_mul(scalar as DoubleLimb));
    assert_eq!(
        reconstructed,
        expected,
        "addmul_1 algebraic identity failed: len={}",
        src.len()
    );
}

fn check_subtract_multiply_identity(
    dst_before: &[Limb],
    src: &[Limb],
    scalar: Limb,
    carry_out: Limb,
    borrow_out: Limb,
    dst_after: &[Limb],
) {
    if src.len() != 1 {
        return;
    }
    let value_before = dst_before[0] as DoubleLimb;
    let value_after = dst_after[0] as DoubleLimb;
    let product = (src[0] as DoubleLimb).wrapping_mul(scalar as DoubleLimb);
    // The oracle's high product carry and low-limb subtraction borrow are
    // independent. Their sum is the full borrow above the retained low limb.
    let full_borrow = (carry_out as DoubleLimb).wrapping_add(borrow_out as DoubleLimb);
    let reconstructed = value_after.wrapping_sub(full_borrow.wrapping_shl(Limb::BITS));
    let expected = value_before.wrapping_sub(product);
    assert_eq!(
        reconstructed, expected,
        "submul_1: {value_before:#x} - {:#x}*{scalar:#x}: value_after={value_after:#x} carry={carry_out:#x} borrow={borrow_out:#x}",
        src[0],
    );
}

fn adversarial_weight(len: usize) -> Vec<Vec<Limb>> {
    vec![
        vec![0; len],
        vec![Limb::MAX; len],
        (0..len)
            .map(|index| {
                if index.is_multiple_of(2) {
                    Limb::MAX
                } else {
                    0
                }
            })
            .collect(),
        (0..len)
            .map(|index| {
                if index.is_multiple_of(2) {
                    0
                } else {
                    Limb::MAX
                }
            })
            .collect(),
        vec![Limb::MAX.wrapping_sub(7); len],
    ]
}

fn adversarial_edge(len: usize) -> Vec<Vec<Limb>> {
    if len == 0 {
        return vec![];
    }
    let mut low_carry = vec![0; len];
    if let Some(low_limb) = low_carry.first_mut() {
        *low_limb = Limb::MAX;
    }
    let mut high_carry = vec![0; len];
    if let Some(high_limb) = high_carry.last_mut() {
        *high_limb = Limb::MAX;
    }
    vec![low_carry, high_carry]
}

fn all_patterns(len: usize) -> Vec<Vec<Limb>> {
    let mut patterns = adversarial_weight(len);
    patterns.extend(adversarial_edge(len));
    patterns
}

fn scalars() -> [Limb; 6] {
    [
        0,
        1,
        2,
        Limb::MAX.wrapping_div(2),
        Limb::MAX.wrapping_sub(1),
        Limb::MAX,
    ]
}

fn all_cases(len: usize) -> Vec<(Vec<Limb>, Vec<Limb>, Limb)> {
    let dst_patterns = all_patterns(len);
    let src_patterns = all_patterns(len);
    let mut cases = Vec::new();
    for dst in &dst_patterns {
        for src in &src_patterns {
            for scalar in scalars() {
                cases.push((dst.clone(), src.clone(), scalar));
            }
        }
    }
    cases
}

proptest! {
    #[test]
    fn prop_addmul_1_matches_oracle_for_adversarial_cases(
        len in select(&ADVERSARIAL_LENS),
    ) {
        for (dst_before, src, scalar) in all_cases(len) {
            let mut expected_dst = dst_before.clone();
            let mut actual_dst = dst_before.clone();
            let expected_carry =
                reference_add_multiply_limb(&mut expected_dst, &src, scalar);

            // SAFETY: both vectors contain exactly `len` initialized limbs.
            let actual_carry = unsafe {
                ArchKernels::add_mul_limbs_unchecked(
                    actual_dst.as_mut_ptr(),
                    src.as_ptr(),
                    len,
                    scalar,
                )
            };

            prop_assert_eq!(
                expected_carry,
                actual_carry,
                "addmul_1 carry: len={}, scalar={:#018x}",
                len,
                scalar,
            );
            prop_assert_eq!(&expected_dst, &actual_dst, "addmul_1 dst: len={}", len);
            check_add_multiply_identity(&dst_before, &src, scalar, actual_carry, &actual_dst);
        }
    }

    #[test]
    fn prop_addmul_1_matches_oracle_for_generated_cases(
        case in equal_length_limb_vecs(0..=24),
        scalar in any::<Limb>(),
    ) {
        let (src, dst_initial) = case;
        let len = src.len();
        let mut expected_dst = dst_initial.clone();
        let expected_carry = reference_add_multiply_limb(&mut expected_dst, &src, scalar);
        let mut actual_dst = dst_initial.clone();

        // SAFETY: both vectors contain exactly `len` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::add_mul_limbs_unchecked(
                actual_dst.as_mut_ptr(),
                src.as_ptr(),
                len,
                scalar,
            )
        };

        prop_assert_eq!(
            expected_carry,
            actual_carry,
            "addmul_1 carry mismatch at len={}",
            len,
        );
        prop_assert_eq!(
            &expected_dst,
            &actual_dst,
            "addmul_1 dst mismatch at len={}",
            len,
        );
        check_add_multiply_identity(&dst_initial, &src, scalar, actual_carry, &actual_dst);
    }

    #[test]
    fn prop_addmul_1_zero_len_returns_zero(scalar in any::<Limb>()) {
        // SAFETY: `len == 0` guarantees the kernel does not dereference either pointer.
        let carry = unsafe {
            ArchKernels::add_mul_limbs_unchecked(
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
                scalar,
            )
        };
        prop_assert_eq!(carry, 0);
    }

    #[test]
    fn prop_submul_1_matches_oracle_for_adversarial_cases(
        len in select(&ADVERSARIAL_LENS),
    ) {
        for (dst_before, src, scalar) in all_cases(len) {
            let mut expected_dst = dst_before.clone();
            let mut actual_dst = dst_before.clone();
            let expected_carries =
                reference_subtract_multiply_limb(&mut expected_dst, &src, scalar);

            // SAFETY: both vectors contain exactly `len` initialized limbs.
            let actual_carries = unsafe {
                ArchKernels::sub_mul_limbs_unchecked(
                    actual_dst.as_mut_ptr(),
                    src.as_ptr(),
                    len,
                    scalar,
                )
            };

            prop_assert_eq!(
                expected_carries,
                actual_carries,
                "submul_1: len={}, scalar={:#018x}",
                len,
                scalar,
            );
            prop_assert_eq!(&expected_dst, &actual_dst, "submul_1 dst: len={}", len);
            check_subtract_multiply_identity(
                &dst_before,
                &src,
                scalar,
                actual_carries.0,
                actual_carries.1,
                &actual_dst,
            );
        }
    }

    #[test]
    fn prop_submul_1_matches_oracle_for_generated_cases(
        case in equal_length_limb_vecs(0..=24),
        scalar in any::<Limb>(),
    ) {
        let (src, dst_initial) = case;
        let len = src.len();
        let mut expected_dst = dst_initial.clone();
        let expected_carries =
            reference_subtract_multiply_limb(&mut expected_dst, &src, scalar);
        let mut actual_dst = dst_initial.clone();

        // SAFETY: both vectors contain exactly `len` initialized limbs.
        let actual_carries = unsafe {
            ArchKernels::sub_mul_limbs_unchecked(
                actual_dst.as_mut_ptr(),
                src.as_ptr(),
                len,
                scalar,
            )
        };

        prop_assert_eq!(
            expected_carries,
            actual_carries,
            "submul_1 mismatch at len={}",
            len,
        );
        prop_assert_eq!(
            &expected_dst,
            &actual_dst,
            "submul_1 dst mismatch at len={}",
            len,
        );
        check_subtract_multiply_identity(
            &dst_initial,
            &src,
            scalar,
            actual_carries.0,
            actual_carries.1,
            &actual_dst,
        );
    }

    #[test]
    fn prop_submul_1_zero_len_returns_zero_pair(scalar in any::<Limb>()) {
        // SAFETY: `len == 0` guarantees the kernel does not dereference either pointer.
        let result = unsafe {
            ArchKernels::sub_mul_limbs_unchecked(
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
                scalar,
            )
        };
        prop_assert_eq!(result, (0, 0));
    }

    #[test]
    fn prop_addmul_2_matches_overlapping_row_oracle(
        case in (0_usize..=24).prop_flat_map(|len| {
            let dst_len = len.wrapping_add(1);
            (limb_vec(len..=len), limb_vec(dst_len..=dst_len))
        }),
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        let (src, dst_initial) = case;
        let (low_scalar, high_scalar) = scalars;
        let len = src.len();
        let mut expected_dst = dst_initial.clone();
        let expected_carries = reference_add_multiply_two(
            &mut expected_dst,
            &src,
            low_scalar,
            high_scalar,
        );
        let mut actual_dst = dst_initial;
        let kernel = ArchKernels::selected_add_mul_2_limbs_unchecked();

        // SAFETY: src has `len` limbs, dst has `len+1`, and their independent
        // vector allocations cannot overlap.
        let actual_carries = unsafe {
            kernel(
                actual_dst.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            )
        };

        prop_assert_eq!(actual_carries, expected_carries);
        prop_assert_eq!(&actual_dst, &expected_dst);
    }

    #[test]
    fn prop_addmul_2_zero_len_touches_no_memory(
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        let kernel = ArchKernels::selected_add_mul_2_limbs_unchecked();
        // SAFETY: the zero-length contract forbids either pointer from being
        // dereferenced and requires the two zero carries returned below.
        let carries = unsafe {
            kernel(
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
                scalars.0,
                scalars.1,
            )
        };
        prop_assert_eq!(carries, (0, 0));
    }

    #[test]
    fn prop_mul_2_matches_two_base_b_rows(
        src in limb_vec(1..=24),
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        let (low_scalar, high_scalar) = scalars;
        let expected = reference_multiply_two(&src, low_scalar, high_scalar);
        let dst_len = src.len().wrapping_add(2);
        let mut zeroed = vec![0; dst_len];
        let mut poisoned = vec![Limb::MAX; dst_len];

        // SAFETY: each destination has `src.len()+2` writable limbs, and neither
        // destination aliases the initialized `src` span.
        unsafe {
            ArchKernels::selected_mul_2_limbs_unchecked()(
                zeroed.as_mut_ptr(),
                src.as_ptr(),
                src.len(),
                low_scalar,
                high_scalar,
            );
        }
        // SAFETY: the same proven spans apply to the independently poisoned
        // destination. Different initial contents verify the write-only contract.
        unsafe {
            ArchKernels::selected_mul_2_limbs_unchecked()(
                poisoned.as_mut_ptr(),
                src.as_ptr(),
                src.len(),
                low_scalar,
                high_scalar,
            );
        }

        prop_assert_eq!(&zeroed, &expected);
        prop_assert_eq!(&poisoned, &expected);
    }

    #[test]
    fn prop_mul_2_zero_len_touches_no_memory(
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        // SAFETY: the kernel's zero-length contract returns before dereferencing
        // either pointer on every backend.
        unsafe {
            ArchKernels::selected_mul_2_limbs_unchecked()(
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
                scalars.0,
                scalars.1,
            );
        }
    }
}
