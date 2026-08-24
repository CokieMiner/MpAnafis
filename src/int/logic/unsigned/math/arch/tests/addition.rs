//! Properties for addition, paired addition/subtraction, and dual-add kernels.

#![allow(
    unsafe_code,
    reason = "The properties call unsafe kernels only with owned, length-proven buffers"
)]

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::super::ArchKernels;
use crate::int::types::Limb;

fn reference_add_assign(dst: &mut [Limb], src: &[Limb]) -> Limb {
    let mut carry = false;
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        let (sum, next_carry) = dst_limb.carrying_add(*src_limb, carry);
        *dst_limb = sum;
        carry = next_carry;
    }
    Limb::from(carry)
}

fn reference_sub_assign(dst: &mut [Limb], src: &[Limb]) -> Limb {
    let mut borrow = false;
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        let (difference, next_borrow) = dst_limb.borrowing_sub(*src_limb, borrow);
        *dst_limb = difference;
        borrow = next_borrow;
    }
    Limb::from(borrow)
}

fn reference_add_three(src_a: &[Limb], src_b: &[Limb]) -> (Vec<Limb>, Limb) {
    let mut dst = Vec::with_capacity(src_a.len());
    let mut carry = false;
    for (src_a_limb, src_b_limb) in src_a.iter().zip(src_b) {
        let (sum, next_carry) = src_a_limb.carrying_add(*src_b_limb, carry);
        dst.push(sum);
        carry = next_carry;
    }
    (dst, Limb::from(carry))
}

fn reference_add_sub(sum: &mut [Limb], difference: &mut [Limb]) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    for (sum_limb, difference_limb) in sum.iter_mut().zip(difference) {
        let original_sum = *sum_limb;
        let original_difference = *difference_limb;
        let (partial_sum, overflow_a) = original_sum.overflowing_add(original_difference);
        let (final_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        let (partial_difference, underflow_a) = original_sum.overflowing_sub(original_difference);
        let (final_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        *sum_limb = final_sum;
        *difference_limb = final_difference;
        carry = overflow_a | overflow_b;
        borrow = underflow_a | underflow_b;
    }
    (Limb::from(carry), Limb::from(borrow))
}

fn reference_add_reverse_sub(sum: &mut [Limb], difference: &mut [Limb]) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    for (sum_limb, difference_limb) in sum.iter_mut().zip(difference) {
        let original_sum = *sum_limb;
        let original_difference = *difference_limb;
        let (partial_sum, overflow_a) = original_sum.overflowing_add(original_difference);
        let (final_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        let (partial_difference, underflow_a) = original_difference.overflowing_sub(original_sum);
        let (final_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        *sum_limb = final_sum;
        *difference_limb = final_difference;
        carry = overflow_a | overflow_b;
        borrow = underflow_a | underflow_b;
    }
    (Limb::from(carry), Limb::from(borrow))
}

fn reference_add_two(
    dst_a: &mut [Limb],
    src_a: &[Limb],
    dst_b: &mut [Limb],
    src_b: &[Limb],
) -> (Limb, Limb) {
    let carry_a = reference_add_assign(dst_a, src_a);
    let carry_b = reference_add_assign(dst_b, src_b);
    (carry_a, carry_b)
}

/// Full carry propagation across every block boundary, at every length.
///
/// `add_limbs_unchecked` dispatches into unrolled blocks of differing widths,
/// and a carry has to survive each boundary between them. Random limbs do not
/// test that: a limb carries out only when it is `MAX`, so a ripple crossing a
/// boundary has probability near `2^-64` per limb and the property tests above
/// have effectively never produced one.
///
/// These operands force it. `MAX + 1` ripples the whole width from limb zero,
/// and `MAX + MAX` carries out of every limb, so any block boundary that drops
/// or duplicates CF is wrong here at the exact length where the boundary falls.
/// The range covers every combination of block selectors the routine has.
#[test]
fn add_limbs_propagates_carry_across_every_block_boundary() {
    for len in 0..=80_usize {
        for (label, src) in [
            ("MAX + 1", {
                let mut src = vec![Limb::MIN; len];
                if let Some(first) = src.first_mut() {
                    *first = 1;
                }
                src
            }),
            ("MAX + MAX", vec![Limb::MAX; len]),
        ] {
            let mut actual = vec![Limb::MAX; len];
            let mut expected = actual.clone();
            let expected_carry = reference_add_assign(&mut expected, &src);

            // SAFETY: both vectors contain exactly `len` initialized limbs.
            let actual_carry =
                unsafe { ArchKernels::add_limbs_unchecked(actual.as_mut_ptr(), src.as_ptr(), len) };

            assert_eq!(actual, expected, "{label} limbs differ at len {len}");
            assert_eq!(
                actual_carry, expected_carry,
                "{label} carry-out differs at len {len}"
            );
        }
    }
}

/// The borrow counterpart of
/// [`add_limbs_propagates_carry_across_every_block_boundary`].
///
/// `0 - 1` borrows out of every limb, so the chain runs the full width and any
/// block boundary that drops or duplicates CF is wrong at the exact length
/// where that boundary falls. Random operands borrow across a boundary about
/// as rarely as they carry across one.
#[test]
fn sub_limbs_propagates_borrow_across_every_block_boundary() {
    for len in 0..=80_usize {
        for (label, src) in [
            ("0 - 1", {
                let mut src = vec![Limb::MIN; len];
                if let Some(first) = src.first_mut() {
                    *first = 1;
                }
                src
            }),
            ("0 - MAX", vec![Limb::MAX; len]),
        ] {
            let mut actual = vec![Limb::MIN; len];
            let mut expected = actual.clone();
            let expected_borrow = reference_sub_assign(&mut expected, &src);

            // SAFETY: both vectors contain exactly `len` initialized limbs.
            let actual_borrow =
                unsafe { ArchKernels::sub_limbs_unchecked(actual.as_mut_ptr(), src.as_ptr(), len) };

            assert_eq!(actual, expected, "{label} limbs differ at len {len}");
            assert_eq!(
                actual_borrow, expected_borrow,
                "{label} borrow-out differs at len {len}"
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_add_limbs_matches_reference(
        words in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..=64),
    ) {
        let mut actual_dst: Vec<Limb> = words.iter().map(|&(dst, _)| dst).collect();
        let src: Vec<Limb> = words.iter().map(|&(_, src)| src).collect();
        let mut expected_dst = actual_dst.clone();
        let expected_carry = reference_add_assign(&mut expected_dst, &src);

        // SAFETY: both vectors contain exactly `words.len()` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::add_limbs_unchecked(
                actual_dst.as_mut_ptr(),
                src.as_ptr(),
                words.len(),
            )
        };

        prop_assert_eq!(actual_dst, expected_dst);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_add_limbs_three_matches_reference(
        words in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..=64),
    ) {
        let src_a: Vec<Limb> = words.iter().map(|&(src_a, _)| src_a).collect();
        let src_b: Vec<Limb> = words.iter().map(|&(_, src_b)| src_b).collect();
        let (expected_dst, expected_carry) = reference_add_three(&src_a, &src_b);
        let mut actual_dst = vec![0; words.len()];

        // SAFETY: all three vectors contain exactly `words.len()` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                actual_dst.as_mut_ptr(),
                src_a.as_ptr(),
                src_b.as_ptr(),
                words.len(),
            )
        };

        prop_assert_eq!(actual_dst, expected_dst);
        prop_assert_eq!(actual_carry, expected_carry);
    }

    #[test]
    fn prop_sub_limbs_matches_reference(
        words in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..=64),
    ) {
        let mut actual_dst: Vec<Limb> = words.iter().map(|&(dst, _)| dst).collect();
        let src: Vec<Limb> = words.iter().map(|&(_, src)| src).collect();
        let mut expected_dst = actual_dst.clone();
        let expected_borrow = reference_sub_assign(&mut expected_dst, &src);

        // SAFETY: both vectors contain exactly `words.len()` initialized limbs.
        let actual_borrow = unsafe {
            ArchKernels::sub_limbs_unchecked(
                actual_dst.as_mut_ptr(),
                src.as_ptr(),
                words.len(),
            )
        };

        prop_assert_eq!(actual_dst, expected_dst);
        prop_assert_eq!(actual_borrow, expected_borrow);
    }

    #[test]
    fn prop_sub_limbs_three_matches_reference(
        words in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..=64),
    ) {
        let src_a: Vec<Limb> = words.iter().map(|&(src_a, _)| src_a).collect();
        let src_b: Vec<Limb> = words.iter().map(|&(_, src_b)| src_b).collect();
        let mut expected_dst = src_a.clone();
        let expected_borrow = reference_sub_assign(&mut expected_dst, &src_b);
        let mut actual_dst = vec![0; words.len()];

        // SAFETY: all three vectors contain exactly `words.len()` initialized limbs.
        let actual_borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                actual_dst.as_mut_ptr(),
                src_a.as_ptr(),
                src_b.as_ptr(),
                words.len(),
            )
        };

        prop_assert_eq!(actual_dst, expected_dst);
        prop_assert_eq!(actual_borrow, expected_borrow);
    }

    #[test]
    fn prop_addition_carry_ripples_through_every_limb(len in 1_usize..=64) {
        let mut add_assign_dst = vec![Limb::MAX; len];
        let src_a = vec![Limb::MAX; len];
        let mut one = vec![0; len];
        if let Some(low_limb) = one.first_mut() {
            *low_limb = 1;
        }
        let mut three_dst = vec![0; len];

        // MAX at limb zero plus one produces carry_1. Every higher MAX plus
        // carry_i again produces zero and carry_(i+1), so induction proves all
        // `len` result limbs are zero and the final carry is one.
        // SAFETY: every vector contains exactly `len` initialized limbs.
        let assign_carry = unsafe {
            ArchKernels::add_limbs_unchecked(
                add_assign_dst.as_mut_ptr(),
                one.as_ptr(),
                len,
            )
        };
        // SAFETY: every vector contains exactly `len` initialized limbs.
        let three_carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                three_dst.as_mut_ptr(),
                src_a.as_ptr(),
                one.as_ptr(),
                len,
            )
        };

        prop_assert_eq!(add_assign_dst, vec![0; len]);
        prop_assert_eq!(assign_carry, 1);
        prop_assert_eq!(three_dst, vec![0; len]);
        prop_assert_eq!(three_carry, 1);
    }

    #[test]
    fn prop_add_sub_kernel_matches_reference(
        limbs in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..129),
    ) {
        let (mut expected_sum, mut expected_difference): (Vec<_>, Vec<_>) =
            limbs.iter().copied().unzip();
        let mut actual_sum = expected_sum.clone();
        let mut actual_difference = expected_difference.clone();
        let expected_carries = reference_add_sub(&mut expected_sum, &mut expected_difference);
        // SAFETY: the vectors are disjoint and each has exactly `limbs.len()` elements.
        let actual_carries = unsafe {
            ArchKernels::add_sub_limbs_unchecked(
                actual_sum.as_mut_ptr(),
                actual_difference.as_mut_ptr(),
                limbs.len(),
            )
        };
        prop_assert_eq!(actual_sum, expected_sum);
        prop_assert_eq!(actual_difference, expected_difference);
        prop_assert_eq!(actual_carries, expected_carries);
    }

    #[test]
    fn prop_add_sub_from_kernel_matches_reference_and_exact_alias(
        limbs in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..129),
    ) {
        let (mut expected_sum, mut expected_difference): (Vec<_>, Vec<_>) =
            limbs.iter().copied().unzip();
        let expected_carries = reference_add_sub(&mut expected_sum, &mut expected_difference);
        let (left, source): (Vec<_>, Vec<_>) = limbs.iter().copied().unzip();
        let kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();

        let mut disjoint_sum = left.clone();
        let mut disjoint_difference = vec![0; limbs.len()];
        // SAFETY: all spans contain limbs.len() elements and are pairwise
        // disjoint, as required by the selected architecture kernel.
        let disjoint_carries = unsafe {
            kernel(
                disjoint_sum.as_mut_ptr(),
                disjoint_difference.as_mut_ptr(),
                source.as_ptr(),
                limbs.len(),
            )
        };

        let mut aliased_sum = left;
        let mut aliased_difference_source = source;
        let aliased_source = aliased_difference_source.as_ptr();
        // SAFETY: the sum span is disjoint and the difference output exactly
        // aliases its source, which is the kernel's explicitly supported alias.
        let aliased_carries = unsafe {
            kernel(
                aliased_sum.as_mut_ptr(),
                aliased_difference_source.as_mut_ptr(),
                aliased_source,
                limbs.len(),
            )
        };

        prop_assert_eq!(disjoint_sum.as_slice(), expected_sum.as_slice());
        prop_assert_eq!(disjoint_difference.as_slice(), expected_difference.as_slice());
        prop_assert_eq!(disjoint_carries, expected_carries);
        prop_assert_eq!(aliased_sum.as_slice(), expected_sum.as_slice());
        prop_assert_eq!(aliased_difference_source.as_slice(), expected_difference.as_slice());
        prop_assert_eq!(aliased_carries, expected_carries);
    }

    #[test]
    fn prop_add_reverse_sub_kernel_matches_reference(
        limbs in proptest::collection::vec((any::<Limb>(), any::<Limb>()), 0..129),
    ) {
        let (mut expected_sum, mut expected_difference): (Vec<_>, Vec<_>) =
            limbs.iter().copied().unzip();
        let mut actual_sum = expected_sum.clone();
        let mut actual_difference = expected_difference.clone();
        let expected_carries =
            reference_add_reverse_sub(&mut expected_sum, &mut expected_difference);
        // SAFETY: the vectors are disjoint and each has exactly `limbs.len()` elements.
        let actual_carries = unsafe {
            ArchKernels::add_reverse_sub_limbs_unchecked(
                actual_sum.as_mut_ptr(),
                actual_difference.as_mut_ptr(),
                limbs.len(),
            )
        };
        prop_assert_eq!(actual_sum, expected_sum);
        prop_assert_eq!(actual_difference, expected_difference);
        prop_assert_eq!(actual_carries, expected_carries);
    }

    #[test]
    fn prop_add_two_kernel_matches_reference(
        limbs in proptest::collection::vec(
            (any::<Limb>(), any::<Limb>(), any::<Limb>(), any::<Limb>()),
            0..129,
        ),
    ) {
        let mut expected_dst_a: Vec<_> = limbs.iter().map(|values| values.0).collect();
        let src_a: Vec<_> = limbs.iter().map(|values| values.1).collect();
        let mut expected_dst_b: Vec<_> = limbs.iter().map(|values| values.2).collect();
        let src_b: Vec<_> = limbs.iter().map(|values| values.3).collect();
        let mut actual_dst_a = expected_dst_a.clone();
        let mut actual_dst_b = expected_dst_b.clone();
        let expected_carries =
            reference_add_two(&mut expected_dst_a, &src_a, &mut expected_dst_b, &src_b);
        // SAFETY: all four vectors are disjoint and have exactly `limbs.len()` elements.
        let actual_carries = unsafe {
            ArchKernels::add_two_limbs_unchecked(
                actual_dst_a.as_mut_ptr(),
                src_a.as_ptr(),
                actual_dst_b.as_mut_ptr(),
                src_b.as_ptr(),
                limbs.len(),
            )
        };
        prop_assert_eq!(actual_dst_a, expected_dst_a);
        prop_assert_eq!(actual_dst_b, expected_dst_b);
        prop_assert_eq!(actual_carries, expected_carries);
    }
}

/// Exercise the AVX2 implementation directly; the normal property above
/// follows runtime selection and can therefore choose ADX on this host.
#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(miri)
))]
#[test]
fn avx2_add_sub_from_preserves_carries_and_exact_alias() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let patterns = [
        (Limb::MIN, Limb::MIN),
        (Limb::MAX, Limb::MIN),
        (Limb::MIN, Limb::MAX),
        (Limb::MAX, Limb::MAX),
        (1, Limb::MAX),
        (Limb::MAX, 1),
    ];
    let mut cases = Vec::new();
    for len in 0..=37_usize {
        for &(left_limb, right_limb) in &patterns {
            cases.push((vec![left_limb; len], vec![right_limb; len]));
        }
    }
    let mut random_state = 0x9e37_79b9_7f4a_7c15_usize;
    for len in [4_usize, 5, 7, 8, 9, 16] {
        for _ in 0..64 {
            let mut left = Vec::with_capacity(len);
            let mut source = Vec::with_capacity(len);
            for _ in 0..len {
                random_state ^= random_state.wrapping_shl(13);
                random_state ^= random_state.wrapping_shr(7);
                random_state ^= random_state.wrapping_shl(17);
                left.push(random_state);
                random_state ^= random_state.rotate_left(29);
                source.push(random_state);
            }
            cases.push((left, source));
        }
    }
    for (left, source) in cases {
        let mut expected_sum = left.clone();
        let mut expected_difference = source.clone();
        let expected_carries = reference_add_sub(&mut expected_sum, &mut expected_difference);

        let len = left.len();
        let mut actual_sum = left.clone();
        let mut actual_difference = vec![0; len];
        // SAFETY: every span has `len` initialized limbs and the
        // destination/source aliasing matches the backend contract.
        let actual_carries = unsafe {
            super::super::add_sub_from_limbs_unchecked::add_sub_from_limbs_unchecked_avx2_test(
                actual_sum.as_mut_ptr(),
                actual_difference.as_mut_ptr(),
                source.as_ptr(),
                len,
            )
        };
        assert_eq!(actual_sum, expected_sum);
        assert_eq!(actual_difference, expected_difference);
        assert_eq!(actual_carries, expected_carries);

        let mut aliased_sum = left;
        let mut aliased_difference = source;
        let aliased_source = aliased_difference.as_ptr();
        // SAFETY: exact `difference == source` aliasing is explicitly
        // supported, and the source is loaded before each store.
        let aliased_carries = unsafe {
            super::super::add_sub_from_limbs_unchecked::add_sub_from_limbs_unchecked_avx2_test(
                aliased_sum.as_mut_ptr(),
                aliased_difference.as_mut_ptr(),
                aliased_source,
                len,
            )
        };
        assert_eq!(aliased_sum, expected_sum);
        assert_eq!(aliased_difference, expected_difference);
        assert_eq!(aliased_carries, expected_carries);
    }
}
