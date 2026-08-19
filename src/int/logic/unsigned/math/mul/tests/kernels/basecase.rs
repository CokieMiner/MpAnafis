//! Property tests for write-complete basecase multiplication and squaring.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::{ArchKernels, DoubleLimb, Limb, Schoolbook};

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "The test reference uses exact widened schoolbook columns over generated owned vectors"
)]
fn reference_schoolbook(left: &[Limb], right: &[Limb]) -> Vec<Limb> {
    let mut result = vec![Limb::MIN; left.len().wrapping_add(right.len())];
    for (row, scalar) in left.iter().copied().enumerate() {
        let mut carry = DoubleLimb::MIN;
        for (column, source) in right.iter().copied().enumerate() {
            let output_index = row.wrapping_add(column);
            let product = (scalar as DoubleLimb)
                .wrapping_mul(source as DoubleLimb)
                .wrapping_add(result[output_index] as DoubleLimb)
                .wrapping_add(carry);
            result[output_index] = product as Limb;
            carry = product.wrapping_shr(Limb::BITS);
        }
        result[row.wrapping_add(right.len())] = carry as Limb;
    }
    result
}

proptest! {
    #[test]
    fn prop_portable_2x2_overwrites_and_matches_schoolbook(
        left in proptest::collection::vec(any::<Limb>(), 2..=2),
        right in proptest::collection::vec(any::<Limb>(), 2..=2),
        dirty_seed in any::<Limb>(),
    ) {
        let expected = reference_schoolbook(&left, &right);
        let mut actual = vec![dirty_seed | 1; 4];
        // SAFETY: both inputs contain exactly two limbs, the destination
        // contains four writable limbs, and all three vectors are disjoint.
        unsafe {
            ArchKernels::mul_2x2_portable_unchecked(
                actual.as_mut_ptr(),
                left.as_ptr(),
                right.as_ptr(),
            );
        }
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_portable_3x3_overwrites_and_matches_schoolbook(
        left in proptest::collection::vec(any::<Limb>(), 3..=3),
        right in proptest::collection::vec(any::<Limb>(), 3..=3),
        dirty_seed in any::<Limb>(),
    ) {
        let expected = reference_schoolbook(&left, &right);
        let mut actual = vec![dirty_seed | 1; 6];
        // SAFETY: both inputs contain exactly three limbs, the destination
        // contains six writable limbs, and all three vectors are disjoint.
        unsafe {
            ArchKernels::mul_3x3_portable_unchecked(
                actual.as_mut_ptr(),
                left.as_ptr(),
                right.as_ptr(),
            );
        }
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_schoolbook_square_overwrites_dirty_destination(
        operand in proptest::collection::vec(any::<Limb>(), 1..65),
        dirty_seed in any::<Limb>(),
    ) {
        let result_len = operand.len().wrapping_mul(2);
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
