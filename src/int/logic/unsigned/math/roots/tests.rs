//! Properties and regressions for the root family.

use proptest::prelude::*;

use super::*;

/// Regression: `sqrt_rem_recursive` used to split on `ceil(len / 4)`, which
/// left the recursive half short for any length that is not a multiple of
/// four. A nine-limb operand reaches a six-limb level where the seed was too
/// coarse for the single correction step, and `isqrt` returned the wrong
/// root. Sparse operands hit it first because their low limbs
/// are zero, but the split is wrong regardless of the limb contents.
#[test]
fn isqrt_resolves_every_width_for_sparse_and_dense_operands() {
    for len in 1_usize..=24 {
        let mut sparse: alloc::vec::Vec<Limb> = alloc::vec![0; len];
        if let Some(top) = sparse.last_mut() {
            *top = Limb::MAX ^ 0x5555;
        }
        let sparse_value = InternalArbiUint::from_limbs(sparse);
        let sparse_root = sparse_value.isqrt();
        assert!(
            sparse_root.mul(&sparse_root) <= sparse_value,
            "sparse root too large at {len} limbs"
        );

        let dense: alloc::vec::Vec<Limb> = alloc::vec![Limb::MAX.wrapping_div(3); len];
        let dense_value = InternalArbiUint::from_limbs(dense);
        let dense_root = dense_value.isqrt();
        assert!(
            dense_root.mul(&dense_root) <= dense_value,
            "dense root too large at {len} limbs"
        );
    }
}

proptest! {
    /// The residue screen is one sided: it may pass a non-square, but it must
    /// never reject a square.
    #[test]
    fn prop_screen_never_rejects_a_square(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=3),
    ) {
        let root = InternalArbiUint::from_limbs(limbs);
        let square = root.mul(&root);
        prop_assert!(may_be_square(&square));
    }

    /// Survivors of the screen still have to agree with the square root, so the
    /// screen must not change what `is_perfect_square` answers.
    #[test]
    fn prop_screen_agrees_with_is_perfect_square(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=3),
    ) {
        let value = InternalArbiUint::from_limbs(limbs);
        if !may_be_square(&value) {
            prop_assert!(!value.is_perfect_square());
        }
    }

    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest for root operations"
    )]
    #[test]
    fn roots_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=3),
        n in 2_u32..=5,
    ) {
        let val = InternalArbiUint::from_limbs(limbs);

        // isqrt property: r^2 <= x < (r+1)^2
        let sqrt_r = val.isqrt();
        let r_sq = sqrt_r.mul(&sqrt_r);
        prop_assert!(r_sq <= val);

        let mut r_plus_1 = sqrt_r;
        r_plus_1.add_assign(&InternalArbiUint::one());
        let r_plus_1_sq = r_plus_1.mul(&r_plus_1);
        if r_plus_1_sq.significant_bits() >= r_sq.significant_bits() {
            prop_assert!(val < r_plus_1_sq);
        }

        let direct_root = val.isqrt();
        let paired_root = val.sqrt_rem().0;
        prop_assert_eq!(direct_root, paired_root);

        // is_perfect_square matches isqrt roundtrip
        let square_root = val.isqrt();
        let is_square = square_root.mul(&square_root) == val;
        prop_assert_eq!(val.is_perfect_square(), is_square);

        // sqrt_rem returns (root, remainder) where root^2 + rem == x
        let (root, rem) = val.sqrt_rem();
        let root_sq = root.mul(&root);
        let mut sum = root_sq;
        sum.add_assign(&rem);
        prop_assert_eq!(&sum, &val);
        prop_assert!(rem < InternalArbiUint::from_u64(2).shl(root.significant_bits() + 1));

        // nth_root property: r^n <= x < (r+1)^n
        let r_nth = val.nth_root(n);
        let r_nth_pow = r_nth.pow(n);
        prop_assert!(r_nth_pow <= val);

        let mut r_nth_plus_1 = r_nth;
        r_nth_plus_1.add_assign(&InternalArbiUint::one());
        let r_nth_plus_1_pow = r_nth_plus_1.pow(n);
        if r_nth_plus_1_pow.significant_bits() >= r_nth_pow.significant_bits() {
            prop_assert!(val < r_nth_plus_1_pow);
        }
    }

    /// The recursive seed only engages above `HALVING_FLOOR_BITS`, which the
    /// three-limb operands above never reach for `isqrt`. These widths force
    /// it, and both roots must still bracket the true value exactly.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest for root operations"
    )]
    #[test]
    fn prop_wide_roots_bracket_exactly(
        limbs in proptest::collection::vec(any::<Limb>(), 8..=24),
        n in 3_u32..=7,
    ) {
        let val = InternalArbiUint::from_limbs(limbs);

        let root = val.isqrt();
        prop_assert!(root.mul(&root) <= val);
        let mut above = root;
        above.add_assign(&InternalArbiUint::one());
        prop_assert!(val < above.mul(&above));

        let nth = val.nth_root(n);
        prop_assert!(nth.pow(n) <= val);
        let mut nth_above = nth;
        nth_above.add_assign(&InternalArbiUint::one());
        prop_assert!(val < nth_above.pow(n));
    }
}
