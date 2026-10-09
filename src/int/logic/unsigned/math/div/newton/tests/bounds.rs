//! Lower reciprocal estimates and exact integer error bounds.

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, LIMB_BITS, Limb};

proptest! {
    #[test]
    fn block_reciprocal_is_a_lower_bound_with_less_than_two_units_error(
        mut limbs in proptest::collection::vec(prop_oneof![Just(Limb::MAX), any::<Limb>()], 1..=160),
        block_seed in any::<usize>(),
        guard_full in any::<bool>(),
    ) {
        *limbs.last_mut().expect("nonempty divisor") |= 1 << Limb::BITS.wrapping_sub(1);
        let n = limbs.len();
        let block = block_seed.rem_euclid(n).checked_add(1).expect("bounded block");
        let denominator = InternalMpUint::from_limbs(limbs);
        let mut scratch = DivScratch::default();
        let (storage, skip) = if guard_full {
            Division::newton_block_reciprocal::<true>(denominator.limbs(), block, &mut scratch)
        } else {
            Division::newton_block_reciprocal::<false>(denominator.limbs(), block, &mut scratch)
        };
        let width = block.checked_add(skip).and_then(|value| value.checked_add(1)).expect("bounded reciprocal width");
        prop_assert_eq!(storage.limbs().len(), width);
        prop_assert_eq!(storage.limbs().last(), Some(&1));
        let reciprocal = InternalMpUint::from_limbs_slice(storage.limbs().get(skip..).expect("reciprocal guard"));
        let power = power_of_base(n.checked_add(block).expect("bounded power width"));
        let product = reciprocal.mul(&denominator);
        prop_assert!(product <= power);
        prop_assert!(power.sub(&product) < denominator.add(&denominator));
        if skip == 1 {
            let guarded_power = power.shl(LIMB_BITS);
            let guarded_product = storage.mul(&denominator);
            prop_assert!(guarded_product <= guarded_power);
            prop_assert!(guarded_power.sub(&guarded_product) < denominator.mul(&InternalMpUint::from_limb(6)));
        }
    }

    #[test]
    fn reciprocal_is_a_lower_approximation(
        mut limbs in proptest::collection::vec(any::<Limb>(), 1..=100),
    ) {
        *limbs.last_mut().expect("nonempty divisor") |= 1 << Limb::BITS.wrapping_sub(1);
        let divisor = InternalMpUint::from_limbs(limbs);
        let mut scratch = DivScratch::default();
        let reciprocal = Division::newton_reciprocal(divisor.limbs(), &mut scratch);
        let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; divisor.limbs().len().checked_mul(2).expect("bounded reciprocal numerator")]);
        let mut exact = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        let _ = Division::algorithm_d::<true, true, false, false>(
            numerator.limbs(), divisor.limbs(), &mut exact, &mut remainder, &mut scratch,
        );
        prop_assert!(reciprocal <= exact);
        prop_assert!(exact.sub(&reciprocal) <= InternalMpUint::from_limb(2));
    }
}

/// Checks guarded reciprocal bounds while the retained divisor allocation grows
/// from inline to heap storage, then serves shorter padded operands.
#[test]
fn guarded_reciprocal_preserves_bounds_across_buffer_reuse() {
    let mut scratch = DivScratch::default();
    for n in [2_usize, 65, 5, 1] {
        for limbs in tight_bound_denominators(n) {
            let denominator = InternalMpUint::from_limbs(limbs);
            let (storage, discarded) =
                Division::newton_block_reciprocal::<true>(denominator.limbs(), n, &mut scratch);
            assert_eq!(discarded, 1);
            assert_eq!(
                storage.limbs().len(),
                n.checked_add(2).expect("guarded width")
            );
            assert_eq!(storage.limbs().last(), Some(&1));
            let exponent = n
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .expect("bounded reciprocal power");
            let power = power_of_base(exponent);
            assert!(storage.mul(&denominator) <= power);
            assert!(power < storage.add(&InternalMpUint::from_limb(2)).mul(&denominator));
        }
    }
}

/// Checks exact seeds when the complemented numerator has zero high limbs.
#[test]
fn reciprocal_basecase_matches_exact_division_at_normalization_boundaries() {
    let mut scratch = DivScratch::default();
    for n in [1_usize, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
        let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n * 2]);
        for (pattern, limbs) in tight_bound_denominators(n).into_iter().enumerate() {
            let divisor = InternalMpUint::from_limbs(limbs);
            let reciprocal = Division::reciprocal_basecase(divisor.limbs(), &mut scratch);
            let mut exact = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let _ = Division::algorithm_d::<true, false, false, false>(
                numerator.limbs(),
                divisor.limbs(),
                &mut exact,
                &mut remainder,
                &mut DivScratch::default(),
            );
            assert_eq!(reciprocal, exact, "{n} limbs, divisor pattern {pattern}");
            assert_eq!(reciprocal.limbs().len(), n + 1);
            assert_eq!(reciprocal.limbs().last(), Some(&1));
            scratch.dummy_rem = reciprocal;
        }
    }
}

/// Checks V*D <= B^(2n) < (V+2)*D and rejects V-2 and V+2.
#[test]
fn newton_reciprocal_satisfies_its_contract_and_rejects_neighbours() {
    for n in [66_usize, 129, 130, 200, 512, 1_000, 2_000, 4_096] {
        if cfg!(miri) && n > 130 {
            continue;
        }
        let divisor = InternalMpUint::from_limbs(
            (0..n)
                .map(|i| {
                    Limb::MAX.wrapping_sub(i.wrapping_mul(40_503))
                        | (1 << Limb::BITS.wrapping_sub(1))
                })
                .collect(),
        );
        let value = Division::newton_reciprocal(divisor.limbs(), &mut DivScratch::default());
        let power = power_of_base(n * 2);
        let product = value.mul(&divisor);
        assert!(product <= power);
        assert!(power < value.add(&InternalMpUint::from_limb(2)).mul(&divisor));
        let below = value.sub(&InternalMpUint::from_limb(2));
        assert!(below.add(&InternalMpUint::from_limb(2)).mul(&divisor) <= power);
        let above = value.add(&InternalMpUint::from_limb(2));
        assert!(above.mul(&divisor) > power);
    }
}

/// For D=B^n/2 the basecase reciprocal is exactly one unit below B^(2n)/D.
#[test]
fn newton_reciprocal_contract_holds_at_the_one_unit_basecase() {
    for n in [4_usize, 5, 66, 129] {
        let mut limbs = alloc::vec![0; n];
        *limbs.last_mut().expect("nonempty divisor") = 1 << Limb::BITS.wrapping_sub(1);
        let divisor = InternalMpUint::from_limbs(limbs);
        let value = Division::newton_reciprocal(divisor.limbs(), &mut DivScratch::default());
        let power = power_of_base(n * 2);
        assert!(value.mul(&divisor) <= power);
        assert!(power < value.add(&InternalMpUint::from_limb(2)).mul(&divisor));
    }
}

/// Checks the parity-dependent reciprocal bound without machine-width powers.
#[test]
fn newton_reciprocal_error_bound() {
    for n in [
        4_usize, 5, 64, 65, 127, 129, 131, 200, 257, 512, 513, 1_000, 2_000, 4_095, 4_096, 4_097,
    ] {
        if cfg!(miri) && n > 65 {
            continue;
        }
        for limbs in tight_bound_denominators(n) {
            let divisor = InternalMpUint::from_limbs(limbs);
            let value = Division::newton_reciprocal(divisor.limbs(), &mut DivScratch::default());
            assert_eq!(
                value.limbs().len(),
                n.checked_add(1).expect("reciprocal width")
            );
            assert_eq!(value.limbs().last(), Some(&1));
            let power = power_of_base(n * 2);
            let product = value.mul(&divisor);
            assert!(product <= power);
            let left = power
                .sub(&product)
                .mul(&InternalMpUint::from_limbs(alloc::vec![0, 0, 1]));
            let factor = if n.is_multiple_of(2) {
                alloc::vec![4, 2, 1]
            } else {
                alloc::vec![0, 6, 1]
            };
            let right = divisor.mul(&InternalMpUint::from_limbs(factor));
            assert!(left < right, "reciprocal bound at {n} limbs");
        }
    }
}

/// Checks B*(B^(n+b)-V_b*D) < (B+6)*D.
#[test]
fn newton_block_reciprocal_bound() {
    let mut scratch = DivScratch::default();
    for n in [4_usize, 5, 64, 65, 129, 200, 512, 1_000] {
        if cfg!(miri) && n > 65 {
            continue;
        }
        let denominator = InternalMpUint::from_limbs(
            tight_bound_denominators(n)
                .get(3)
                .expect("dense pattern")
                .clone(),
        );
        for (block, guard_full) in [
            (1_usize, false),
            (n.div_ceil(2), false),
            (n, false),
            (n, true),
        ] {
            let (storage, discarded) = if guard_full {
                Division::newton_block_reciprocal::<true>(denominator.limbs(), block, &mut scratch)
            } else {
                Division::newton_block_reciprocal::<false>(denominator.limbs(), block, &mut scratch)
            };
            let reciprocal = InternalMpUint::from_limbs_slice(
                storage
                    .limbs()
                    .get(discarded..)
                    .expect("guarded reciprocal"),
            );
            let power = power_of_base(n + block);
            let product = reciprocal.mul(&denominator);
            assert!(product <= power);
            let left = power
                .sub(&product)
                .mul(&InternalMpUint::from_limbs(alloc::vec![0, 1]));
            let right = denominator.mul(&InternalMpUint::from_limbs(alloc::vec![6, 1]));
            assert!(left < right, "block bound at {n}/{block} limbs");
        }
    }
}

fn power_of_base(exponent: usize) -> InternalMpUint {
    let mut limbs = alloc::vec![0; exponent.checked_add(1).expect("power width")];
    *limbs.last_mut().expect("nonempty power") = 1;
    InternalMpUint::from_limbs(limbs)
}

fn tight_bound_denominators(n: usize) -> [alloc::vec::Vec<Limb>; 5] {
    let high_bit = Limb::from(1_u8) << LIMB_BITS.wrapping_sub(1);
    let mut half = alloc::vec![0; n];
    *half.last_mut().expect("nonempty width") = high_bit;
    let mut half_plus_one = half.clone();
    *half_plus_one.first_mut().expect("nonempty width") |= 1;
    let mersenne = alloc::vec![Limb::MAX; n];
    let mut dense = alloc::vec![Limb::MAX; n];
    for (index, limb) in dense.iter_mut().enumerate() {
        *limb = Limb::MAX.wrapping_sub(index.wrapping_mul(40_503));
    }
    *dense.last_mut().expect("nonempty width") |= high_bit;
    let mut sparse = alloc::vec![0; n];
    *sparse.last_mut().expect("nonempty width") = high_bit | 1;
    if n > 1 {
        *sparse
            .get_mut(n.checked_sub(2).expect("at least two limbs"))
            .expect("second high limb") = Limb::MAX;
    }
    [half, half_plus_one, mersenne, dense, sparse]
}
