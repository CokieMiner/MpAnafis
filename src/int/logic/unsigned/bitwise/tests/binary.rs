//! Logical limb oracles and explicit bit and byte permutations.

use alloc::{vec, vec::Vec};

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use crate::BoundedPrecision;

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn logical_operations_match_limb_oracles_and_finite_width_complements() {
    let check = |left: &[Limb], right: &[Limb], width: usize| {
        let a = InternalMpUint::from_limbs_slice(left);
        let b = InternalMpUint::from_limbs_slice(right);
        let span = left.len().max(right.len());
        let and: Vec<_> = (0..span)
            .map(|index| {
                left.get(index).copied().unwrap_or(0) & right.get(index).copied().unwrap_or(0)
            })
            .collect();
        let or: Vec<_> = (0..span)
            .map(|index| {
                left.get(index).copied().unwrap_or(0) | right.get(index).copied().unwrap_or(0)
            })
            .collect();
        let xor: Vec<_> = (0..span)
            .map(|index| {
                left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0)
            })
            .collect();
        for (actual, reverse, expected) in [
            (a.bitand(&b), b.bitand(&a), and),
            (a.bitor(&b), b.bitor(&a), or),
            (a.bitxor(&b), b.bitxor(&a), xor),
        ] {
            assert_eq!(actual, InternalMpUint::from_limbs(expected));
            assert_eq!(actual, reverse);
        }
        assert_eq!(a.bitand(&a), a);
        assert_eq!(a.bitor(&a), a);
        assert_eq!(a.bitxor(&a), InternalMpUint::zero());
        let mask = InternalMpUint::max_for_bits(width);
        let complement = a.not(width);
        for bit in 0..width {
            assert_eq!(complement.get_bit(bit), !a.get_bit(bit));
        }
        assert!(
            complement.fits_in_bits(width),
            "complement is limited to the supplied width"
        );
        assert_eq!(complement.not(width), a.bitand(&mask));
        assert_eq!(a.bitand(&b).not(width), a.not(width).bitor(&b.not(width)));
        if a.limbs().len().min(b.limbs().len()) <= INLINE_LIMBS {
            assert_eq!(a.bitand(&b).capacity(), INLINE_LIMBS);
        }
        if width
            <= LIMB_BITS
                .checked_mul(INLINE_LIMBS)
                .expect("inline bit count")
        {
            assert_eq!(complement.capacity(), INLINE_LIMBS);
        }
    };
    for left in [0, 1, 3, 4, 5, 16] {
        for right in [0, 1, 3, 4, 5] {
            for width in [
                1,
                LIMB_BITS.checked_sub(1).expect("positive limb width"),
                LIMB_BITS,
                LIMB_BITS
                    .checked_mul(4)
                    .and_then(|bits| bits.checked_add(1))
                    .expect("heap width"),
            ] {
                check(&vec![Limb::MAX; left], &vec![Limb::MAX; right], width);
            }
        }
    }
    let limit = if cfg!(miri) { 5 } else { 32 };
    let strategy = (
        collection::vec(any::<Limb>(), 0..=limit),
        collection::vec(any::<Limb>(), 0..=limit),
        1_usize..=LIMB_BITS.checked_mul(8).expect("bounded bit width"),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(left, right, width)| {
            check(&left, &right, width);
            Ok(())
        })
        .expect("logical limb property");
}

#[test]
fn rotations_reversal_and_byte_swap_match_explicit_permutations() {
    let check = |limbs: &[Limb], width: usize, shift: u32| {
        let value = InternalMpUint::from_limbs_slice(limbs);
        let expected = value.bitand(&InternalMpUint::max_for_bits(width));
        let reversed = value.reverse_bits(width);
        let precision = BoundedPrecision::new(width).expect("positive bounded window");
        let left = value.rotate_left(shift, precision);
        let right = value.rotate_right(shift, precision);
        let amount = usize::try_from(
            u64::from(shift)
                .checked_rem(u64::try_from(width).expect("bounded width"))
                .expect("positive width"),
        )
        .expect("the residue fits usize");
        for bit in 0..width {
            let reverse_index = width
                .checked_sub(1)
                .and_then(|top| top.checked_sub(bit))
                .expect("in-window bit");
            let left_source = bit
                .checked_add(width)
                .and_then(|index| index.checked_sub(amount))
                .expect("bounded index")
                .rem_euclid(width);
            let right_source = bit
                .checked_add(amount)
                .expect("bounded index")
                .rem_euclid(width);
            assert_eq!(reversed.get_bit(bit), value.get_bit(reverse_index));
            assert_eq!(left.get_bit(bit), value.get_bit(left_source));
            assert_eq!(right.get_bit(bit), value.get_bit(right_source));
        }
        assert!(
            reversed.fits_in_bits(width) && left.fits_in_bits(width) && right.fits_in_bits(width),
            "permutations stay in their window"
        );
        assert_eq!(reversed.reverse_bits(width), expected);
        assert_eq!(left.rotate_right(shift, precision), expected);
        let mut bytes = value.to_le_bytes();
        bytes.resize(width.div_ceil(8), 0);
        bytes.reverse();
        assert_eq!(
            value.swap_bytes(Some(width)),
            InternalMpUint::from_le_bytes(&bytes)
        );
    };
    for width in [
        1,
        3,
        LIMB_BITS.checked_sub(1).expect("positive limb width"),
        LIMB_BITS,
        LIMB_BITS.checked_add(1).expect("partial limb"),
        LIMB_BITS.checked_mul(5).expect("heap window"),
    ] {
        for shift in [0, 1, 65_536, u32::MAX] {
            for limbs in [&[][..], &[1][..], &[8][..], &[Limb::MAX; 5][..]] {
                check(limbs, width, shift);
            }
        }
    }
    let reversed_zero = InternalMpUint::zero().reverse_bits(usize::MAX);
    assert!(
        reversed_zero.is_zero(),
        "reversing zero has no significant output"
    );
    assert_eq!(reversed_zero.capacity(), INLINE_LIMBS);
    let strategy = (
        collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 20 }),
        1_usize..=LIMB_BITS.checked_mul(20).expect("bounded window"),
        any::<u32>(),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, width, shift)| {
            check(&limbs, width, shift);
            Ok(())
        })
        .expect("bit permutation property");
}
