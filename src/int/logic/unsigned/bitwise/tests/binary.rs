//! Properties for unsigned bitwise logic.

use alloc::vec;

use proptest::prelude::*;

use super::*;

proptest! {
    #[test]
    fn heap_bitwise_paths_prop(
        limbs_a in proptest::collection::vec(any::<Limb>(), 2..=5),
        limbs_b in proptest::collection::vec(any::<Limb>(), 2..=5),
    ) {
        let a = InternalMpUint::from_limbs(limbs_a);
        let b = InternalMpUint::from_limbs(limbs_b);

        let and = a.bitand(&b);
        let or = a.bitor(&b);
        let xor = a.bitxor(&b);

        prop_assert!(and == b.bitand(&a));
        prop_assert_eq!(or, b.bitor(&a));
        prop_assert_eq!(xor, b.bitxor(&a));

        prop_assert!(a.bitand(&a) == a);
        prop_assert!(a.bitor(&a) == a);
        prop_assert!(a.bitxor(&a) == InternalMpUint::zero());

        let width = a.significant_bits().max(b.significant_bits());
        if width > 0 {
            let a_not = a.not(width);
            let b_not = b.not(width);
            let lhs = and.not(width);
            let rhs = a_not.bitor(&b_not);
            let mask = InternalMpUint::max_for_bits(width);
            prop_assert_eq!(lhs.bitand(&mask), rhs.bitand(&mask));
        }
    }

    #[test]
    fn partial_limb_masks_prop(bits in 1_usize..=(LIMB_BITS * 2)) {
        let value = InternalMpUint::zero();
        let masked_not = value.not(bits);
        let expected_limbs = if bits <= LIMB_BITS {
            vec![low_bits_mask(bits)]
        } else {
            let mut limbs = vec![Limb::MAX; bits.div_ceil(LIMB_BITS)];
            let rem = bits % LIMB_BITS;
            if rem > 0 && let Some(last) = limbs.last_mut() {
                *last = low_bits_mask(rem);
            }
            if limbs.len() > 1 && limbs.last() == Some(&0) {
                let _ = limbs.pop();
            }
            limbs
        };
        prop_assert_eq!(
            masked_not,
            InternalMpUint::from_limbs(expected_limbs),
        );
    }
}
