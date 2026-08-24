use proptest::prelude::*;

use super::*;

fn limb_vec(max_len: usize) -> impl Strategy<Value = Vec<Limb>> {
    proptest::collection::vec(any::<usize>(), 0..=max_len)
}

fn normalized_limb_vec(max_len: usize) -> impl Strategy<Value = Vec<Limb>> {
    limb_vec(max_len).prop_map(|mut limbs| {
        while limbs.last() == Some(&0) {
            limbs.truncate(limbs.len().wrapping_sub(1));
        }
        limbs
    })
}

proptest! {
    #[test]
    fn prop_from_limbs_trims_trailing_zero_limbs(input_limbs in limb_vec(12)) {
        let value = InternalMpUint::from_limbs(input_limbs.clone());
        let mut expected_limbs = input_limbs;
        while expected_limbs.last() == Some(&0) {
            expected_limbs.truncate(expected_limbs.len().wrapping_sub(1));
        }
        prop_assert_eq!(value.limbs(), expected_limbs.as_slice());

        match &value.repr {
            UintRepr::Inline { len, .. } => {
                prop_assert!(usize::from(*len) <= INLINE_LIMBS);
                prop_assert!(expected_limbs.len() <= INLINE_LIMBS);
            }
            UintRepr::Heap(limbs) => {
                prop_assert!(limbs.len() > INLINE_LIMBS);
                prop_assert_eq!(limbs.as_slice(), expected_limbs.as_slice());
            }
        }
    }

    #[test]
    fn prop_from_limbs_normalized_preserves_exact_normalized_limbs(
        input_limbs in normalized_limb_vec(12).prop_filter("non-empty", |limbs| !limbs.is_empty())
    ) {
        // SAFETY: the generator removes empty inputs and high zero limbs.
        let value = unsafe { InternalMpUint::from_limbs_normalized(input_limbs.clone()) };
        prop_assert_eq!(value.limbs(), input_limbs.as_slice());
        prop_assert_eq!(value.limbs().last().copied(), input_limbs.last().copied());
    }

    #[test]
    fn prop_clone_from_slice_matches_from_limbs(input_limbs in limb_vec(12)) {
        let mut cloned_value = InternalMpUint::one();
        cloned_value.clone_from_slice(&input_limbs);
        let expected_value = InternalMpUint::from_limbs(input_limbs);
        prop_assert_eq!(cloned_value.limbs(), expected_value.limbs());
        prop_assert_eq!(cloned_value.limbs().len(), expected_value.limbs().len());
    }

    #[test]
    fn prop_increment_then_decrement_roundtrip(input_limbs in normalized_limb_vec(8)) {
        let mut value = InternalMpUint::from_limbs(input_limbs);
        let original_value = value.clone();
        value.increment();
        value.decrement();
        prop_assert_eq!(value.limbs(), original_value.limbs());
        prop_assert_eq!(value.repr, original_value.repr);
    }

    #[test]
    fn prop_ensure_capacity_set_len_get_limbs_preserves_initialized_prefix(
        input_limbs in normalized_limb_vec(8),
        extra_len in 0_usize..=6,
    ) {
        let mut value = InternalMpUint::from_limbs(input_limbs);
        let target_len = value.limbs().len().wrapping_add(extra_len);
        let original_prefix = value.limbs().to_vec();

        // SAFETY: the initialized prefix is preserved and the newly exposed
        // suffix is initialized before normalization.
        let slice = unsafe { value.ensure_capacity_set_len_get_limbs(target_len) };
        let prefix_len = original_prefix.len();
        prop_assert_eq!(
            slice.get(..prefix_len).expect("prefix lies within target"),
            original_prefix.as_slice()
        );
        slice
            .get_mut(prefix_len..)
            .expect("suffix lies within target")
            .fill(0);
        value.normalize();
        prop_assert_eq!(value.limbs(), original_prefix.as_slice());
    }
}
