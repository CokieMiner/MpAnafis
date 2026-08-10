//! Internal storage resize, clone, swap, and reserve properties.

use super::*;

proptest! {
    #[test]
    fn prop_memory_resize(value in strategies::uint(8), new_len in 0_usize..20) {
        let mut resized = value;
        resized.value.resize(new_len);
        prop_assert_eq!(
            resized.value.limbs().len(),
            new_len,
            "resize did not change length correctly"
        );
    }

    #[test]
    fn prop_memory_clone_from(left_value in strategies::uint(8), right_value in strategies::uint(8)) {
        let mut cloned = left_value;
        cloned.value.clone_from(&right_value.value);
        prop_assert_eq!(cloned.value, right_value.value, "clone_from did not match clone");
    }

    #[test]
    fn prop_memory_swap(left_value in strategies::uint(8), right_value in strategies::uint(8)) {
        let mut left_swapped = left_value;
        let mut right_swapped = right_value;
        let left_original = left_swapped.value.clone();
        let right_original = right_swapped.value.clone();

        left_swapped.value.swap(&mut right_swapped.value);
        prop_assert_eq!(&left_swapped.value, &right_original, "swap did not work for a");
        prop_assert_eq!(&right_swapped.value, &left_original, "swap did not work for b");

        left_swapped.value.swap(&mut right_swapped.value);
        prop_assert_eq!(
            &left_swapped.value,
            &left_original,
            "double swap did not restore a"
        );
        prop_assert_eq!(
            &right_swapped.value,
            &right_original,
            "double swap did not restore b"
        );
    }

    #[test]
    fn prop_memory_reserve(value in strategies::uint(8), extra in 0_usize..10) {
        let mut reserved = value;
        let original_capacity = reserved.value.capacity();
        reserved.value.reserve(extra);
        prop_assert!(
            reserved.value.capacity() >= original_capacity.wrapping_add(extra)
                || reserved.value.capacity() >= original_capacity,
            "reserve bounds"
        );
        reserved.value.shrink_to_fit();
        prop_assert!(
            reserved.value.capacity() <= original_capacity.wrapping_add(extra),
            "shrink_to_fit bounds"
        );
    }
}
