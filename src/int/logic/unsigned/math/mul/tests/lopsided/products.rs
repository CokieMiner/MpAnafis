//! Arbitrary blocked products with complete and partial final blocks.

use core::{num::NonZeroUsize, ops::Rem};

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::{
    int::logic::unsigned::math::mul::{
        KARATSUBA_THRESHOLD, Limb, Lopsided, Multiplication, Schoolbook,
    },
    parallel::SequentialExecutor,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 24 }))]

    #[test]
    fn arbitrary_products_cover_orders_partial_blocks_and_dirty_reuse(
        smaller_len in if cfg!(miri) { 2_usize..=5 } else { KARATSUBA_THRESHOLD.max(2)..=KARATSUBA_THRESHOLD.max(2).checked_add(24).expect("test width fits") },
        blocks in 8_usize..=12,
        tail_seed in any::<usize>(),
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 64 } else { 1024 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 5 } else { 128 }),
        dirty in any::<Limb>(),
    ) {
        let smaller_width = NonZeroUsize::new(smaller_len).expect("nonempty operand");
        let tail = tail_seed.rem(smaller_width);
        let larger_len = smaller_len.checked_mul(blocks).and_then(|width| width.checked_add(tail)).expect("test operand fits");
        let left = left_words.get(..larger_len).expect("generated prefix fits");
        let right = right_words.get(..smaller_len).expect("generated prefix fits");
        let width = larger_len.checked_add(smaller_len).expect("test product fits");
        let mut expected = vec![0; width];
        Schoolbook::mul(&mut expected, left, right);
        let mut output = vec![dirty; width.checked_add(2).expect("guards fit")];
        let mut scratch = vec![Limb::MAX; Multiplication::required_scratch(larger_len, smaller_len)];
        for (a, b) in [(left, right), (right, left)] {
            for _ in 0..2 {
                let (product, guards) = output.split_at_mut(width);
                Multiplication::mul_limbs_with_slice_scratch(a, b, product, &mut scratch);
                prop_assert_eq!(&*product, expected.as_slice());
                prop_assert_eq!(&*guards, &[dirty; 2]);
            }
        }
        let forced_block = NonZeroUsize::new(smaller_len.div_ceil(2).checked_add(dirty.rem(smaller_width)).expect("bounded block fits")).expect("positive block");
        let mut forced_scratch = vec![Limb::MAX; Lopsided::mul_scratch_len(larger_len, smaller_len, forced_block, 1)];
        let (product, guards) = output.split_at_mut(width);
        product.fill(dirty);
        Lopsided::mul(product, left, right, &mut forced_scratch, forced_block, &SequentialExecutor);
        prop_assert_eq!(&*product, expected.as_slice());
        prop_assert_eq!(&*guards, &[dirty; 2]);
    }
}
