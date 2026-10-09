//! Worker-region partitioning with complete and partial final batches.

use core::num::NonZeroUsize;

use alloc::vec;

use crate::{
    int::logic::unsigned::math::mul::{Limb, Lopsided, Schoolbook},
    parallel::ParallelExecutor,
};

#[derive(Debug)]
struct BatchExecutor(NonZeroUsize);

impl ParallelExecutor for BatchExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        self.0
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (left(), right())
    }
}

#[test]
fn worker_partitions_cover_complete_and_partial_tails() {
    let block_len = 3_usize;
    let block_width = NonZeroUsize::new(block_len).expect("nonempty block");
    let smaller = vec![Limb::MAX; 4];
    for workers in [1, 2, 3, 4] {
        let executor = BatchExecutor(NonZeroUsize::new(workers).expect("positive budget"));
        for full_blocks in 2_usize..=8 {
            for tail in 0..block_len {
                let larger_len = full_blocks
                    .checked_mul(block_len)
                    .and_then(|width| width.checked_add(tail))
                    .expect("test width fits");
                let larger = vec![Limb::MAX; larger_len];
                let width = larger_len
                    .checked_add(smaller.len())
                    .expect("test product fits");
                let mut expected = vec![0; width];
                Schoolbook::mul(&mut expected, &larger, &smaller);
                let mut scratch =
                    vec![
                        Limb::MAX;
                        Lopsided::mul_scratch_len(larger_len, smaller.len(), block_width, workers)
                    ];
                let mut output = vec![Limb::MAX; width.checked_add(2).expect("guards fit")];
                for _ in 0..2 {
                    let (_, after_prefix) = output.split_at_mut(1);
                    let (product, _) = after_prefix.split_at_mut(width);
                    Lopsided::mul(
                        product,
                        &larger,
                        &smaller,
                        &mut scratch,
                        block_width,
                        &executor,
                    );
                    assert_eq!(product, expected, "budget {workers}, tail {tail}");
                    assert_eq!(output.first(), Some(&Limb::MAX));
                    assert_eq!(output.last(), Some(&Limb::MAX));
                }
            }
        }
    }
}
