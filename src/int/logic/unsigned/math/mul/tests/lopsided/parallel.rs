//! Native worker budgets, sequential scratch fallback, and transform blocks.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use crate::{
    int::logic::unsigned::math::mul::{Limb, Lopsided, Schoolbook},
    parallel::DefaultExecutor,
};

#[test]
#[cfg_attr(miri, ignore = "Rayon worker pools require native thread execution")]
fn worker_budgets_preserve_products_with_full_or_sequential_workspace() {
    let smaller_len = 32_usize;
    let larger_len = 333_usize;
    let block = NonZeroUsize::new(smaller_len).expect("nonempty block");
    let left: Vec<Limb> = (0..larger_len)
        .map(|index| index.wrapping_mul(37).wrapping_add(1))
        .collect();
    let right = vec![Limb::MAX; smaller_len];
    let width = larger_len
        .checked_add(smaller_len)
        .expect("test product fits");
    let mut expected = vec![0; width];
    Schoolbook::mul(&mut expected, &left, &right);
    for workers in [1, 2, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test pool");
        pool.install(|| {
            DefaultExecutor::with_resolved(|executor| {
                for regions in [1, workers] {
                    let mut scratch =
                        vec![
                            Limb::MAX;
                            Lopsided::mul_scratch_len(larger_len, smaller_len, block, regions)
                        ];
                    let mut output = vec![Limb::MAX; width.checked_add(3).expect("guards fit")];
                    for _ in 0..2 {
                        scratch.fill(Limb::MAX);
                        Lopsided::mul(&mut output, &left, &right, &mut scratch, block, executor);
                        let (product, guards) = output.split_at(width);
                        assert_eq!(product, expected, "workers {workers}, regions {regions}");
                        assert_eq!(guards, &[Limb::MAX; 3]);
                    }
                }
            });
        });
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Wide transform blocks and Rayon pools require native execution"
)]
fn transform_blocks_use_exact_dirty_workspace_for_each_worker_budget() {
    let smaller_len = 4096_usize;
    let block = NonZeroUsize::new(smaller_len).expect("nonempty block");
    let larger_len = smaller_len
        .checked_mul(4)
        .and_then(|width| width.checked_add(17))
        .expect("test width fits");
    let left: Vec<Limb> = (0..larger_len)
        .map(|index| index.wrapping_mul(37).wrapping_add(1))
        .collect();
    let right = vec![Limb::MAX; smaller_len];
    let width = larger_len
        .checked_add(smaller_len)
        .expect("test product fits");
    let mut expected = vec![0; width];
    Schoolbook::mul(&mut expected, &left, &right);
    for workers in [1, 2, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test pool");
        pool.install(|| {
            DefaultExecutor::with_resolved(|executor| {
                let mut scratch =
                    vec![
                        Limb::MAX;
                        Lopsided::mul_scratch_len(larger_len, smaller_len, block, workers)
                    ];
                let mut output = vec![Limb::MAX; width];
                for _ in 0..2 {
                    Lopsided::mul(&mut output, &left, &right, &mut scratch, block, executor);
                    assert_eq!(output, expected, "workers {workers}");
                }
            });
        });
    }
}
