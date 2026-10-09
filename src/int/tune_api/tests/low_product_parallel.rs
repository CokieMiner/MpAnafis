//! Prepared low-product worker budgets across changes of the active Rayon pool.

use alloc::vec;

use rayon::ThreadPoolBuilder;

use crate::tune_api::{Limb, LowProductAlgorithm, LowProductRunner};

use super::strategies::integer;

#[test]
#[cfg_attr(
    miri,
    ignore = "Rayon pool changes and SSA products at 8192 limbs exceed Miri's execution budget"
)]
fn prepared_transform_children_retain_the_sized_worker_budget() {
    let narrow = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one worker");
    let wide = ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("three workers");
    let len = 8_192_usize;
    let bits = len
        .checked_mul(usize::try_from(Limb::BITS).expect("native limb width"))
        .expect("bounded reference width");
    let left = vec![Limb::MAX; len];
    let right = vec![Limb::MAX >> 1; len];
    for (construction_pool, execution_pool) in [(&narrow, &wide), (&wide, &narrow)] {
        for algorithm in [LowProductAlgorithm::Mulders, LowProductAlgorithm::Full] {
            let mut runner = construction_pool.install(|| LowProductRunner::new(algorithm, len));
            for rhs in [&right, &left] {
                let expected = integer(&left)
                    .checked_mul(&integer(rhs))
                    .expect("unlimited reference product")
                    .bit_range(0, bits);
                let mut output = vec![Limb::MAX; len];
                {
                    let mut prepared = runner.prepare(&mut output, &left, rhs);
                    execution_pool.install(|| {
                        prepared.run();
                        prepared.run();
                    });
                }
                assert_eq!(integer(&output), expected);
            }
        }
    }
}
