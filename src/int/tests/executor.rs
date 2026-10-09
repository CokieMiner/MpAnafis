//! Multiplication through a scoped Rayon execution pool.

use crate::MpUint;

#[test]
#[cfg_attr(
    miri,
    ignore = "Rayon worker startup reaches a Stacked Borrows violation in crossbeam-epoch"
)]
fn scoped_thread_pool_execution_matches_result() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("build thread pool");

    let left = MpUint::from(987_654_321_u64);
    let right = MpUint::from(123_456_789_u64);

    let pooled_res = pool.install(|| &left * &right);
    assert_eq!(pooled_res, &left * &right);
}
