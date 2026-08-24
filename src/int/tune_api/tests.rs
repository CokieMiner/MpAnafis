//! Regression tests for reusable tuner execution state.

#![cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
#![expect(
    clippy::panic,
    reason = "test-only pool construction and fixed geometry failures are explicit invariants"
)]

use alloc::vec;

use rayon::ThreadPoolBuilder;

use super::{
    Limb, MultiplicationAlgorithm, Schoolbook, SquaringAlgorithm, Tuner,
    tier::transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

#[test]
fn prepared_ssa_runners_keep_their_scratch_width_across_rayon_pools() {
    const LEN: usize = 64;
    const PRODUCT_LEN: usize = 128;

    let mut a = vec![Limb::MAX; LEN];
    let mut b = vec![Limb::MAX; LEN];
    let Some(a_first) = a.first_mut() else {
        panic!("the test operand must be nonempty");
    };
    *a_first = 3;
    let Some(b_first) = b.first_mut() else {
        panic!("the test operand must be nonempty");
    };
    *b_first = 5;

    let mut expected_product = vec![Limb::MIN; PRODUCT_LEN];
    Schoolbook::mul(&mut expected_product, &a, &b);
    let mut expected_square = vec![Limb::MIN; PRODUCT_LEN];
    Schoolbook::sqr(&mut expected_square, &a);

    let Ok(narrow_pool) = ThreadPoolBuilder::new().num_threads(1).build() else {
        panic!("the narrow test pool must build");
    };
    let Ok(wide_pool) = ThreadPoolBuilder::new().num_threads(4).build() else {
        panic!("the wide test pool must build");
    };

    let mut multiplication =
        narrow_pool.install(|| Tuner::multiplication(MultiplicationAlgorithm::SsaForced, LEN, LEN));
    let mut product = vec![Limb::MIN; PRODUCT_LEN];
    wide_pool.install(|| multiplication.run(&mut product, &a, &b));
    assert_eq!(
        product, expected_product,
        "multiplication scratch must retain its planned width"
    );

    let mut squaring = narrow_pool.install(|| Tuner::squaring(SquaringAlgorithm::SsaForced, LEN));
    let mut square = vec![Limb::MIN; PRODUCT_LEN];
    wide_pool.install(|| squaring.run(&mut square, &a));
    assert_eq!(
        square, expected_square,
        "squaring scratch must retain its planned width"
    );

    let Some(mut transform) = narrow_pool.install(|| {
        Tuner::bench_ssa_multiplication(
            SsaGeometryPolicy::Forced,
            TransformExecutor::Default,
            SsaScratchPolicy::Reusable,
            &a,
            &b,
        )
    }) else {
        panic!("the forced SSA test geometry must be valid");
    };
    let mut transformed_product = vec![Limb::MIN; PRODUCT_LEN];
    wide_pool.install(|| transform.prepare(&mut transformed_product).run());
    assert_eq!(
        transformed_product, expected_product,
        "standalone SSA scratch must retain its planned width"
    );
}
