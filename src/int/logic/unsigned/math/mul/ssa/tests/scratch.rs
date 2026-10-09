//! Prepared plans and caller-supplied scratch sizing.

use super::*;

#[cfg_attr(
    miri,
    ignore = "large capacity planning is covered by native execution"
)]
#[test]
fn direct_product_scratch_matches_capacity_bound_plans() {
    let len = 1_048_576;
    let workers = NonZeroUsize::new(8).expect("nonzero worker budget");
    let scratch_len = Ssa::mul_scratch_len_for_parallelism(len, len, workers.get());
    let mut operand = vec![0; len];
    for active in [1, len] {
        operand
            .get_mut(..active)
            .expect("active prefix fits")
            .fill(Limb::MAX);
        let plan =
            SsaMultiplicationPlan::try_new(&operand, &operand, TransformChoice::PLANNED, workers)
                .expect("representable capacity-bound plan");
        assert_eq!(scratch_len, plan.scratch_len, "active width {active}");
    }
}

#[test]
fn supplied_ssa_scratch_is_executor_sized_and_never_replaced() {
    let len = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS).wrapping_add(1);
    let mut a = vec![Limb::MAX; len];
    let mut b = vec![Limb::MAX.wrapping_sub(1); len];
    let high_bit = Limb::from(1_u8).wrapping_shl(Limb::BITS.wrapping_sub(1));
    *a.last_mut().expect("nonempty operand") |= high_bit;
    *b.last_mut().expect("nonempty operand") |= high_bit;

    let executor = CountingExecutor::default();
    let sequential_len = Ssa::mul_scratch_len_for_parallelism(len, len, 1);
    let parallel_len = Ssa::mul_scratch_len_for_parallelism(len, len, executor.parallelism().get());
    assert!(parallel_len > sequential_len);

    let mut undersized = vec![Limb::MIN; sequential_len];
    let mut untouched = vec![Limb::MAX; len.wrapping_mul(2)];
    assert!(!Ssa::try_mul_with_executor(
        &mut untouched,
        &a,
        &b,
        TransformChoice::FORCED,
        &mut undersized,
        &executor,
    ));
    assert!(untouched.iter().all(|limb| *limb == Limb::MAX));

    let mut exact = vec![Limb::MIN; parallel_len];
    let mut actual = vec![Limb::MIN; len.wrapping_mul(2)];
    let mut expected = vec![Limb::MIN; actual.len()];
    Schoolbook::mul(&mut expected, &a, &b);
    assert!(Ssa::try_mul_with_executor(
        &mut actual,
        &a,
        &b,
        TransformChoice::FORCED,
        &mut exact,
        &executor,
    ));
    assert_eq!(actual, expected);

    let sq_len = len.saturating_mul(3);
    let mut sq_a = vec![Limb::MAX; sq_len];
    *sq_a.last_mut().expect("nonempty operand") |= high_bit;

    let sequential_square_len = Ssa::sqr_scratch_len_for_parallelism(sq_len, 1);
    let parallel_square_len =
        Ssa::sqr_scratch_len_for_parallelism(sq_len, executor.parallelism().get());
    assert!(
        parallel_square_len > sequential_square_len,
        "parallel_square_len: {parallel_square_len}, sequential_square_len: {sequential_square_len}"
    );

    let mut undersized_square = vec![Limb::MIN; sequential_square_len];
    let mut untouched_sq = vec![Limb::MAX; sq_len.wrapping_mul(2)];
    assert!(!Ssa::try_sqr_with_executor(
        &mut untouched_sq,
        &sq_a,
        TransformChoice::FORCED,
        &mut undersized_square,
        &executor,
    ));
    assert!(untouched_sq.iter().all(|limb| *limb == Limb::MAX));

    let mut exact_square = vec![Limb::MIN; parallel_square_len];
    let mut actual_square = vec![Limb::MIN; sq_len.wrapping_mul(2)];
    let mut expected_square = vec![Limb::MIN; actual_square.len()];
    Schoolbook::mul(&mut expected_square, &sq_a, &sq_a);
    assert!(Ssa::try_sqr_with_executor(
        &mut actual_square,
        &sq_a,
        TransformChoice::FORCED,
        &mut exact_square,
        &executor,
    ));
    assert_eq!(actual_square, expected_square);
}

#[cfg(feature = "_internal-tune")]
#[test]
fn forced_direct_shared_products_validate_the_selected_arena_before_writes() {
    let a = vec![Limb::MAX; 5];
    let b = vec![Limb::MAX - 1; 5];
    let x = vec![Limb::MAX - 2; 5];
    let half = SsaPlan::best_crt_half_width_for_operands(5, 5, SsaOperation::Pair)
        .expect("the small balanced pair has a legal ring");
    let required = FftPlan::new_for_pair(2 * half * LIMB_BITS).transform_mul_two_by_one_scratch(1);
    let mut arena = vec![Limb::MAX; required];
    let mut out_a = vec![Limb::MAX; 10];
    let mut out_b = vec![Limb::MAX; 10];
    assert!(
        !Ssa::try_mul_two_by_one_with_executor(
            &mut out_a,
            &mut out_b,
            &a,
            &b,
            &x,
            TransformChoice::FORCED_DIRECT_FERMAT,
            arena
                .get_mut(..required - 1)
                .expect("the shorter prefix lies inside the arena"),
            &SequentialExecutor,
        ),
        "the selected direct arena is validated before execution"
    );
    assert_eq!(out_a, vec![Limb::MAX; 10], "first output remains untouched");
    assert_eq!(
        out_b,
        vec![Limb::MAX; 10],
        "second output remains untouched"
    );
    assert!(
        Ssa::try_mul_two_by_one_with_executor(
            &mut out_a,
            &mut out_b,
            &a,
            &b,
            &x,
            TransformChoice::FORCED_DIRECT_FERMAT,
            &mut arena,
            &SequentialExecutor,
        ),
        "the exact selected arena executes the shared direct product"
    );
    let mut expected = vec![0; 10];
    Schoolbook::mul(&mut expected, &a, &x);
    assert_eq!(out_a, expected, "first exact direct product");
    Schoolbook::mul(&mut expected, &b, &x);
    assert_eq!(out_b, expected, "second exact direct product");
}
