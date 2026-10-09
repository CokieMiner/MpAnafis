//! In-place coefficient transforms, exact scratch, and shared-input lifetimes.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Explicit small test geometries bound coefficient, product, and sentinel offsets"
)]

use super::{
    super::product::{PointwiseMulStrategy, PointwiseSquareStrategy},
    *,
};

#[test]
fn dense_transforms_reconstruct_into_consumed_operands() {
    for limbs in [4, 9] {
        for seed in [0, 1, Limb::MAX] {
            check_dense(limbs * LIMB_BITS, seed);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 12 }))]

    #[test]
    fn prop_dense_transforms_reuse_exact_scratch(seed in any::<Limb>(), limbs in 2_usize..=12) {
        check_dense(limbs * LIMB_BITS, seed);
    }
}

fn check_dense(bits: usize, seed: Limb) {
    let ml = SsaRing::mod_limbs(bits);
    let cl = ml + 1;
    let product = MulTransformPlan::new(FftPlan::new(bits));
    let square = SquareTransformPlan::new(FftPlan::new_for_square(bits));
    let left = coefficient(ml, seed, 0);
    let right = coefficient(ml, seed.wrapping_add(1), 1);
    let expected_product = reference_product(&left, &right);
    let expected_square = reference_product(&left, &left);
    let mut actual = vec![Limb::MAX; cl + 2];
    let mut mul_work = vec![Limb::MAX; product.transform_mul_scratch(1) + 2];
    let mut sqr_work = vec![Limb::MAX; square.transform_sqr_scratch(1) + 2];
    let mul_end = mul_work.len() - 1;
    let sqr_end = sqr_work.len() - 1;
    for _ in 0..2 {
        actual[1..=cl].copy_from_slice(&left);
        // SAFETY: the input coefficients have zero guards, and the exact
        // disjoint scratch windows come from their retained sequential plans.
        // The output is the original left allocation, bounded by sentinels.
        unsafe {
            product.mul_assign_left(&mut actual[1..=cl], &right, &mut mul_work[1..mul_end]);
        }
        assert_eq!(&actual[1..=cl], expected_product);
        actual[1..=cl].copy_from_slice(&left);
        // SAFETY: the same guarded input span is disjoint from the exact
        // square arena; each repeated call must overwrite its dirty workspace.
        unsafe {
            square.sqr_assign(&mut actual[1..=cl], &mut sqr_work[1..sqr_end]);
        }
        assert_eq!(&actual[1..=cl], expected_square);
        for buffer in [&actual, &mul_work, &sqr_work] {
            assert_eq!(buffer.first(), Some(&Limb::MAX));
            assert_eq!(buffer.last(), Some(&Limb::MAX));
        }
    }
}

#[test]
fn nested_pointwise_small_geometry_reuses_consumed_inputs() {
    let bits = 4 * LIMB_BITS;
    let product = MulTransformPlan::new(FftPlan::new(bits));
    let square = SquareTransformPlan::new(FftPlan::new_for_square(bits));
    let mul = PointwiseMulPlan {
        bits,
        scratch_len: NonZeroUsize::new(product.transform_mul_scratch(1))
            .expect("nested product has a nonempty arena"),
        strategy: PointwiseMulStrategy::Transform(product.into()),
    };
    let sqr = PointwiseSquarePlan {
        bits,
        scratch_len: NonZeroUsize::new(square.transform_sqr_scratch(1))
            .expect("nested square has a nonempty arena"),
        strategy: PointwiseSquareStrategy::Transform(square.into()),
    };
    check_pointwise(&mul, &sqr, &SequentialExecutor);
}

#[test]
fn pointwise_empty_pair_preserves_dirty_arenas() {
    let bits = 4 * LIMB_BITS;
    let plan = PointwiseMulPlan::from(bits);
    let executor = CountingExecutor::default();
    let leaf_budget = FftPlan::new(bits).pointwise_leaf_count(executor.parallelism().get());
    let mut first = [Limb::MAX; 2];
    let mut second = [Limb::MAX; 2];
    let mut shared = [Limb::MAX; 2];
    let mut arena = vec![Limb::MAX; plan.scratch_len.get() + 2];
    let end = arena.len() - 1;
    // SAFETY: the three disjoint empty windows contain exactly zero
    // coefficients. The enclosing plan supplies its admitted leaf budget,
    // and this empty prefix requires one complete private product arena.
    unsafe {
        SsaPointwise::pointwise_multiply_pair_with_executor(
            [&mut first[1..1], &mut second[1..1], &mut shared[1..1]],
            0,
            leaf_budget,
            &plan,
            &executor,
            &mut arena[1..end],
        );
    }
    for coefficient in [first, second, shared] {
        assert_eq!(
            coefficient,
            [Limb::MAX; 2],
            "empty pair preserves its matrix sentinels"
        );
    }
    assert!(
        arena.iter().all(|&limb| limb == Limb::MAX),
        "empty pair leaves its arena untouched"
    );
    assert_eq!(
        executor.joins.load(Ordering::Relaxed),
        0,
        "empty pair has no parallel work"
    );
}

#[test]
fn pointwise_zero_residues_preserve_exact_dirty_arenas() {
    for ml in [1, 2, 4, 6, SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS) + 1] {
        if cfg!(miri) && ml > 6 {
            continue;
        }
        let bits = ml * LIMB_BITS;
        let cl = ml + 1;
        let mut mul = PointwiseMulPlan::from(bits);
        if ml == 6 {
            let factorized =
                NegacyclicPlan::for_factor(ml, NonZeroUsize::new(3).expect("positive test factor"))
                    .expect("six data limbs admit factor-three multiplication");
            mul.scratch_len = NonZeroUsize::new(cl + factorized.scratch_len)
                .expect("factorized product has a result slot and workspace");
            mul.strategy = PointwiseMulStrategy::Negacyclic(factorized);
        }
        let sqr = PointwiseSquarePlan::from(bits);
        let width = 4 * cl;
        // Both canonical zero and the semi-normal representative B^ml+1
        // normalize to zero. The latter exercises guard correction before
        // the shared classification and the in-place square shortcut.
        for (low, guard) in [(0, 0), (1, 1)] {
            let mut left = vec![Limb::MAX; width + 2];
            let mut second = left.clone();
            let mut shared = left.clone();
            for slot in left[1..=width].chunks_exact_mut(cl) {
                slot[ml] = 0;
            }
            for slot in second[1..=width].chunks_exact_mut(cl) {
                slot[ml] = 1;
            }
            for slot in shared[1..=width].chunks_exact_mut(cl) {
                slot.fill(0);
                slot[0] = low;
                slot[ml] = guard;
            }
            let mut squared = shared.clone();
            let mut mul_work = vec![Limb::MAX; mul.scratch_len.get() + 2];
            let mut sqr_work = vec![Limb::MAX; sqr.scratch_len.get() + 2];
            let mul_end = mul_work.len() - 1;
            let sqr_end = sqr_work.len() - 1;
            // SAFETY: four complete semi-normal coefficient slots in each
            // disjoint matrix; sequential execution owns exactly one strategy
            // arena. Canaries lie outside every matrix and workspace window.
            unsafe {
                SsaPointwise::pointwise_multiply_pair_with_executor(
                    [
                        &mut left[1..=width],
                        &mut second[1..=width],
                        &mut shared[1..=width],
                    ],
                    4,
                    NonZeroUsize::MIN,
                    &mul,
                    &SequentialExecutor,
                    &mut mul_work[1..mul_end],
                );
                SsaPointwise::pointwise_square_with_executor(
                    &mut squared[1..=width],
                    4,
                    NonZeroUsize::MIN,
                    &sqr,
                    &SequentialExecutor,
                    &mut sqr_work[1..sqr_end],
                );
            }
            for matrix in [&left, &second, &shared, &squared] {
                assert!(
                    matrix[1..=width].iter().all(|&limb| limb == 0),
                    "zero residue at ml={ml}"
                );
            }
            for buffer in [&left, &second, &shared, &squared, &mul_work, &sqr_work] {
                assert_eq!(
                    buffer.first(),
                    Some(&Limb::MAX),
                    "leading canary at ml={ml}"
                );
                assert_eq!(
                    buffer.last(),
                    Some(&Limb::MAX),
                    "trailing canary at ml={ml}"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Production pointwise crossover widths are covered by a small forced nested-transform fixture under Miri"
)]
#[expect(
    clippy::panic,
    reason = "the crossover test rejects an unexpected retained strategy"
)]
fn nested_pointwise_products_preserve_shared_inputs_and_exact_arenas() {
    for bits in [
        SSA_BASE_MODULUS_BITS - LIMB_BITS,
        SSA_BASE_MODULUS_BITS,
        SSA_BASE_MODULUS_BITS + LIMB_BITS,
        SSA_BASE_MODULUS_BITS * 2,
    ] {
        let mul = PointwiseMulPlan::from(bits);
        let sqr = PointwiseSquarePlan::from(bits);
        if bits > SSA_BASE_MODULUS_BITS {
            let PointwiseMulStrategy::Transform(product) = &mul.strategy else {
                panic!("the coefficient above the crossover must execute a nested transform");
            };
            let PointwiseSquareStrategy::Transform(square) = &sqr.strategy else {
                panic!("the coefficient above the crossover must execute a nested square");
            };
            assert_eq!(mul.scratch_len.get(), product.transform_mul_scratch(1));
            assert_eq!(sqr.scratch_len.get(), square.transform_sqr_scratch(1));
        }
        check_pointwise(&mul, &sqr, &SequentialExecutor);
        let executor = CountingExecutor::default();
        check_pointwise(&mul, &sqr, &executor);
        assert!(executor.joins.load(Ordering::Relaxed) > 0);
    }
}

fn check_pointwise<E: ParallelExecutor>(
    mul: &PointwiseMulPlan,
    sqr: &PointwiseSquarePlan,
    executor: &E,
) {
    let count = 7;
    let ml = SsaRing::mod_limbs(mul.bits);
    let cl = ml + 1;
    let width = count * cl;
    let matrix = |offset| {
        let mut result = vec![Limb::MAX; width + 2];
        for (index, slot) in result[1..=width].chunks_exact_mut(cl).enumerate() {
            slot.copy_from_slice(&coefficient(ml, 43, (index + offset) % count));
        }
        result
    };
    let mut left = matrix(0);
    let mut second = matrix(3);
    let mut shared = matrix(1);
    let original_shared = shared.clone();
    let mut squared = matrix(2);
    let leaf_budget = FftPlan::new(mul.bits).pointwise_leaf_count(executor.parallelism().get());
    let leaves = leaf_budget.get().min(1 << count.ilog2());
    let mut mul_work = vec![Limb::MAX; leaves * mul.scratch_len.get() + 2];
    let mut sqr_work = vec![Limb::MAX; leaves * sqr.scratch_len.get() + 2];
    let mul_end = mul_work.len() - 1;
    let sqr_end = sqr_work.len() - 1;
    for round in 0..3 {
        let expected = |a: &[Limb], b: &[Limb]| -> Vec<Limb> {
            a[1..=width]
                .chunks_exact(cl)
                .zip(b[1..=width].chunks_exact(cl))
                .flat_map(|(x, y)| reference_product(x, y))
                .collect()
        };
        let expected_left = expected(&left, &shared);
        let expected_second = expected(&second, &shared);
        let expected_square = expected(&squared, &squared);
        // SAFETY: each matrix contains seven complete canonical coefficients.
        // The exact leaf arenas and matrix windows are disjoint and bounded by
        // sentinels. Paired execution may read shared again after overwriting left.
        unsafe {
            if round == 1 {
                SsaPointwise::pointwise_multiply_with_executor(
                    &mut left[1..=width],
                    &mut shared[1..=width],
                    count,
                    leaf_budget,
                    mul,
                    executor,
                    &mut mul_work[1..mul_end],
                );
            } else {
                SsaPointwise::pointwise_multiply_pair_with_executor(
                    [
                        &mut left[1..=width],
                        &mut second[1..=width],
                        &mut shared[1..=width],
                    ],
                    count,
                    leaf_budget,
                    mul,
                    executor,
                    &mut mul_work[1..mul_end],
                );
                assert_eq!(&second[1..=width], expected_second);
            }
            SsaPointwise::pointwise_square_with_executor(
                &mut squared[1..=width],
                count,
                leaf_budget,
                sqr,
                executor,
                &mut sqr_work[1..sqr_end],
            );
        }
        assert_eq!(&left[1..=width], expected_left);
        assert_eq!(&squared[1..=width], expected_square);
        assert_eq!(shared, original_shared);
        for buffer in [&left, &second, &shared, &squared, &mul_work, &sqr_work] {
            assert_eq!(buffer.first(), Some(&Limb::MAX));
            assert_eq!(buffer.last(), Some(&Limb::MAX));
        }
    }
}

fn coefficient(ml: usize, seed: Limb, pattern: usize) -> Vec<Limb> {
    let mut result = vec![0; ml + 1];
    match pattern {
        2 => {}
        3 => result[ml] = 1,
        4 => result[0] = 1,
        5 => result[..ml].fill(Limb::MAX),
        6 => result[ml - 1] = 1 << (LIMB_BITS - 1),
        _ => {
            let mut state = seed;
            for limb in &mut result[..ml] {
                state = state.wrapping_mul(33).wrapping_add(7);
                *limb = state;
            }
        }
    }
    result
}

/// Schoolbook product followed by independent subtraction of its high half.
fn reference_product(a: &[Limb], b: &[Limb]) -> Vec<Limb> {
    let ml = a.len() - 1;
    if a[ml] == 1 {
        return reference_negate(b);
    }
    if b[ml] == 1 {
        return reference_negate(a);
    }
    let mut full = vec![0; ml * 2];
    Schoolbook::mul(&mut full, &a[..ml], &b[..ml]);
    let mut result = vec![0; ml + 1];
    let mut borrow = false;
    for index in 0..ml {
        let (difference, first) = full[index].overflowing_sub(full[index + ml]);
        let (value, second) = difference.overflowing_sub(Limb::from(borrow));
        result[index] = value;
        borrow = first || second;
    }
    // A negative low-high difference needs 2^n+1, so the wrapped subtraction
    // gains one. Carry into the guard represents the canonical residue -1.
    for limb in &mut result {
        let (value, carry) = limb.overflowing_add(Limb::from(borrow));
        *limb = value;
        borrow = carry;
    }
    assert!(!borrow);
    result
}

fn reference_negate(value: &[Limb]) -> Vec<Limb> {
    if value.iter().all(|limb| *limb == 0) {
        return value.to_vec();
    }
    let mut result = vec![0; value.len()];
    let mut borrow = false;
    for (index, (dst, source)) in result.iter_mut().zip(value).enumerate() {
        let modulus_limb = Limb::from(index == 0 || index == value.len() - 1);
        let (difference, first) = modulus_limb.overflowing_sub(*source);
        let (digit, second) = difference.overflowing_sub(Limb::from(borrow));
        *dst = digit;
        borrow = first || second;
    }
    assert!(!borrow);
    result
}
