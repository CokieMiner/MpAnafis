//! Shared-operand transform and Mersenne-recursion tests.

#![expect(
    unsafe_code,
    reason = "Paired-product fixtures initialize complete guarded inputs and disjoint output and worker-sized arena spans"
)]

use super::*;

#[test]
fn shared_products_match_schoolbook_and_reuse_exact_guarded_arenas() {
    let shapes = [
        (1, 2, 1),
        (2, 1, 1),
        (512, 512, 1024),
        (510, 512, 1024),
        (1000, 1020, 2048),
    ]
    .into_iter()
    .chain(
        [
            1, 2, 3, 4, 5, 9, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 192, 2048,
        ]
        .map(|len| (len, len, len)),
    );
    for (len_a, len_b, len_x) in shapes {
        if cfg!(miri) && len_x > 17 {
            continue;
        }
        let a = dense_operand(len_a, 0x9E37_79B9, 7);
        let b = dense_operand(len_b, 0x85EB_CA6B, 11);
        let full_x = dense_operand(len_x, 0xC2B2_AE3D, 5);
        let mut x = full_x.clone();
        let mut expected_a = vec![0; len_a.wrapping_add(len_x)];
        let mut expected_b = vec![0; len_b.wrapping_add(len_x)];

        let mut actual_a = vec![Limb::MAX; expected_a.len() + 2];
        let mut actual_b = vec![Limb::MAX; expected_b.len() + 2];
        let executor = SequentialExecutor;
        let scratch_len = Ssa::mul_two_by_one_scratch_len_for_parallelism(len_a, len_b, len_x, 1);
        assert_ne!(scratch_len, 0, "the shared product admits a ring");
        let mut scratch = vec![Limb::MAX; scratch_len + 2];
        for choice in [TransformChoice::PLANNED, TransformChoice::FORCED] {
            for active_x in [len_x, len_x.div_ceil(2)] {
                x.copy_from_slice(&full_x);
                x.iter_mut().skip(active_x).for_each(|limb| *limb = 0);
                Schoolbook::mul(&mut expected_a, &a, &x);
                Schoolbook::mul(&mut expected_b, &b, &x);
                actual_a.fill(Limb::MAX);
                actual_b.fill(Limb::MAX);
                assert!(
                    Ssa::try_mul_two_by_one_with_executor(
                        actual_a
                            .get_mut(1..=expected_a.len())
                            .expect("first exact output"),
                        actual_b
                            .get_mut(1..=expected_b.len())
                            .expect("second exact output"),
                        &a,
                        &b,
                        &x,
                        choice,
                        scratch
                            .get_mut(1..=scratch_len)
                            .expect("exact queried arena"),
                        &executor,
                    ),
                    "SSA declined ({len_a}, {len_b}, {len_x}) with {choice:?}"
                );
                assert_eq!(
                    actual_a.get(1..=expected_a.len()).expect("first product"),
                    expected_a
                );
                assert_eq!(
                    actual_b.get(1..=expected_b.len()).expect("second product"),
                    expected_b
                );
                for guarded in [&actual_a, &actual_b, &scratch] {
                    assert_eq!(
                        (guarded.first(), guarded.last()),
                        (Some(&Limb::MAX), Some(&Limb::MAX))
                    );
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "This 4096-limb fixture crosses the production recursive Mersenne dispatch threshold"
)]
fn bnm1_recursion_scratch_matches_the_active_executor() {
    const LEN: usize = 4_096;
    let a = dense_operand(LEN, 0x9E37_79B9, 7);
    let b = dense_operand(LEN, 0x85EB_CA6B, 11);
    let executor = CountingExecutor::default();
    let scratch_len = Ssa::mul_scratch_len_for_parallelism(LEN, LEN, executor.parallelism().get());
    let mut scratch = vec![Limb::MIN; scratch_len];
    let mut actual = vec![Limb::MAX; LEN.wrapping_mul(2)];
    let mut expected = vec![Limb::MIN; actual.len()];
    Schoolbook::mul(&mut expected, &a, &b);
    assert!(Ssa::try_mul_with_executor(
        &mut actual,
        &a,
        &b,
        TransformChoice::PLANNED,
        &mut scratch,
        &executor,
    ));
    assert_eq!(actual, expected);
}

#[test]
fn fused_bnm1_matches_independent_products() {
    let mut widths = vec![
        1,
        2,
        4,
        SSA_BNM1_BASECASE_LIMBS.wrapping_sub(1),
        SSA_BNM1_BASECASE_LIMBS,
        SSA_BNM1_BASECASE_LIMBS.wrapping_mul(2),
        64,
        128,
        256,
        1024,
        4096,
    ];
    widths.sort_unstable();
    widths.dedup();
    for n in widths {
        if cfg!(miri) && n > 4 {
            continue;
        }
        let a = dense_operand(n, 0x9E37_79B9, 7);
        let b = dense_operand(n, 0x85EB_CA6B, 11);
        let x = dense_operand(n, 0xC2B2_AE3D, 5);
        let executor = CountingExecutor::default();
        let parallelism = executor.parallelism().get();
        let mut dst_a = vec![0; n];
        let mut dst_b = vec![0; n];
        let scratch_len =
            SsaCrt::mul_mod_bnm1_two_by_one_scratch_len_for_parallelism(n, parallelism);
        let mut scratch = vec![Limb::MAX; scratch_len];
        SsaCrt::mul_mod_bnm1_two_by_one(
            &mut dst_a,
            &mut dst_b,
            &a,
            &b,
            &x,
            &mut scratch,
            &executor,
        );

        let mut reference_a = vec![0; n];
        let mut reference_b = vec![0; n];
        let single_len = SsaCrt::mul_mod_bnm1_scratch_len_for_parallelism(n, parallelism);
        let mut single_scratch = vec![0; single_len];
        mul_mod_bnm1(&mut reference_a, &a, &x, &mut single_scratch, &executor);
        mul_mod_bnm1(&mut reference_b, &b, &x, &mut single_scratch, &executor);
        assert_eq!(dst_a, reference_a, "first product at {n} limbs");
        assert_eq!(dst_b, reference_b, "second product at {n} limbs");

        // Reuse the exact dirty arena with the two ordinary operands exchanged.
        // The first fold must release the shared product buffer before reuse.
        SsaCrt::mul_mod_bnm1_two_by_one(
            &mut dst_a,
            &mut dst_b,
            &b,
            &a,
            &x,
            &mut scratch,
            &executor,
        );
        assert_eq!(dst_a, reference_b, "reused first product at {n} limbs");
        assert_eq!(dst_b, reference_a, "reused second product at {n} limbs");
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Production shared-product dispatch crossovers use native fixtures up to 2048 limbs"
)]
fn production_dispatch_matches_independent_products() {
    const SHAPES: [(usize, usize, usize); 8] = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 1024),
        (510, 512, 1024),
        (512, 510, 1024),
        (1024, 1024, 2048),
        (1000, 1020, 2048),
        (2048, 2048, 2048),
    ];
    for (len_a, len_b, len_x) in SHAPES {
        let a = sequence_operand(len_a, 1_234_567);
        let b = sequence_operand(len_b, 7_654_321);
        let x = sequence_operand(len_x, 9_876_543);
        let mut expected_a = vec![0; a.len() + x.len()];
        let mut expected_b = vec![0; b.len() + x.len()];
        let mut actual_a = vec![0; expected_a.len()];
        let mut actual_b = vec![0; expected_b.len()];
        let mut scratch = crate::int::logic::unsigned::math::mul::MulScratch::default();
        Multiplication::mul_limbs_with_scratch(&a, &x, &mut expected_a, &mut scratch);
        Multiplication::mul_limbs_with_scratch(&b, &x, &mut expected_b, &mut scratch);
        Multiplication::mul_two_by_one(&a, &b, &x, &mut actual_a, &mut actual_b, &mut scratch);
        assert_eq!(actual_a, expected_a);
        assert_eq!(actual_b, expected_b);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The recursive Mersenne arena fixture uses 4096-limb inputs; small exact-arena cases remain enabled"
)]
fn direct_bnm1_call_uses_executor_sized_scratch() {
    const N: usize = 4096;
    let a = dense_operand(N, 0x9E37_79B9, 7);
    let b = dense_operand(N, 0x85EB_CA6B, 11);
    let mut dst = vec![0; N];
    let executor = CountingExecutor::default();
    let scratch_len =
        SsaCrt::mul_mod_bnm1_scratch_len_for_parallelism(N, executor.parallelism().get());
    let mut scratch = vec![0; scratch_len];
    mul_mod_bnm1(&mut dst, &a, &b, &mut scratch, &executor);

    let mut product = vec![0; 2 * N];
    Schoolbook::mul(&mut product, &a, &b);
    let (low, high) = product.split_at(N);
    let mut expected = low.to_vec();
    let mut carry = Addition::add_slice_in_place(&mut expected, high);
    if carry > 0 {
        carry = SsaCarry::add_full_in_place(&mut expected, &[carry]);
        if carry > 0 {
            let _ = SsaCarry::add_full_in_place(&mut expected, &[carry]);
        }
    }
    assert_eq!(dst, expected);
}

fn mul_mod_bnm1<E: ParallelExecutor>(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
    executor: &E,
) {
    let plan = CrtMulPlan::new(a.len()).expect("representable test Mersenne geometry");
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), dst.len());
    assert!(
        scratch.len()
            >= SsaCrt::mul_mod_bnm1_scratch_len_for_parallelism(
                a.len(),
                executor.parallelism().get()
            )
    );
    // SAFETY: the test helper validates all operand, destination, scratch,
    // executor, and halving dimensions before executing the retained tree.
    unsafe {
        SsaCrt::mul_mod_bnm1_prepared(dst, a, b, scratch, executor, &plan);
    }
}

fn dense_operand(len: usize, multiplier: Limb, rotation: u32) -> Vec<Limb> {
    (0..len)
        .map(|index| {
            Limb::MAX
                .wrapping_sub(index.wrapping_mul(multiplier))
                .rotate_left(rotation)
        })
        .collect()
}

fn sequence_operand(len: usize, multiplier: Limb) -> Vec<Limb> {
    (0..len)
        .map(|index| {
            let value = Limb::try_from(index & 0xffff).expect("16 bits fit every Limb");
            value.wrapping_mul(multiplier) | 1
        })
        .collect()
}
