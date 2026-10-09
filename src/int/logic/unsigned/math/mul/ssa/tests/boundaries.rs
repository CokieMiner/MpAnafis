//! Significant-width, exponent, and concurrent execution boundary coverage.

#![expect(
    unsafe_code,
    clippy::integer_division,
    clippy::indexing_slicing,
    reason = "Boundary cases construct exact nonempty capacities and validate sentinel windows explicitly"
)]

use super::*;

#[test]
fn capacity_scratch_covers_significant_width_geometries() {
    for capacity in 2..=if cfg!(miri) { 16 } else { 4096 } {
        for workers in [1, 3, 8] {
            let mul_scratch = Ssa::mul_scratch_len_for_parallelism(capacity, capacity, workers);
            let sqr_scratch = Ssa::sqr_scratch_len_for_parallelism(capacity, workers);
            let mut input = vec![0; capacity];
            for active in [
                1,
                (capacity / 3).max(1),
                capacity / 2,
                capacity - 1,
                capacity,
            ] {
                input.fill(0);
                input
                    .iter_mut()
                    .take(active)
                    .for_each(|limb| *limb = Limb::MAX);
                for top in [1, Limb::MAX] {
                    input[active - 1] = top;
                    let mul = SsaMultiplicationPlan::try_new(
                        &input,
                        &input,
                        TransformChoice::FORCED,
                        NonZeroUsize::new(workers).expect("nonzero workers"),
                    )
                    .expect("representable multiplication");
                    let sqr = SsaSquaringPlan::try_new(&input, TransformChoice::FORCED, workers)
                        .expect("representable square");
                    assert!(
                        mul.scratch_len <= mul_scratch,
                        "mul capacity={capacity}, active={active}, workers={workers}: {} > {mul_scratch}",
                        mul.scratch_len
                    );
                    assert!(
                        sqr.scratch_len <= sqr_scratch,
                        "square capacity={capacity}, active={active}, workers={workers}: {} > {sqr_scratch}",
                        sqr.scratch_len
                    );
                }
            }
        }
    }
}

#[test]
fn every_reduced_shift_matches_repeated_doubling() {
    for ml in [1, 2, 3, 5, 9, 17] {
        if cfg!(miri) && ml > 2 {
            continue;
        }
        let bits = ml * LIMB_BITS;
        let cl = ml + 1;
        for guard in [0, 1] {
            for fill in [0, 1, Limb::MAX] {
                let mut input = vec![fill; cl];
                input[ml] = guard;
                let mut expected = input.clone();
                let mut work = vec![Limb::MAX; cl];
                // SAFETY: complete initialized coefficient in a limb-aligned ring.
                unsafe {
                    let _ = SsaRing::normalize(&mut expected, bits);
                }
                for shift in 0..2 * bits {
                    let mut direct = vec![Limb::MAX; cl];
                    let mut inplace = input.clone();
                    // SAFETY: complete disjoint coefficients and scratch, guard <= 1,
                    // and the loop establishes 0 <= shift < 2*bits.
                    unsafe {
                        SsaRing::shift_from(&mut direct, &input, shift, bits);
                        SsaRing::shift_in_place(&mut inplace, shift, bits, &mut work);
                        let _ = SsaRing::normalize(&mut direct, bits);
                        let _ = SsaRing::normalize(&mut inplace, bits);
                    }
                    assert_eq!(direct, expected, "out-of-place ml={ml}, shift={shift}");
                    assert_eq!(inplace, expected, "in-place ml={ml}, shift={shift}");
                    let previous = expected.clone();
                    // SAFETY: canonical complete disjoint coefficients. Doubling
                    // by ring addition provides an independent shift oracle.
                    unsafe {
                        assert_eq!(SsaCarry::add_full_in_place(&mut expected, &previous), 0);
                        let _ = SsaRing::normalize(&mut expected, bits);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
#[test]
#[cfg_attr(
    miri,
    ignore = "Native Rayon pools and products up to 3073 limbs are covered by small logical-executor tests under Miri"
)]
fn rayon_operations_reuse_exact_dirty_scratch() {
    use crate::parallel::DefaultExecutor;

    for workers in [1, 2, 3, 5, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test thread pool");
        pool.install(|| {
            DefaultExecutor::with_resolved(|executor| {
                for len in [127, 257, 1023, 3073] {
                    let a: Vec<_> = (0..len).map(|i| Limb::MAX.wrapping_sub(i)).collect();
                    let b: Vec<_> = (0..len)
                        .map(|i| i.wrapping_mul(37).wrapping_add(1))
                        .collect();
                    let mut expected = vec![0; 2 * len];
                    Schoolbook::mul(&mut expected, &a, &b);
                    let plan = SsaMultiplicationPlan::try_new(
                        &a,
                        &b,
                        TransformChoice::FORCED,
                        executor.parallelism(),
                    )
                    .expect("representable product");
                    let mut scratch = vec![Limb::MAX; plan.scratch_len];
                    let mut output = vec![Limb::MAX; expected.len() + 2];
                    for _ in 0..2 {
                        // SAFETY: the window and scratch cover the retained plan;
                        // execution uses the exact pool width used at construction.
                        unsafe {
                            plan.run_with_scratch(
                                &mut output[1..=expected.len()],
                                &mut scratch,
                                executor,
                            );
                        }
                        assert_eq!(&output[1..=expected.len()], expected);
                        assert_eq!(output[0], Limb::MAX);
                        assert_eq!(output[expected.len() + 1], Limb::MAX);
                    }
                    let square = SsaSquaringPlan::try_new(&a, TransformChoice::FORCED, workers)
                        .expect("representable square");
                    scratch.resize(square.scratch_len, Limb::MAX);
                    Schoolbook::mul(&mut expected, &a, &a);
                    // SAFETY: complete output, exactly sufficient scratch, and matching pool.
                    unsafe {
                        square.run_with_scratch(
                            &mut output[1..=expected.len()],
                            &mut scratch,
                            executor,
                        );
                    }
                    assert_eq!(&output[1..=expected.len()], expected);
                    scratch.resize(
                        Ssa::mul_two_by_one_scratch_len_for_parallelism(len, len, len, workers),
                        Limb::MAX,
                    );
                    let mut second = vec![Limb::MAX; expected.len()];
                    assert!(Ssa::try_mul_two_by_one_with_executor(
                        &mut output[1..=expected.len()],
                        &mut second,
                        &a,
                        &b,
                        &a,
                        TransformChoice::FORCED,
                        &mut scratch,
                        executor
                    ));
                    assert_eq!(&output[1..=expected.len()], expected);
                    Schoolbook::mul(&mut expected, &b, &a);
                    assert_eq!(second, expected);
                }
            });
        });
    }
}

#[cfg(feature = "rayon")]
#[test]
#[cfg_attr(
    miri,
    ignore = "This 32768-limb fixture crosses the production Rayon reconstruction threshold"
)]
fn parallel_block_reconstruction_resolves_dense_signed_overlaps() {
    use crate::parallel::DefaultExecutor;

    let limbs = 32768;
    let bits = limbs * LIMB_BITS;
    let plan = FftPlan::new_for_square(bits);
    assert!(
        ReconstructionBlocks::new(
            plan.transform_len,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            3,
        )
        .is_some()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("test pool");
    pool.install(|| {
        DefaultExecutor::with_resolved(|executor| {
            // B^n-1 represents -2 in the Fermat ring. Dense all-maximum chunks
            // produce both signs at their convolution magnitude bounds; squaring
            // must reconstruct four after every overlapping carry is resolved.
            let mut input = vec![Limb::MAX; limbs + 1];
            input[limbs] = 0;
            let mut output = vec![Limb::MAX; limbs + 1];
            let mut scratch = vec![Limb::MAX; plan.transform_sqr_scratch(3)];
            for _ in 0..2 {
                // SAFETY: complete guarded input/output and exactly planned scratch
                // in the same three-worker pool used for sizing.
                unsafe {
                    SsaTransform::fft_sqr_mod_slices_with_executor(
                        &mut output,
                        &input,
                        bits,
                        true,
                        Some(&plan),
                        executor,
                        &mut scratch,
                    );
                }
                assert_eq!(output[0], 4);
                assert!(output[1..].iter().all(|limb| *limb == 0));
            }
        });
    });
}
