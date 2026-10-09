//! Fork-grain policy and executor-width behaviour of the SSA transform.

#![expect(
    unsafe_code,
    reason = "Executor-policy fixtures establish complete disjoint matrices and explicitly bounded worker arenas"
)]

use super::*;

#[test]
fn parallel_grain_is_based_on_work_per_worker() {
    let executor = CountingExecutor::default();
    let workers = executor.parallelism().get();
    let threshold = SSA_PARALLEL_MIN_LIMB_WORK;

    // The fork grain is the smaller child of a binary split (half the range),
    // not the range divided by the pool width: two items each carrying the
    // threshold still fork even across eight workers.
    assert!(transform::SsaTransform::has_parallel_work(
        2, threshold, workers
    ));
    // A smaller child carrying one limb below the threshold stays sequential.
    assert!(!transform::SsaTransform::has_parallel_work(
        2,
        threshold.saturating_sub(1),
        workers
    ));
}

#[test]
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn direct_fermat_requires_the_measured_wide_pool() {
    assert_eq!(Ssa::direct_fermat_threshold(1), None);
    assert_eq!(Ssa::direct_fermat_threshold(2), None);
    assert_eq!(Ssa::direct_fermat_threshold(7), None);
    assert_eq!(Ssa::direct_fermat_threshold(8), Some(1_048_576));
    assert_eq!(Ssa::direct_fermat_threshold(16), Some(1_048_576));
}

#[test]
fn internal_parallel_policy_matches_sequential_fft_product() {
    let modulus_bits = 512;
    let ml = SsaRing::mod_limbs(modulus_bits);
    let cl = SsaRing::coeff_limbs(modulus_bits).get();
    let plan = FftPlan::new(modulus_bits);
    let mut left = vec![0; ml];
    let mut right = vec![0; ml];
    *left.first_mut().expect("nonempty ring") = 7;
    *right.first_mut().expect("nonempty ring") = 11;

    let mut parallel_result = vec![0; cl];
    let parallel = CountingExecutor::default();
    let mut parallel_scratch = vec![0; plan.transform_mul_scratch(parallel.parallelism().get())];
    // SAFETY: both operands omit the guard and carry their exact nonzero widths;
    // the destination and scratch satisfy the plan's validated transform layout.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut parallel_result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &parallel,
            &mut parallel_scratch,
        );
    }

    let mut sequential_result = vec![0; cl];
    let mut sequential_scratch = vec![0; plan.transform_mul_scratch(1)];
    // SAFETY: the same validated layout is used for the independent sequential run.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut sequential_result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &crate::parallel::SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_result, sequential_result);
    assert!(parallel.joins.load(Ordering::Relaxed) > 0);
}

#[test]
fn one_slot_executor_avoids_all_fft_joins() {
    let modulus_bits = 512;
    let ml = SsaRing::mod_limbs(modulus_bits);
    let cl = SsaRing::coeff_limbs(modulus_bits).get();
    let plan = FftPlan::new(modulus_bits);
    let mut left = vec![0; ml];
    let mut right = vec![0; ml];
    *left.first_mut().expect("nonempty ring") = 7;
    *right.first_mut().expect("nonempty ring") = 11;

    let executor = OneSlotCountingExecutor::default();
    let mut result = vec![0; cl];
    let mut scratch = vec![0; plan.transform_mul_scratch(1)];
    // SAFETY: both operands carry exact nonzero widths, and the result and
    // scratch spans come directly from this forced plan.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &executor,
            &mut scratch,
        );
    }

    assert_eq!(
        result.first().copied(),
        Some(77),
        "the one-slot path must compute the same product"
    );
    assert_eq!(
        executor.joins.load(Ordering::Relaxed),
        0,
        "parallelism one must not enter fork-shaped control flow"
    );
}

#[test]
fn internal_parallel_policy_splits_large_fft_ranges_without_changing_results() {
    // Keep the matrix small while making each transform coefficient expensive
    // enough to cross the generated work-per-worker parallelism threshold.
    let mod_bits = 16_384;
    let transform_len = 64;
    let cl = SsaRing::coeff_limbs(mod_bits).get();
    let root_shift = mod_bits.wrapping_mul(2).div_euclid(transform_len);
    let mut parallel_matrix = vec![0; transform_len.wrapping_mul(cl)];
    for (index, coefficient) in parallel_matrix.chunks_exact_mut(cl).enumerate() {
        *coefficient.first_mut().expect("nonempty coefficient") = index + 1;
    }
    let mut sequential_matrix = parallel_matrix.clone();
    let parallel = CountingExecutor::default();
    let mut parallel_scratch = vec![0; cl.wrapping_mul(8)];
    let mut sequential_scratch = vec![0; cl];

    // SAFETY: the matrix has transform_len complete canonical coefficients, the
    // root divides the Fermat period, and parallel scratch has two disjoint slots.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut parallel_matrix,
            transform_len,
            root_shift,
            mod_bits,
            false,
            transform_len,
            &parallel,
            &mut parallel_scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut sequential_matrix,
            transform_len,
            root_shift,
            mod_bits,
            false,
            transform_len,
            &SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_matrix, sequential_matrix);
    // The inverse DIT recursion uses the same range partitioning policy. Run it
    // on both outputs to verify that its recombination order also agrees.
    // SAFETY: the forward outputs remain complete matrices and the same root is
    // valid for the inverse transform.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut parallel_matrix,
            transform_len,
            root_shift,
            mod_bits,
            true,
            transform_len,
            &parallel,
            &mut parallel_scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut sequential_matrix,
            transform_len,
            root_shift,
            mod_bits,
            true,
            transform_len,
            &SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_matrix, sequential_matrix);
    assert!(parallel.joins.load(Ordering::Relaxed) >= 2);
}
