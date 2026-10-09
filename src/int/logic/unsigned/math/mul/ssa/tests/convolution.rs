//! Fused convolution subtrees and ring drivers against staged and scalar oracles.

#![expect(
    unsafe_code,
    reason = "Complete canonical matrices, primitive roots, and operation-sized private arenas satisfy each convolution and transform contract"
)]

use crate::int::logic::unsigned::math::ArchKernels;

use super::{super::transform::TransformContext, *};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 16 }))]

    #[test]
    fn fused_subtrees_match_staged_products_squares_and_shared_spectra(seed in any::<Limb>()) {
        for len in [4_usize, 16] {
            let bits = 512;
            let cl = SsaRing::coeff_limbs(bits).get();
            let root = bits.checked_mul(2).expect("period fits").div_euclid(len);
            let product = PointwiseMulPlan::from(bits);
            let square = PointwiseSquarePlan::from(bits);
            let ctx = TransformContext::new(bits, ArchKernels::selected_add_sub_from_limbs_unchecked(), &SequentialExecutor);
            let inputs = [matrix(len, cl, seed ^ 0x1111), matrix(len, cl, seed ^ 0x2222), matrix(len, cl, seed ^ 0x3333)];
            let mut staged = inputs.clone();
            let mut recursive = inputs.first().expect("three matrices").clone();
            let work_len = len.checked_mul(cl).and_then(|n| n.checked_mul(6)).expect("arena fits").max(product.scratch_len.get()).max(square.scratch_len.get());
            let mut work = vec![Limb::MAX; work_len];
            // SAFETY: matrices contain len complete canonical coefficients,
            // root*len=2*bits, and work covers either retained leaf strategy.
            unsafe {
                SsaTransform::fft_recursive_dif_with_executor(&mut recursive, len, root, &mut work, len, &ctx);
                for input in &mut staged {
                    SsaTransform::fft_in_place_with_executor(input, len, root, bits, false, len, &SequentialExecutor, &mut work);
                }
            }
            prop_assert_eq!(&recursive, staged.first().expect("first matrix"));
            let [mut expected_a, mut expected_b, mut spectrum] = staged;
            let mut expected_square = recursive;
            let [mut fused_a, mut fused_b, mut fused_x] = inputs.clone();
            let mut fused_product = inputs.first().expect("first matrix").clone();
            let mut product_right = inputs.get(2).expect("shared matrix").clone();
            let mut fused_square = inputs.first().expect("first matrix").clone();
            // SAFETY: the staged matrices establish every requested frequency;
            // each fused call receives complete independent canonical matrices
            // and a private arena reused only after the preceding call returns.
            unsafe {
                SsaPointwise::pointwise_multiply_with_executor(&mut expected_a, &mut spectrum, len, NonZeroUsize::MIN, &product, &SequentialExecutor, &mut work);
                SsaPointwise::pointwise_multiply_with_executor(&mut expected_b, &mut spectrum, len, NonZeroUsize::MIN, &product, &SequentialExecutor, &mut work);
                SsaPointwise::pointwise_square_with_executor(&mut expected_square, len, NonZeroUsize::MIN, &square, &SequentialExecutor, &mut work);
                for output in [&mut expected_a, &mut expected_b] {
                    SsaTransform::fft_in_place_with_executor(output, len, root, bits, true, len, &SequentialExecutor, &mut work);
                }
                SsaTransform::fft_recursive_dit_with_executor(&mut expected_square, len, root, &mut work, len, &ctx);
                SsaTransform::convolve_subtrees(&mut fused_product, &mut product_right, len, root, &product, &SequentialExecutor, &mut work);
                SsaTransform::convolve_square_subtrees(&mut fused_square, len, root, &square, &SequentialExecutor, &mut work);
                SsaTransform::convolve_pair_subtrees([&mut fused_a, &mut fused_b, &mut fused_x], len, root, &product, &SequentialExecutor, &mut work);
            }
            prop_assert_eq!(&fused_product, &expected_a);
            prop_assert_eq!(&fused_square, &expected_square);
            prop_assert_eq!(fused_a, expected_a);
            prop_assert_eq!(fused_b, expected_b);
            prop_assert_eq!(fused_x, spectrum);
        }
    }

    #[test]
    fn complete_ring_drivers_match_scalar_products_with_exact_dirty_arenas(seed in any::<Limb>()) {
        check_drivers(seed, &SequentialExecutor);
        check_drivers(seed, &CountingExecutor::default());
    }
}

fn check_drivers<E: ParallelExecutor>(seed: Limb, executor: &E) {
    let bits = 512;
    let cl = SsaRing::coeff_limbs(bits).get();
    let left = matrix(1, cl, seed ^ 0xDDDD);
    let right = matrix(1, cl, seed ^ 0xEEEE);
    let shared = matrix(1, cl, seed ^ 0xCCCC);
    let workers = executor.parallelism().get();
    let product_plan = FftPlan::new(bits);
    let square_plan = FftPlan::new_for_square(bits);
    let pair_plan = FftPlan::new_for_pair(bits);
    let mut product_work = vec![Limb::MAX; product_plan.transform_mul_scratch(workers)];
    let mut square_work = vec![Limb::MAX; square_plan.transform_sqr_scratch(workers)];
    let mut pair_work = vec![Limb::MAX; pair_plan.transform_mul_two_by_one_scratch(workers)];
    let mut oracle_work = vec![Limb::MAX; SsaPointwise::fermat_basecase_scratch_len(bits)];
    let mut actual = vec![Limb::MAX; cl];
    let mut paired = vec![Limb::MAX; cl];
    let mut expected = vec![0; cl];
    let mut expected_pair = vec![0; cl];
    for _ in 0..2 {
        product_work.fill(Limb::MAX);
        square_work.fill(Limb::MAX);
        pair_work.fill(Limb::MAX);
        // SAFETY: every input is a complete canonical coefficient, outputs
        // are disjoint complete slots, and each exact arena matches its plan
        // and executor. Forcing transforms admits this small ring geometry.
        unsafe {
            SsaTransform::fft_mul_mod_slices_with_executor(
                &mut actual,
                &left,
                &right,
                bits,
                None,
                true,
                Some(&product_plan),
                executor,
                &mut product_work,
            );
            SsaPointwise::fermat_basecase_mul_into(
                &mut expected,
                &left,
                &right,
                bits,
                &mut oracle_work,
            );
            assert_eq!(actual, expected);
            SsaTransform::fft_sqr_mod_slices_with_executor(
                &mut actual,
                &left,
                bits,
                true,
                Some(&square_plan),
                executor,
                &mut square_work,
            );
            SsaPointwise::fermat_basecase_sqr_into(&mut expected, &left, bits, &mut oracle_work);
            assert_eq!(actual, expected);
            SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                &mut actual,
                &mut paired,
                &left,
                &right,
                &shared,
                bits,
                None,
                true,
                Some(&pair_plan),
                executor,
                &mut pair_work,
            );
            SsaPointwise::fermat_basecase_mul_into(
                &mut expected,
                &left,
                &shared,
                bits,
                &mut oracle_work,
            );
            SsaPointwise::fermat_basecase_mul_into(
                &mut expected_pair,
                &right,
                &shared,
                bits,
                &mut oracle_work,
            );
        }
        assert_eq!(actual, expected);
        assert_eq!(paired, expected_pair);
    }
}

fn matrix(len: usize, cl: usize, seed: Limb) -> Vec<Limb> {
    let mut values = vec![0; len.checked_mul(cl).expect("matrix fits")];
    let mut state = seed;
    for coefficient in values.chunks_exact_mut(cl) {
        for word in coefficient
            .iter_mut()
            .take(cl.checked_sub(1).expect("guarded width"))
        {
            state = state.wrapping_mul(33).wrapping_add(7);
            *word = state;
        }
    }
    *values.first_mut().expect("nonempty matrix") |= 1;
    values
}
