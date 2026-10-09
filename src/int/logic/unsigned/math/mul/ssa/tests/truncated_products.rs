//! Truncated multiplication, squaring, and paired-product entry points.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::integer_division,
    reason = "Bounded test rings provide complete disjoint operands, outputs, and exact execution arenas"
)]

use alloc::{vec, vec::Vec};

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{CountingExecutor, FftPlan, LIMB_BITS, Limb, SsaPointwise, SsaRing, SsaTransform};

#[test]
fn sparse_irregular_and_crossover_supports_match_full_products() {
    for bits in [512, 3072, 8192, 12288] {
        if cfg!(miri) && bits > 512 {
            continue;
        }
        let plan = FftPlan::new_for_pair(bits);
        let chunk = plan.chunk_bits.get();
        for widths in [
            [1, 1, 1],
            [0, 1, chunk],
            [chunk - 1, chunk, chunk + 1],
            [bits / 4 - 1, bits / 4, bits / 4 + 1],
            [bits / 2 - 1, bits / 4 + 1, bits / 2],
            [bits / 2 + 1, bits / 4, bits / 2],
            [bits - 1, bits / 8, bits / 4],
        ] {
            check_products(bits, widths, &SequentialExecutor);
            check_products(bits, widths, &CountingExecutor::default());
        }
    }
}

#[test]
fn inverse_scaling_covers_each_power_of_two_prefix_boundary() {
    for &bits in if cfg!(miri) {
        &[512_usize][..]
    } else {
        &[3072_usize, 8192, 12288][..]
    } {
        let plan = FftPlan::new_for_pair(bits);
        let mut boundary = 2;
        while boundary < plan.transform_len {
            for count in [boundary - 1, boundary, boundary + 1] {
                // A one-coefficient shared factor preserves each input support.
                // The paired outputs exercise different inverse scales together.
                let widths = [(count - 1) * plan.chunk_bits.get() + 1, 1, 1];
                check_products(bits, widths, &SequentialExecutor);
                check_products(bits, widths, &CountingExecutor::default());
            }
            boundary *= 2;
        }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(
    miri,
    ignore = "Native Rayon scheduling uses three workers; deterministic executors cover smaller truncated products under Miri"
)]
#[test]
fn rayon_products_match_full_products_in_a_three_worker_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("test thread pool");
    pool.install(|| {
        crate::parallel::DefaultExecutor::with_resolved(|executor| {
            check_products(32768, [8193, 12287, 4097], executor);
        });
    });
}

fn check_products<E: ParallelExecutor>(bits: usize, widths: [usize; 3], executor: &E) {
    let [a_bits, b_bits, x_bits] = widths;
    let a = narrow_operand(a_bits, bits, 31);
    let b = narrow_operand(b_bits, bits, 37);
    let x = narrow_operand(x_bits, bits, 41);
    let cl = SsaRing::coeff_limbs(bits).get();
    let plan = FftPlan::new_for_pair(bits);
    let mut oracle_scratch = vec![Limb::MAX; SsaPointwise::fermat_basecase_scratch_len(bits)];
    let mut expected_a = vec![0; cl];
    let mut expected_b = vec![0; cl];
    let mut expected_square = vec![0; cl];
    // SAFETY: all inputs and outputs are canonical guarded coefficients, and
    // the independent full-width product arena is exactly basecase-sized.
    unsafe {
        SsaPointwise::fermat_basecase_mul_into(&mut expected_a, &a, &x, bits, &mut oracle_scratch);
        SsaPointwise::fermat_basecase_mul_into(&mut expected_b, &b, &x, bits, &mut oracle_scratch);
        SsaPointwise::fermat_basecase_sqr_into(&mut expected_square, &a, bits, &mut oracle_scratch);
    }
    let workers = executor.parallelism().get();
    let mut pair_scratch = vec![Limb::MAX; plan.transform_mul_two_by_one_scratch(workers)];
    let mut mul_scratch = vec![Limb::MAX; plan.transform_mul_scratch(workers)];
    let mut sqr_scratch = vec![Limb::MAX; plan.transform_sqr_scratch(workers)];
    let mut actual_a = vec![Limb::MAX; cl + 2];
    let mut actual_b = vec![Limb::MAX; cl + 2];
    for _ in 0..2 {
        // SAFETY: each destination window is a complete guarded coefficient;
        // all planned arenas are exact and supports match the canonical inputs.
        unsafe {
            SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                &mut actual_a[1..=cl],
                &mut actual_b[1..=cl],
                &a,
                &b,
                &x,
                bits,
                Some((a_bits, b_bits, x_bits)),
                true,
                Some(&plan),
                executor,
                &mut pair_scratch,
            );
        }
        assert_eq!(
            &actual_a[1..=cl],
            expected_a,
            "pair A bits={bits}, widths={widths:?}"
        );
        assert_eq!(
            &actual_b[1..=cl],
            expected_b,
            "pair B bits={bits}, widths={widths:?}"
        );
        // SAFETY: the same guarded input/output and exact scratch contracts
        // apply. The square's short input gives its actual significant width.
        unsafe {
            SsaTransform::fft_mul_mod_slices_with_executor(
                &mut actual_a[1..=cl],
                &a,
                &x,
                bits,
                Some((a_bits, x_bits)),
                true,
                Some(&plan),
                executor,
                &mut mul_scratch,
            );
            SsaTransform::fft_sqr_mod_slices_with_executor(
                &mut actual_b[1..=cl],
                &a[..a_bits.div_ceil(LIMB_BITS)],
                bits,
                true,
                Some(&plan),
                executor,
                &mut sqr_scratch,
            );
        }
        assert_eq!(
            &actual_a[1..=cl],
            expected_a,
            "mul bits={bits}, widths={widths:?}"
        );
        assert_eq!(
            &actual_b[1..=cl],
            expected_square,
            "square bits={bits}, widths={widths:?}"
        );
        for output in [&actual_a, &actual_b] {
            assert_eq!((output[0], output[cl + 1]), (Limb::MAX, Limb::MAX));
        }
    }
}

fn narrow_operand(width: usize, bits: usize, seed: Limb) -> Vec<Limb> {
    let mut operand = vec![0; SsaRing::coeff_limbs(bits).get()];
    let mut state = seed;
    for limb in operand.iter_mut().take(width.div_ceil(LIMB_BITS)) {
        state = state.wrapping_mul(33).wrapping_add(7);
        *limb = state;
    }
    if width > 0 {
        let top = (width - 1) / LIMB_BITS;
        let used = 1 + (width - 1) % LIMB_BITS;
        operand[top] &= Limb::MAX >> (LIMB_BITS - used);
        operand[top] |= 1 << (used - 1);
    }
    operand
}
