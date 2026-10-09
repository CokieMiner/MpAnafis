//! Operand staging with untouched implicit-zero matrix tails.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Small admitted FFT plans bound complete matrix slots, exact source supports, and guarded scratch windows"
)]

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::{
    int::logic::unsigned::math::mul::ssa::tests::CountingExecutor,
    parallel::{ParallelExecutor, SequentialExecutor},
};

use super::super::{FftPlan, LIMB_BITS, Limb, SsaCoefficients, SsaRing, SsaTransform};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 64 }))]

    #[test]
    fn dirty_staging_matches_physically_padded_dense_transforms(
        bits in prop::sample::select(if cfg!(miri) { vec![512_usize, 3072] } else { vec![512_usize, 3072, 8192, 32768] }),
        support in any::<usize>(),
        seed in any::<Limb>(),
        dirty in any::<Limb>(),
    ) {
        let plan = FftPlan::new(bits);
        let active = 1 + support % plan.transform_len;
        check_staging(&plan, active, seed, dirty, &SequentialExecutor);
        check_staging(&plan, active, seed, dirty, &CountingExecutor::default());
    }
}

#[test]
fn first_stage_support_boundaries_preserve_dirty_tails_and_canaries() {
    for bits in [512, 3072, 8192] {
        if cfg!(miri) && bits > 512 {
            continue;
        }
        let plan = FftPlan::new(bits);
        let half = plan.transform_len >> 1;
        for active in [
            1,
            half.saturating_sub(1).max(1),
            half,
            half + 1,
            plan.transform_len - 1,
            plan.transform_len,
        ] {
            for dirty in [0, Limb::MAX] {
                check_staging(&plan, active, Limb::MAX, dirty, &SequentialExecutor);
                check_staging(
                    &plan,
                    active,
                    Limb::MAX,
                    dirty,
                    &CountingExecutor::default(),
                );
            }
        }
    }
}

fn check_staging<E: ParallelExecutor>(
    plan: &FftPlan,
    active: usize,
    seed: Limb,
    dirty: Limb,
    executor: &E,
) {
    // A one-bit final chunk can occupy only part of its source limb. Its
    // allocated bit capacity may cover additional chunks outside the support.
    let width = (active - 1) * plan.chunk_bits.get() + 1;
    let mut source = vec![seed; width.div_ceil(LIMB_BITS)];
    let used = 1 + (width - 1) % LIMB_BITS;
    let top = source.last_mut().expect("nonempty source");
    *top &= Limb::MAX >> (LIMB_BITS - used);
    *top |= 1 << (used - 1);
    let cl = plan.inner_cl.get();
    let span = plan.mat_limbs.get();
    let work_len = cl * plan.parallel_slots(executor.parallelism().get());
    let mut scratch = vec![Limb::MAX; work_len + 2];
    let mut expected = dense_forward(plan, &source, &mut scratch[1..=work_len]);
    let mut actual = vec![dirty; span + 2];
    actual[0] = Limb::MAX;
    actual[span + 1] = Limb::MAX;
    if active <= plan.transform_len >> 1 {
        // SAFETY: the source's exact width proves active<=K/2. Both complete
        // matrix halves and the private staging arena use the admitted plan.
        let fused = unsafe {
            SsaCoefficients::split_twisted_and_stage1_dif_with_executor(
                &source,
                &mut actual[1..=span],
                plan.transform_len,
                active,
                plan.chunk_bits,
                plan.inner_cl,
                plan.periods,
                plan.twist_step_half,
                plan.twist_step_half,
                executor,
                &mut scratch[1..=work_len],
            )
        };
        assert!(fused, "the plan has at least two coefficients");
        let half_span = span >> 1;
        for tail in [
            &actual[1 + active * cl..=half_span],
            &actual[1 + half_span + active * cl..=span],
        ] {
            assert!(
                tail.iter().all(|&limb| limb == dirty),
                "implicit zero tails stay untouched"
            );
        }
        // SAFETY: the fused split established exactly active initialized slots
        // in each half. Sparse DIF writes all frequencies without reading tails.
        unsafe {
            SsaTransform::fft_in_place_from_stage2_with_executor(
                &mut actual[1..=span],
                plan.transform_len,
                plan.twist_step_half,
                plan.inner_bits,
                active,
                executor,
                &mut scratch[1..=work_len],
            );
        }
    } else {
        // SAFETY: source has the declared support and every matrix/scratch slot
        // exists. The sparse forward accepts arbitrary initialized tail data.
        unsafe {
            SsaTransform::stage_and_run_forward_fft(
                &source,
                &mut actual[1..=span],
                &mut scratch[1..=work_len],
                plan,
                false,
                active,
                executor,
            );
        }
    }
    for (a, b) in actual[1..=span]
        .chunks_exact_mut(cl)
        .zip(expected.chunks_exact_mut(cl))
    {
        // SAFETY: both forward transforms established complete semi-normalized
        // coefficients. Canonicalization compares their ring values.
        unsafe {
            let _ = SsaRing::normalize(a, plan.inner_bits);
            let _ = SsaRing::normalize(b, plan.inner_bits);
        }
        assert_eq!(
            a,
            b,
            "staging support={active}, workers={}",
            executor.parallelism()
        );
    }
    for buffer in [&actual, &scratch] {
        assert_eq!(buffer.first(), Some(&Limb::MAX), "leading canary");
        assert_eq!(buffer.last(), Some(&Limb::MAX), "trailing canary");
    }
}

fn dense_forward(plan: &FftPlan, source: &[Limb], scratch: &mut [Limb]) -> Vec<Limb> {
    let mut matrix = vec![0; plan.mat_limbs.get()];
    // SAFETY: admitted geometry supplies the exact complete matrix and a
    // disjoint two-slot scratch window. The reference materializes all zeros
    // and evaluates every frequency using the dense forward support.
    unsafe {
        SsaCoefficients::split_twisted(
            source,
            &mut matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_cl,
            plan.periods,
            plan.twist_step_half,
            scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut matrix,
            plan.transform_len,
            plan.twist_step_half,
            plan.inner_bits,
            false,
            plan.transform_len,
            &SequentialExecutor,
            scratch,
        );
    }
    matrix
}
