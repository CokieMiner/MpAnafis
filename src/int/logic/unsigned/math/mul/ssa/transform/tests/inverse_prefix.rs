//! Retained inverse prefixes compared with complete radix-four stages.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded test geometries establish complete disjoint radix-four quarters and exact prefix spans"
)]

use core::num::NonZeroUsize;

use alloc::vec;

use proptest::prelude::*;

use crate::parallel::SequentialExecutor;

use super::super::{ArchKernels, LIMB_BITS, Limb, SsaRing, SsaTransform, TransformContext};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 24 }))]

    #[test]
    fn retained_prefixes_match_complete_inverse_stages(
        log in 0_u32..=if cfg!(miri) { 1 } else { 3 },
        data_limbs in prop::sample::select(vec![1_usize, 2, 5, 16]),
        seed in any::<Limb>(),
    ) {
        check_prefixes(1 << log, data_limbs, seed);
    }
}

#[test]
fn carry_and_zero_boundaries_preserve_every_prefix() {
    for data_limbs in [1, 2, 5, 16] {
        for seed in [0, 1, Limb::MAX] {
            check_prefixes(if cfg!(miri) { 1 } else { 8 }, data_limbs, seed);
        }
    }
}

fn check_prefixes(quarter: usize, data_limbs: usize, seed: Limb) {
    let bits = data_limbs * LIMB_BITS;
    let cl = data_limbs + 1;
    let len = 4 * quarter;
    let root = bits
        .checked_mul(2)
        .expect("test period fits")
        .div_euclid(len);
    let mut input = vec![seed; len * cl];
    for (index, slot) in input.chunks_exact_mut(cl).enumerate() {
        slot[cl - 1] = index & 1;
    }
    let ctx = TransformContext::new(
        bits,
        ArchKernels::selected_add_sub_from_limbs_unchecked(),
        &SequentialExecutor,
    );
    let mut scratch = vec![Limb::MAX; cl];
    let mut expected = input.clone();
    run_stage(&mut expected, quarter, root, len, &ctx, &mut scratch);
    for slot in expected.chunks_exact_mut(cl) {
        // SAFETY: the complete inverse stage retains initialized guarded coefficients.
        unsafe {
            let _ = SsaRing::normalize(slot, bits);
        }
    }
    for needed in 0..=len {
        let mut actual = input.clone();
        run_stage(&mut actual, quarter, root, needed, &ctx, &mut scratch);
        for slot in actual.chunks_exact_mut(cl).take(needed) {
            // SAFETY: each retained prefix slot is complete and semi-normalized.
            unsafe {
                let _ = SsaRing::normalize(slot, bits);
            }
        }
        assert_eq!(&actual[..needed * cl], &expected[..needed * cl]);
    }
}

fn run_stage(
    matrix: &mut [Limb],
    quarter: usize,
    root: usize,
    needed: usize,
    ctx: &TransformContext<'_, SequentialExecutor>,
    scratch: &mut [Limb],
) {
    let span = quarter * ctx.cl.get();
    let (q0, after_first) = matrix.split_at_mut(span);
    let (q1, after_second) = after_first.split_at_mut(span);
    let (q2, q3) = after_second.split_at_mut(span);
    // SAFETY: quarter>0, root*4*quarter=2*bits, needed<=4*quarter;
    // all four initialized quarters and the complete scratch coefficient are disjoint.
    unsafe {
        SsaTransform::dit_radix4_stage(
            [q0, q1, q2, q3],
            NonZeroUsize::new(quarter).expect("positive quarter"),
            root,
            scratch,
            needed,
            ctx,
        );
    }
}
