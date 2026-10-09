//! Wide TFT column scheduling compared with sequential execution.

#![expect(
    unsafe_code,
    clippy::indexing_slicing,
    clippy::integer_division,
    reason = "The bounded matrix provides complete coefficients, valid roots, and disjoint per-worker staging spans"
)]

use core::{mem::size_of, num::NonZeroUsize, sync::atomic::Ordering};

use alloc::vec;

use crate::parallel::SequentialExecutor;

use super::super::{
    super::tests::CountingExecutor, ArchKernels, CACHE_BLOCK_BYTES, Limb, SsaRing, SsaTransform,
    TruncatedTransform,
};

#[test]
#[cfg_attr(
    miri,
    ignore = "The wide column sweep crosses the production parallel grain; smaller matrix partitions run under Miri"
)]
fn wide_column_passes_match_the_sequential_transform() {
    let executor = CountingExecutor::default();
    let mut bits = 512;
    while !SsaTransform::should_parallelize(
        8,
        SsaRing::coeff_limbs(bits).get(),
        SsaRing::coeff_limbs(bits).get(),
        2 * SsaRing::coeff_limbs(bits).get(),
        8,
    ) {
        bits *= 2;
    }
    let len = 64;
    let count = 49;
    let cl = SsaRing::coeff_limbs(bits).get();
    let root = 2 * bits / len;
    let transform = TruncatedTransform {
        bits,
        cl: NonZeroUsize::new(cl).expect("test coefficient has a guard"),
        period: NonZeroUsize::new(2 * bits).expect("test ring is positive"),
        kernel: ArchKernels::selected_add_sub_from_limbs_unchecked(),
        max_resident: (CACHE_BLOCK_BYTES / size_of::<Limb>()) / cl,
    };
    let mut actual = vec![0; len * cl];
    let mut state = 59_usize;
    for slot in actual.chunks_exact_mut(cl) {
        for limb in slot.iter_mut().take(cl - 1) {
            state = state.wrapping_mul(33).wrapping_add(7);
            *limb = state;
        }
    }
    let mut expected = actual.clone();
    let mut single = vec![Limb::MAX; cl];
    let mut parallel = vec![Limb::MAX; 8 * cl];
    // SAFETY: complete initialized matrices, primitive root, implicit zero
    // tail above count, and disjoint per-worker staging arenas.
    unsafe {
        transform.forward(
            &mut expected,
            len,
            root,
            count,
            count,
            &SequentialExecutor,
            &mut single,
        );
        transform.forward(
            &mut actual,
            len,
            root,
            count,
            count,
            &executor,
            &mut parallel,
        );
        actual[count * cl..].fill(Limb::MAX);
        expected[count * cl..].fill(Limb::MAX);
        transform.inverse(
            &mut expected,
            len,
            root,
            count,
            count,
            false,
            &SequentialExecutor,
            &mut single,
        );
        transform.inverse(
            &mut actual,
            len,
            root,
            count,
            count,
            false,
            &executor,
            &mut parallel,
        );
        for slot in expected[..count * cl]
            .chunks_exact_mut(cl)
            .chain(actual[..count * cl].chunks_exact_mut(cl))
        {
            let _ = SsaRing::normalize(slot, bits);
        }
    }
    assert_eq!(&actual[..count * cl], &expected[..count * cl]);
    assert!(executor.joins.load(Ordering::Relaxed) > 0);
}
