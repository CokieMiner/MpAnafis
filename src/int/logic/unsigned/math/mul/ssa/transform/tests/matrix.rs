//! Matrix decomposition against weighted full transforms and mixed inverses.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::integer_division,
    reason = "Small exact test geometries bound all allocation and oracle indexing"
)]

use core::{
    mem::size_of,
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::super::{
    ArchKernels, CACHE_BLOCK_BYTES, CoefficientView, LIMB_BITS, Limb, SsaRing, SsaTransform,
    TruncatedTransform,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 24 }))]

    #[test]
    fn weighted_prefixes_match_full_transforms(log in 1_u32..=if cfg!(miri) { 3 } else { 5 }, seed in any::<Limb>(), support in any::<usize>(), phase in any::<usize>()) {
        let len = 1 << log;
        let bits = 4 * LIMB_BITS;
        check_shape(len, bits, phase % (2 * bits / len), 1 + support % len, &SequentialExecutor, 1, seed);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The production cache crossover uses 32768-bit coefficients; small matrix and forced fused-row cases run under Miri"
)]
fn cache_dispatch_boundary_matches_full_transform() {
    let bits = 32768;
    let cl = SsaRing::coeff_limbs(bits).get();
    let budget = CACHE_BLOCK_BYTES / size_of::<Limb>() / cl;
    let transform = TruncatedTransform {
        bits,
        cl: NonZeroUsize::new(cl).expect("test coefficient has a guard"),
        period: NonZeroUsize::new(2 * bits).expect("test ring is positive"),
        kernel: ArchKernels::selected_add_sub_from_limbs_unchecked(),
        max_resident: budget,
    };
    assert_eq!(transform.max_resident, budget);
    let crossing = (budget + 1).next_power_of_two();
    for len in [crossing / 2, crossing, crossing * 2] {
        let root = 2 * bits / len;
        for n in [len / 2 - 1, len / 2, len / 2 + 1, len - 1, len] {
            let mut time = vec![0; len * cl];
            for (i, slot) in time.chunks_exact_mut(cl).take(n).enumerate() {
                slot.fill(Limb::MAX.wrapping_sub(i));
                slot[cl - 1] = i & 1;
            }
            let mut expected = time.clone();
            let mut actual = time.clone();
            actual[n * cl..].fill(Limb::MAX);
            let mut scratch = vec![Limb::MAX; cl * 8];
            let executor = CountingExecutor::default();
            // SAFETY: both transforms have the same primitive root and valid
            // n-slot support. The full reference has physical zero padding;
            // the candidate's dirty tail is implicit zero and scratch is private.
            unsafe {
                SsaTransform::fft_in_place_with_executor(
                    &mut expected,
                    len,
                    root,
                    bits,
                    false,
                    n,
                    &SequentialExecutor,
                    &mut scratch,
                );
                transform.forward(&mut actual, len, root, n, n, &executor, &mut scratch);
                for (a, b) in actual
                    .chunks_exact_mut(cl)
                    .zip(expected.chunks_exact_mut(cl))
                    .take(n)
                {
                    let _ = SsaRing::normalize(a, bits);
                    let _ = SsaRing::normalize(b, bits);
                    assert_eq!(a, b);
                }
                transform.inverse(&mut actual, len, root, n, n, false, &executor, &mut scratch);
                for (a, b) in actual
                    .chunks_exact_mut(cl)
                    .zip(time.chunks_exact_mut(cl))
                    .take(n)
                {
                    let _ = SsaRing::normalize(a, bits);
                    SsaRing::shift_in_place(b, len.trailing_zeros() as usize, bits, &mut scratch);
                    let _ = SsaRing::normalize(b, bits);
                    assert_eq!(a, b);
                }
            }
        }
    }
}

#[derive(Default)]
struct CountingExecutor(AtomicUsize);

impl ParallelExecutor for CountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).expect("positive worker count")
    }
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _previous = self.0.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

#[test]
fn all_weighted_prefixes_and_mixed_inverse_shapes_match_full_transforms() {
    for len in [2, 4, 8, 16, 32] {
        if cfg!(miri) && len > 8 {
            continue;
        }
        let bits = 4 * LIMB_BITS;
        let root = 2 * bits / len;
        for weight in [0, root / 2, root - 1] {
            for z in 1..=len {
                check_shape(len, bits, weight, z, &SequentialExecutor, 1, 17);
            }
        }
    }
}

#[test]
fn worker_batches_preserve_private_scratch() {
    let executor = CountingExecutor::default();
    for slots in [1, 3, 8] {
        for z in [1, 3, 7, 8, 15, 16, 17, 33, 63, 64] {
            if cfg!(miri) {
                if z > 8 {
                    continue;
                }
                check_shape(8, 32 * LIMB_BITS, 3, z, &executor, slots, 17);
            } else {
                check_shape(64, 16 * LIMB_BITS, 3, z, &executor, slots, 17);
            }
        }
        if slots == 1 {
            assert_eq!(executor.0.load(Ordering::Relaxed), 0);
        }
    }
    assert!(executor.0.load(Ordering::Relaxed) > 0);
}

#[test]
#[cfg(feature = "rayon")]
#[cfg_attr(
    miri,
    ignore = "Rayon exercises native thread-pool scheduling; sequential executor tests cover the same strided partitions under Miri"
)]
fn rayon_workers_preserve_disjoint_strided_columns() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("test pool");
    pool.install(|| {
        crate::parallel::DefaultExecutor::with_resolved(|executor| {
            for z in [1, 17, 32, 47, 64] {
                check_shape(64, 16 * LIMB_BITS, 7, z, executor, 8, 17);
            }
        });
    });
}

fn check_shape<E: ParallelExecutor>(
    len: usize,
    bits: usize,
    weight: usize,
    z: usize,
    executor: &E,
    slots: usize,
    seed: Limb,
) {
    let cl = SsaRing::coeff_limbs(bits).get();
    let root = 2 * bits / len;
    let transform = TruncatedTransform {
        bits,
        cl: NonZeroUsize::new(cl).expect("test coefficient has a guard"),
        period: NonZeroUsize::new(2 * bits).expect("test ring is positive"),
        kernel: ArchKernels::selected_add_sub_from_limbs_unchecked(),
        max_resident: (CACHE_BLOCK_BYTES / size_of::<Limb>()) / cl,
    };
    let mut time = vec![0; len * cl];
    let mut state = seed;
    for slot in time.chunks_exact_mut(cl).take(z) {
        for limb in &mut slot[..cl - 1] {
            state = state.wrapping_mul(33).wrapping_add(7);
            *limb = state;
        }
        slot[cl - 1] = state & 1;
    }
    let mut frequency = time.clone();
    let mut scratch = vec![Limb::MAX; cl * slots + 1];
    // SAFETY: the independent full transform has a physically zero tail,
    // complete disjoint scratch, valid root, and semi-normalized input slots.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut frequency,
            len,
            root,
            bits,
            false,
            z,
            &SequentialExecutor,
            &mut scratch[..cl],
        );
        for (j, slot) in frequency.chunks_exact_mut(cl).enumerate() {
            let reversed = j.reverse_bits() >> (usize::BITS - len.trailing_zeros());
            SsaRing::shift_in_place(slot, weight * reversed, bits, &mut scratch[..cl]);
            let _ = SsaRing::normalize(slot, bits);
        }
    }
    for n in 1..=len {
        let mut matrix = padded(&time, cl, z);
        // SAFETY: padded allocates 2*len complete slots and cl is positive.
        let mut view = unsafe { CoefficientView::new(&mut matrix, 2 * len, transform.cl) };
        // SAFETY: width two divides the padded view; column zero owns the z
        // established even slots and disjoint scratch. The weighted root is valid.
        unsafe {
            view.axis(
                NonZeroUsize::new(2).expect("test row width is positive"),
                true,
                0..1,
                &SequentialExecutor,
                &mut scratch[..cl * slots],
                &|column, _, work| {
                    transform.matrix_forward(column, root, weight, z, n, executor, work);
                },
            );
        }
        compare(&mut matrix, &frequency, n, cl, bits);
    }
    // SAFETY: the initialized time matrix has complete semi-normalized slots;
    // multiplying by len uses a reduced exponent and disjoint scratch.
    unsafe {
        for slot in time.chunks_exact_mut(cl) {
            SsaRing::shift_in_place(
                slot,
                len.trailing_zeros() as usize,
                bits,
                &mut scratch[..cl],
            );
            let _ = SsaRing::normalize(slot, bits);
        }
    }
    for n in 0..=z {
        for extra in [false, true] {
            let outputs = n + usize::from(extra);
            if outputs == 0 || outputs > len {
                continue;
            }
            let mut mixed = time.clone();
            mixed[..n * cl].copy_from_slice(&frequency[..n * cl]);
            let mut matrix = padded(&mixed, cl, z);
            // SAFETY: padded allocates 2*len complete slots and cl is positive.
            let mut view = unsafe { CoefficientView::new(&mut matrix, 2 * len, transform.cl) };
            // SAFETY: width two divides the padded view; column zero contains
            // n weighted frequencies then the scaled known tail and implicit
            // zero. Every branch owns complete disjoint staging scratch.
            unsafe {
                view.axis(
                    NonZeroUsize::new(2).expect("test row width is positive"),
                    true,
                    0..1,
                    &SequentialExecutor,
                    &mut scratch[..cl * slots],
                    &|column, _, work| {
                        transform.matrix_inverse(column, root, weight, z, n, extra, executor, work);
                    },
                );
            }
            let mut expected = time.clone();
            if extra {
                expected[n * cl..outputs * cl].copy_from_slice(&frequency[n * cl..outputs * cl]);
            }
            compare(&mut matrix, &expected, outputs, cl, bits);
        }
    }
    assert_eq!(scratch[cl * slots], Limb::MAX);
}

fn padded(source: &[Limb], cl: usize, z: usize) -> Vec<Limb> {
    let mut result = vec![Limb::MAX; source.len() * 2];
    for (index, slot) in source.chunks_exact(cl).take(z).enumerate() {
        result[index * 2 * cl..index * 2 * cl + cl].copy_from_slice(slot);
    }
    result
}

fn compare(matrix: &mut [Limb], expected: &[Limb], count: usize, cl: usize, bits: usize) {
    for (index, pair) in matrix.chunks_exact_mut(2 * cl).enumerate() {
        assert!(pair[cl..].iter().all(|&limb| limb == Limb::MAX));
        if index < count {
            // SAFETY: requested outputs are complete semi-normalized slots.
            unsafe {
                let _ = SsaRing::normalize(&mut pair[..cl], bits);
            }
            assert_eq!(
                &pair[..cl],
                &expected[index * cl..index * cl + cl],
                "slot {index}, outputs {count}"
            );
        }
    }
}
