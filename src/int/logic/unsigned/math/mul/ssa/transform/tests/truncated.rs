//! TFT frequency-prefix and mixed-coordinate ITFT properties.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::integer_division,
    reason = "TFT properties construct exact matrix spans with small test constants"
)]

use core::{mem::size_of, num::NonZeroUsize};

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::parallel::SequentialExecutor;

use super::super::{
    ArchKernels, CACHE_BLOCK_BYTES, LIMB_BITS, Limb, PointwiseMulPlan, SsaPointwise, SsaRing,
    SsaTransform, TruncatedTransform,
};

fn input_matrix(len: usize, cl: usize, seed: Limb) -> Vec<Limb> {
    let mut matrix = vec![0; len * cl];
    let mut state = seed;
    for slot in matrix.chunks_exact_mut(cl) {
        for limb in slot.iter_mut().take(cl - 1) {
            state = state.wrapping_mul(33).wrapping_add(7);
            *limb = state;
        }
        *slot.last_mut().expect("complete coefficient") = state & 1;
    }
    matrix
}

fn canonicalize(matrix: &mut [Limb], bits: usize) {
    for slot in matrix.chunks_exact_mut(SsaRing::coeff_limbs(bits).get()) {
        // SAFETY: each slot is a complete semi-normalized ring coefficient.
        unsafe {
            let _ = SsaRing::normalize(slot, bits);
        }
    }
}

fn test_transform(bits: usize) -> TruncatedTransform {
    let cl = SsaRing::coeff_limbs(bits);
    TruncatedTransform {
        bits,
        cl,
        period: NonZeroUsize::new(2 * bits).expect("test ring is positive"),
        kernel: ArchKernels::selected_add_sub_from_limbs_unchecked(),
        max_resident: (CACHE_BLOCK_BYTES / size_of::<Limb>()) / cl,
    }
}

#[test]
fn peeled_twiddle_columns_preserve_dirty_support_boundaries() {
    let bits = 2 * LIMB_BITS;
    let cl = SsaRing::coeff_limbs(bits).get();
    let len = 8;
    let width = len * cl;
    let root = 2 * bits / len;
    let transform = test_transform(bits);
    for seed in [0, Limb::MAX] {
        for inputs in 0..=len {
            let source = input_matrix(len, cl, seed);
            let mut expected = source.clone();
            expected[inputs * cl..].fill(0);
            let mut scratch = vec![Limb::MAX; cl + 2];
            // SAFETY: complete initialized matrix with a physical zero tail;
            // the dense reference has a principal root and private exact scratch.
            unsafe {
                SsaTransform::fft_in_place_with_executor(
                    &mut expected,
                    len,
                    root,
                    bits,
                    false,
                    len,
                    &SequentialExecutor,
                    &mut scratch[1..=cl],
                );
            }
            canonicalize(&mut expected, bits);
            for outputs in [1, 3, 5, len] {
                let mut actual = vec![Limb::MAX; width + 2];
                actual[1..=inputs * cl].copy_from_slice(&source[..inputs * cl]);
                // SAFETY: only the stated prefix is live; every destination
                // slot exists, and scratch and both canaries are disjoint.
                unsafe {
                    transform.forward(
                        &mut actual[1..=width],
                        len,
                        root,
                        inputs,
                        outputs,
                        &SequentialExecutor,
                        &mut scratch[1..=cl],
                    );
                }
                canonicalize(&mut actual[1..=outputs * cl], bits);
                assert_eq!(
                    &actual[1..=outputs * cl],
                    &expected[..outputs * cl],
                    "support={inputs}, outputs={outputs}"
                );
                if inputs <= outputs {
                    let mut scaled = source[..outputs * cl].to_vec();
                    scaled[inputs * cl..].fill(0);
                    // SAFETY: all requested frequencies are initialized and
                    // the polynomial support fits their count. The ITFT's
                    // absent tail is implicit zero. Scaling the complete
                    // reference coefficients by len=8 uses exponent three.
                    unsafe {
                        transform.inverse(
                            &mut actual[1..=width],
                            len,
                            root,
                            outputs,
                            outputs,
                            false,
                            &SequentialExecutor,
                            &mut scratch[1..=cl],
                        );
                        for slot in scaled.chunks_exact_mut(cl) {
                            SsaRing::shift_in_place(slot, 3, bits, &mut scratch[1..=cl]);
                        }
                    }
                    canonicalize(&mut actual[1..=outputs * cl], bits);
                    canonicalize(&mut scaled, bits);
                    assert_eq!(
                        &actual[1..=outputs * cl],
                        &scaled,
                        "inverse support={inputs}, outputs={outputs}"
                    );
                }
                for buffer in [&actual, &scratch] {
                    assert_eq!(buffer.first(), Some(&Limb::MAX), "leading canary");
                    assert_eq!(buffer.last(), Some(&Limb::MAX), "trailing canary");
                }
            }
        }
    }
}

#[test]
fn tft_all_input_and_frequency_prefixes_match_full_dif() {
    for bits in [64, 192, 512] {
        let cl = SsaRing::coeff_limbs(bits).get();
        let transform = test_transform(bits);
        for len in [1, 2, 4, 8, 16, 32] {
            if cfg!(miri) && len > 8 {
                continue;
            }
            let root = 2 * bits / len;
            for inputs in 0..=len {
                let source = input_matrix(len, cl, 13);
                let mut expected = source.clone();
                expected[inputs * cl..].fill(0);
                let mut scratch = vec![Limb::MAX; cl];
                // SAFETY: complete matrix with physical zero tail, primitive
                // root, and exactly one disjoint staging coefficient.
                unsafe {
                    SsaTransform::fft_in_place_with_executor(
                        &mut expected,
                        len,
                        root,
                        bits,
                        false,
                        inputs,
                        &SequentialExecutor,
                        &mut scratch,
                    );
                }
                canonicalize(&mut expected, bits);
                for outputs in 0..=len {
                    let mut actual = source.clone();
                    actual[inputs * cl..].fill(Limb::MAX);
                    // SAFETY: only the active prefix is input; arbitrary tail
                    // storage is deliberately not a valid ring representation.
                    unsafe {
                        transform.forward(
                            &mut actual,
                            len,
                            root,
                            inputs,
                            outputs,
                            &SequentialExecutor,
                            &mut scratch,
                        );
                    }
                    canonicalize(&mut actual[..outputs * cl], bits);
                    assert_eq!(
                        &actual[..outputs * cl],
                        &expected[..outputs * cl],
                        "bits={bits}, len={len}, z={inputs}, n={outputs}"
                    );
                }
            }
        }
    }
}

#[test]
fn itft_recovers_every_prefix_without_reading_missing_frequencies() {
    for bits in [64, 192, 512] {
        let cl = SsaRing::coeff_limbs(bits).get();
        let transform = test_transform(bits);
        for len in [1, 2, 4, 8, 16, 32, 64] {
            if cfg!(miri) && len > 8 {
                continue;
            }
            let root = 2 * bits / len;
            for count in 1..=len {
                let mut actual = input_matrix(len, cl, 17);
                let mut expected = actual[..count * cl].to_vec();
                let mut scratch = vec![Limb::MAX; cl];
                // SAFETY: complete initialized slots, primitive root and exactly
                // one coefficient arena. The polynomial support is count.
                unsafe {
                    transform.forward(
                        &mut actual,
                        len,
                        root,
                        count,
                        count,
                        &SequentialExecutor,
                        &mut scratch,
                    );
                    actual[count * cl..].fill(Limb::MAX);
                    transform.inverse(
                        &mut actual,
                        len,
                        root,
                        count,
                        count,
                        false,
                        &SequentialExecutor,
                        &mut scratch,
                    );
                    for slot in expected.chunks_exact_mut(cl) {
                        SsaRing::shift_in_place(
                            slot,
                            len.trailing_zeros() as usize,
                            bits,
                            &mut scratch,
                        );
                    }
                }
                canonicalize(&mut actual[..count * cl], bits);
                canonicalize(&mut expected, bits);
                assert_eq!(
                    &actual[..count * cl],
                    expected,
                    "bits={bits}, len={len}, n={count}"
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 128 }))]
    #[test]
    fn tft_product_matches_full_convolution(log in 1_u32..=if cfg!(miri) { 3 } else { 7 }, seed in any::<Limb>(), left_choice in any::<usize>(), right_choice in any::<usize>()) {
        let len = 1_usize << log;
        let bits = 512;
        let cl = SsaRing::coeff_limbs(bits).get();
        let root = 2 * bits / len;
        let transform = test_transform(bits);
        let left_count = 1 + left_choice % len;
        let right_count = 1 + right_choice % (len - left_count + 1);
        let count = left_count + right_count - 1;
        let mut left = input_matrix(len, cl, seed);
        let mut right = input_matrix(len, cl, seed.wrapping_add(1));
        left[left_count * cl..].fill(0);
        right[right_count * cl..].fill(0);
        let mut expected = left.clone();
        let mut full_right = right.clone();
        let plan = PointwiseMulPlan::from(bits);
        let mut scratch = vec![Limb::MAX; plan.scratch_len.get().max(cl)];
        // SAFETY: all matrices are complete and disjoint; spectra have exactly
        // the declared support and the pointwise strategy matches the ring.
        unsafe {
            SsaTransform::fft_in_place_with_executor(&mut expected, len, root, bits, false, left_count, &SequentialExecutor, &mut scratch);
            SsaTransform::fft_in_place_with_executor(&mut full_right, len, root, bits, false, right_count, &SequentialExecutor, &mut scratch);
            SsaPointwise::pointwise_multiply_with_executor(&mut expected, &mut full_right, len, NonZeroUsize::MIN, &plan, &SequentialExecutor, &mut scratch);
            SsaTransform::fft_in_place_with_executor(&mut expected, len, root, bits, true, len, &SequentialExecutor, &mut scratch);
            transform.forward(&mut left, len, root, left_count, count, &SequentialExecutor, &mut scratch);
            transform.forward(&mut right, len, root, right_count, count, &SequentialExecutor, &mut scratch);
            SsaPointwise::pointwise_multiply_with_executor(&mut left[..count * cl], &mut right[..count * cl], count, NonZeroUsize::MIN, &plan, &SequentialExecutor, &mut scratch);
            left[count * cl..].fill(Limb::MAX);
            transform.inverse(&mut left, len, root, count, count, false, &SequentialExecutor, &mut scratch);
        }
        canonicalize(&mut left[..count * cl], bits);
        canonicalize(&mut expected, bits);
        prop_assert_eq!(&left[..count * cl], &expected[..count * cl]);
        prop_assert!(expected[count * cl..].iter().all(|&limb| limb == 0));
    }
}

#[test]
fn tft_mixed_inverse_covers_all_known_tail_and_extra_frequency_shapes() {
    let bits = 192;
    let cl = SsaRing::coeff_limbs(bits).get();
    let transform = test_transform(bits);
    for len in [2, 4, 8, 16, 32] {
        if cfg!(miri) && len > 8 {
            continue;
        }
        let root = 2 * bits / len;
        for z in 1..=len {
            let mut time = input_matrix(len, cl, 29);
            time[z * cl..].fill(0);
            let mut frequency = time.clone();
            let mut scratch = vec![Limb::MAX; cl];
            // SAFETY: full initialized matrix with a physical zero tail, root,
            // and complete private scratch; time slots are semi-normalized.
            unsafe {
                SsaTransform::fft_in_place_with_executor(
                    &mut frequency,
                    len,
                    root,
                    bits,
                    false,
                    z,
                    &SequentialExecutor,
                    &mut scratch,
                );
                for slot in time.chunks_exact_mut(cl).take(z) {
                    SsaRing::shift_in_place(
                        slot,
                        len.trailing_zeros() as usize,
                        bits,
                        &mut scratch,
                    );
                }
            }
            canonicalize(&mut time, bits);
            canonicalize(&mut frequency, bits);
            for n in 0..=z {
                for extra in [false, true] {
                    if n + usize::from(extra) == 0 || n + usize::from(extra) > len {
                        continue;
                    }
                    let mut mixed = vec![Limb::MAX; len * cl];
                    mixed[..n * cl].copy_from_slice(&frequency[..n * cl]);
                    mixed[n * cl..z * cl].copy_from_slice(&time[n * cl..z * cl]);
                    // SAFETY: n frequencies, len-scaled known tail n..z, and
                    // implicit zeros above z satisfy the mixed ITFT contract.
                    unsafe {
                        transform.inverse(
                            &mut mixed,
                            len,
                            root,
                            z,
                            n,
                            extra,
                            &SequentialExecutor,
                            &mut scratch,
                        );
                    }
                    let output_span = (n + usize::from(extra)) * cl;
                    canonicalize(&mut mixed[..output_span], bits);
                    assert_eq!(
                        &mixed[..n * cl],
                        &time[..n * cl],
                        "len={len}, z={z}, n={n}, f={extra}"
                    );
                    if extra {
                        assert_eq!(
                            &mixed[n * cl..output_span],
                            &frequency[n * cl..output_span],
                            "extra frequency len={len}, z={z}, n={n}"
                        );
                    }
                }
            }
        }
    }
}
