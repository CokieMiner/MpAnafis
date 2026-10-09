//! Prefix block reconstruction with independent signed carry accumulation.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::integer_division,
    reason = "Small explicit reconstruction geometries bound all test offsets and shifts"
)]

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 32 }))]
    #[test]
    fn partial_reconstruction_blocks_cover_arbitrary_prefixes(
        chunk in prop::sample::select(vec![LIMB_BITS, LIMB_BITS + 1, LIMB_BITS + LIMB_BITS / 2]),
        workers in prop::sample::select(vec![2_usize, 3, 4, 8]),
        count in 33_usize..=if cfg!(miri) { 64 } else { 256 },
        seed in any::<Limb>(),
    ) {
        check_blocks(chunk, workers, count, seed);
    }
}

#[test]
fn partial_reconstruction_blocks_match_signed_accumulation() {
    for chunk in [LIMB_BITS, LIMB_BITS + 1, LIMB_BITS + LIMB_BITS / 2] {
        for workers in [2, 3, 4, 8] {
            for count in [33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256] {
                if cfg!(miri) && (workers > 3 || count > 64) {
                    continue;
                }
                for pattern in 0..3 {
                    check_blocks(chunk, workers, count, pattern);
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
#[test]
#[cfg_attr(
    miri,
    ignore = "Native Rayon products at 2048 limbs cross the parallel reconstruction threshold"
)]
fn rayon_truncated_products_reuse_exact_planned_arenas() {
    use crate::parallel::DefaultExecutor;

    let outer_limbs = 2048;
    let bits = outer_limbs * LIMB_BITS;
    let plan = MulTransformPlan::new(FftPlan::new(bits));
    for workers in [3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test pool");
        pool.install(|| {
            DefaultExecutor::with_resolved(|executor| {
                for count in [plan.transform_len / 2 + 1, plan.transform_len - 1] {
                    assert!(
                        ReconstructionBlocks::new(
                            plan.transform_len,
                            count,
                            plan.chunk_bits,
                            plan.inner_bits,
                            workers,
                        )
                        .is_some(),
                        "the shortened product must exercise parallel blocks"
                    );
                    let left_bits = (count / 2 + 1) * plan.chunk_bits.get();
                    let right_bits = (count - count / 2) * plan.chunk_bits.get();
                    let left = dense_prefix(left_bits);
                    let right = dense_prefix(right_bits);
                    let mut expected = vec![0; left.len() + right.len()];
                    Schoolbook::mul(&mut expected, &left, &right);
                    expected.resize(outer_limbs + 1, 0);
                    let mut scratch = vec![Limb::MAX; plan.transform_mul_scratch(workers) + 2];
                    let end = scratch.len() - 1;
                    let mut output = vec![Limb::MAX; outer_limbs + 3];
                    for _ in 0..2 {
                        // SAFETY: the planned limb-aligned ring contains both exact
                        // operand widths, their guarded output, and the exact worker
                        // arena. The proven product support is count < K; omitted
                        // coefficients are zero and all live spans are disjoint.
                        unsafe {
                            SsaTransform::fft_mul_mod_slices_with_executor(
                                &mut output[1..=outer_limbs + 1],
                                &left,
                                &right,
                                bits,
                                Some((left_bits, right_bits)),
                                true,
                                &plan,
                                executor,
                                &mut scratch[1..end],
                            );
                        }
                        assert_eq!(&output[1..=outer_limbs + 1], expected);
                        assert_eq!(output[0], Limb::MAX);
                        assert_eq!(output[outer_limbs + 2], Limb::MAX);
                        assert_eq!(scratch[0], Limb::MAX);
                        assert_eq!(scratch[end], Limb::MAX);
                    }
                }
            });
        });
    }
}

fn check_blocks(chunk: usize, workers: usize, count: usize, seed: Limb) {
    let len = if cfg!(miri) { 64 } else { 256 };
    let inner_limbs = if cfg!(miri) { 4 } else { 16 };
    let bits = inner_limbs * LIMB_BITS;
    let Some(layout) = ReconstructionBlocks::new(
        len,
        count,
        core::num::NonZeroUsize::new(chunk).expect("positive test chunk width"),
        bits,
        workers,
    ) else {
        return;
    };
    let cl = inner_limbs + 1;
    let outer_limbs = len * chunk / LIMB_BITS;
    let mut matrix = vec![0; count * cl];
    let mut expected = vec![0; outer_limbs + cl + 2];
    expected[0] = 1;
    expected[outer_limbs] = 1;
    let mut actual = expected.clone();
    let mut state = seed;
    for (index, coefficient) in matrix.chunks_exact_mut(cl).enumerate() {
        state = state.wrapping_mul(33).wrapping_add(7);
        let magnitude = if seed <= 2 { Limb::MAX } else { state };
        let negative = match seed {
            0 => false,
            1 => true,
            _ => index % 2 == 0,
        };
        coefficient[0] = magnitude;
        if negative {
            // SAFETY: complete initialized canonical coefficient, with guard zero.
            unsafe {
                SsaRing::negate(coefficient, bits);
            }
        }
        oracle_accumulate(&mut expected, index * chunk, magnitude, negative);
    }
    let mut scratch = vec![Limb::MAX; layout.scratch_len() + 2];
    let end = scratch.len() - 1;
    let executor = CountingExecutor::default();
    for _ in 0..2 {
        actual.fill(0);
        actual[0] = 1;
        actual[outer_limbs] = 1;
        // SAFETY: checked block geometry, complete canonical coefficient prefix,
        // and exact private scratch. Each magnitude is below R=2^chunk, so every
        // signed prefix has magnitude < R^len and the outer bias prevents borrow
        // escape. The arena includes each overlap and its final carry position.
        unsafe {
            layout.run(&matrix, &mut actual, &mut scratch[1..end], &executor);
        }
        assert_eq!(
            actual, expected,
            "chunk={chunk}, workers={workers}, count={count}, seed={seed}"
        );
        assert_eq!(scratch[0], Limb::MAX);
        assert_eq!(scratch[end], Limb::MAX);
    }
    assert!(executor.joins.load(Ordering::Relaxed) > 0);
}

fn oracle_accumulate(dst: &mut [Limb], shift: usize, magnitude: Limb, negative: bool) {
    let offset = shift / LIMB_BITS;
    let bits = shift % LIMB_BITS;
    let low = magnitude << bits;
    let high = if bits == 0 {
        0
    } else {
        magnitude >> (LIMB_BITS - bits)
    };
    let mut carry = false;
    for (index, limb) in dst[offset..].iter_mut().enumerate() {
        let source = match index {
            0 => low,
            1 => high,
            _ => 0,
        };
        let (value, first) = if negative {
            limb.overflowing_sub(source)
        } else {
            limb.overflowing_add(source)
        };
        let (result, second) = if negative {
            value.overflowing_sub(Limb::from(carry))
        } else {
            value.overflowing_add(Limb::from(carry))
        };
        *limb = result;
        carry = first || second;
    }
    assert!(!carry, "outer bias and guard contain every signed prefix");
}

#[cfg(feature = "rayon")]
fn dense_prefix(bits: usize) -> Vec<Limb> {
    let mut limbs = vec![Limb::MAX; bits.div_ceil(LIMB_BITS)];
    let remaining = bits % LIMB_BITS;
    if remaining != 0 {
        *limbs.last_mut().expect("nonempty operand") = (1 << remaining) - 1;
    }
    limbs
}
