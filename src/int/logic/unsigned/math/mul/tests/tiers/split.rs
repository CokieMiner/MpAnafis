//! Property tests for SSA coefficient splitting.

use alloc::{vec, vec::Vec};

use super::*;

/// Reference decomposition: plain chunk extraction with no twist.
fn reference_split(
    source: &[Limb],
    matrix: &mut [Limb],
    transform_len: usize,
    chunk_bits: usize,
    inner_bits: usize,
) {
    let layout = SplitLayout::new(chunk_bits, inner_bits);
    let active_chunks = source
        .len()
        .saturating_mul(LIMB_BITS)
        .div_ceil(chunk_bits)
        .min(transform_len);
    let active_limbs = active_chunks.wrapping_mul(layout.cl);
    // SAFETY: active_limbs and every slot offset stay below matrix.len() by
    // construction, mirroring the production split's bounds proofs.
    unsafe {
        matrix.get_unchecked_mut(active_limbs..).fill(0);
        for index in 0..active_chunks {
            let slot = matrix
                .get_unchecked_mut(index.wrapping_mul(layout.cl)..)
                .get_unchecked_mut(..layout.cl);
            extract_chunk(source, slot, index, layout);
        }
    }
}

#[test]
fn fused_whole_bit_twist_matches_two_pass_decomposition() {
    let transform_len = 8_usize;
    let chunk_bits = 128_usize;
    let inner_bits = 512_usize;
    let twist_step_half = 128_usize;
    let cl = SsaRing::coeff_limbs(inner_bits);
    let source: Vec<Limb> = (0_usize..16)
        .map(|index| index.wrapping_mul(0x9E37_79B9) | 1)
        .collect();
    let mut split = vec![0; transform_len.wrapping_mul(cl)];
    let mut expected = vec![0; split.len()];
    let mut actual = vec![0; split.len()];
    let mut scratch = vec![0; cl.wrapping_mul(2)];

    reference_split(&source, &mut split, transform_len, chunk_bits, inner_bits);
    let period = inner_bits.wrapping_mul(2);
    let whole_step = twist_step_half.wrapping_shr(1);
    let mut shift = 0_usize;
    for (input, output) in split.chunks_exact(cl).zip(expected.chunks_exact_mut(cl)) {
        if shift == 0 {
            output.copy_from_slice(input);
        } else {
            // SAFETY: both chunks are disjoint complete coefficients.
            unsafe {
                SsaRing::shift_from(output, input, shift, inner_bits);
            }
        }
        shift = SsaRing::reduce_mod_period(shift.wrapping_add(whole_step), period);
    }

    // SAFETY: actual and scratch have the exact disjoint layouts required.
    unsafe {
        SsaCoefficients::split_twisted(
            &source,
            &mut actual,
            transform_len,
            chunk_bits,
            inner_bits,
            twist_step_half,
            &mut scratch,
        );
    }
    assert_eq!(actual, expected, "fused decomposition changed the twist");
}

#[test]
fn fused_odd_half_bit_twist_matches_sqrt2_decomposition() {
    let transform_len = 8_usize;
    let chunk_bits = 128_usize;
    let inner_bits = 512_usize;
    let twist_step_half = 3_usize;
    let cl = SsaRing::coeff_limbs(inner_bits);
    let source: Vec<Limb> = (0_usize..16)
        .map(|index| index.wrapping_mul(0x9E37_79B9) | 1)
        .collect();
    let mut split = vec![0; transform_len.wrapping_mul(cl)];
    let mut expected = vec![0; split.len()];
    let mut actual = vec![0; split.len()];
    let mut scratch = vec![0; cl.wrapping_mul(2)];

    reference_split(&source, &mut split, transform_len, chunk_bits, inner_bits);
    let half_period = inner_bits.wrapping_mul(4);
    let mut shift = 0_usize;
    for (input, output) in split.chunks_exact(cl).zip(expected.chunks_exact_mut(cl)) {
        output.copy_from_slice(input);
        // SAFETY: the output chunk is canonical and the scratch holds the
        // two-coefficient arena the shift needs.
        unsafe {
            SsaRing::shift(output, shift.wrapping_shr(1), inner_bits, &mut scratch);
        }
        if !shift.is_multiple_of(2) {
            // SAFETY: the same two-coefficient arena covers the factor.
            unsafe {
                SsaRing::mul_sqrt2(output, inner_bits, &mut scratch);
            }
        }
        shift = SsaRing::reduce_mod_period(shift.wrapping_add(twist_step_half), half_period);
    }

    // SAFETY: actual and scratch have the exact disjoint layouts required.
    unsafe {
        SsaCoefficients::split_twisted(
            &source,
            &mut actual,
            transform_len,
            chunk_bits,
            inner_bits,
            twist_step_half,
            &mut scratch,
        );
    }
    assert_eq!(
        actual, expected,
        "fused sqrt(2) twist changed the decomposition"
    );
}
