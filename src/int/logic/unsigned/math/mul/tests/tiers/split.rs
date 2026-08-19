//! Property tests for SSA coefficient splitting.

use alloc::{vec, vec::Vec};

use super::*;

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
    let mut scratch = vec![0; cl];

    SsaCoefficients::split(&source, &mut split, transform_len, chunk_bits, inner_bits);
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
    let fused = unsafe {
        SsaCoefficients::split_twisted(
            &source,
            &mut actual,
            transform_len,
            chunk_bits,
            inner_bits,
            twist_step_half,
            &mut scratch,
        )
    };
    assert!(fused, "an even half-bit step must use the fused path");
    assert_eq!(actual, expected, "fused decomposition changed the twist");
}

#[test]
fn odd_half_bit_twist_declines_without_writing() {
    let mut matrix = vec![Limb::MAX; 18];
    let original = matrix.clone();
    let mut scratch = vec![0; 9];
    // SAFETY: both buffers satisfy the requested two-coefficient layout.
    let fused = unsafe {
        SsaCoefficients::split_twisted(&[1, 2], &mut matrix, 2, 64, 512, 3, &mut scratch)
    };
    assert!(!fused, "odd half-bit twist requires the sqrt(2) path");
    assert_eq!(matrix, original, "declined fusion wrote the matrix");
}
