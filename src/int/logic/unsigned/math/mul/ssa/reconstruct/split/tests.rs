//! Reference tests for SSA coefficient splitting.

#![expect(
    unsafe_code,
    reason = "Splitting tests establish complete source, coefficient and scratch spans before exercising unsafe reference and production kernels"
)]

use core::num::NonZeroUsize;

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
    let coefficient_limbs = SsaRing::coeff_limbs(inner_bits).get();
    let active_chunks = source
        .len()
        .saturating_mul(LIMB_BITS)
        .div_ceil(chunk_bits)
        .min(transform_len);
    matrix.fill(0);
    for chunk in 0..active_chunks {
        let source_start = chunk.saturating_mul(chunk_bits);
        for output_bit in 0..chunk_bits.min(inner_bits) {
            let source_bit = source_start.saturating_add(output_bit);
            let source_limb = source_bit.div_euclid(LIMB_BITS);
            if source_limb >= source.len() {
                break;
            }
            // SAFETY: the guard above breaks when `source_limb >= source.len()`.
            let bit = (unsafe { *source.get_unchecked(source_limb) }
                >> source_bit.wrapping_rem(LIMB_BITS))
                & 1;
            let output_limb = chunk
                .saturating_mul(coefficient_limbs)
                .saturating_add(output_bit.div_euclid(LIMB_BITS));
            // SAFETY: `chunk < active_chunks <= transform_len` and
            // `output_bit < inner_bits` bound the index strictly below
            // `transform_len * coeff_limbs(inner_bits)`, the exact matrix
            // length the callers allocate.
            let output = unsafe { matrix.get_unchecked_mut(output_limb) };
            *output |= bit << output_bit.wrapping_rem(LIMB_BITS);
        }
    }
}

#[test]
fn fused_whole_bit_twist_matches_two_pass_decomposition() {
    let transform_len = 8_usize;
    let chunk_bits = 128_usize;
    let inner_bits = 512_usize;
    let twist_step_half = 128_usize;
    let cl = SsaRing::coeff_limbs(inner_bits).get();
    let source: Vec<Limb> = (0_usize..16)
        .map(|index| index.wrapping_mul(0x9E37_79B9) | 1)
        .collect();
    let mut split = vec![0; transform_len.wrapping_mul(cl)];
    let mut expected = vec![0; split.len()];
    let mut actual = vec![0; split.len()];
    let mut scratch = vec![0; cl.wrapping_mul(2)];

    reference_split(&source, &mut split, transform_len, chunk_bits, inner_bits);
    let period = NonZeroUsize::new(inner_bits.wrapping_mul(2)).expect("positive test ring period");
    let periods = RingPeriods {
        whole: period,
        half: NonZeroUsize::new(inner_bits.wrapping_mul(4)).expect("positive test half-bit period"),
    };
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
            NonZeroUsize::new(chunk_bits).expect("positive test chunk width"),
            NonZeroUsize::new(cl).expect("positive complete test coefficient width"),
            periods,
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
    let cl = SsaRing::coeff_limbs(inner_bits).get();
    let source: Vec<Limb> = (0_usize..16)
        .map(|index| index.wrapping_mul(0x9E37_79B9) | 1)
        .collect();
    let mut split = vec![0; transform_len.wrapping_mul(cl)];
    let mut expected = vec![0; split.len()];
    let mut actual = vec![0; split.len()];
    let mut scratch = vec![0; cl.wrapping_mul(2)];

    reference_split(&source, &mut split, transform_len, chunk_bits, inner_bits);
    let periods = RingPeriods {
        whole: NonZeroUsize::new(inner_bits.wrapping_mul(2)).expect("positive test ring period"),
        half: NonZeroUsize::new(inner_bits.wrapping_mul(4)).expect("positive test half-bit period"),
    };
    let half_period = periods.half;
    let mut shift = 0_usize;
    for (input, output) in split.chunks_exact(cl).zip(expected.chunks_exact_mut(cl)) {
        output.copy_from_slice(input);
        // SAFETY: the output chunk is canonical and the scratch holds the
        // two-coefficient arena the shift needs. The halved half-bit shift is
        // already below the 2n period; zero inputs shift to zero.
        unsafe {
            SsaRing::shift_in_place(output, shift.wrapping_shr(1), inner_bits, &mut scratch);
        }
        if !shift.is_multiple_of(2) {
            // SAFETY: the same two-coefficient arena covers the factor.
            unsafe {
                SsaRing::shift_sqrt2(output, 0, inner_bits, &mut scratch);
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
            NonZeroUsize::new(chunk_bits).expect("positive test chunk width"),
            NonZeroUsize::new(cl).expect("positive complete test coefficient width"),
            periods,
            twist_step_half,
            &mut scratch,
        );
    }
    assert_eq!(
        actual, expected,
        "fused sqrt(2) twist changed the decomposition"
    );
}
