//! Independent small-DFT oracle and the transform cross-check against it.

#![expect(
    unsafe_code,
    reason = "Small DFT comparisons use complete disjoint coefficients, valid primitive roots, and exact scratch spans"
)]

use super::*;

pub fn oracle_add_mod(lhs: &[Limb], rhs: &[Limb], mod_bits: usize) -> Vec<Limb> {
    let cl = ring::SsaRing::coeff_limbs(mod_bits).get();
    let mut result = vec![0; cl];
    let mut carry = 0_u128;
    for ((result_limb, lhs_limb), rhs_limb) in result.iter_mut().zip(lhs).zip(rhs) {
        let sum = u128::try_from(*lhs_limb)
            .expect("limb fits u128")
            .wrapping_add(u128::try_from(*rhs_limb).expect("limb fits u128"))
            .wrapping_add(carry);
        *result_limb = Limb::try_from(sum & u128::try_from(Limb::MAX).expect("limb fits u128"))
            .expect("masked digit fits one limb");
        carry = sum >> Limb::BITS;
    }
    assert_eq!(
        carry, 0,
        "the fixed oracle slot must hold a sum of residues"
    );

    // Each input is below q = 2^n + 1, so the sum is below 2q and one
    // subtraction of q is sufficient. q has a one in data limb zero and the
    // guard limb; this representation also works for every Limb width.
    let guard = result.last().copied().expect("complete guard");
    let data_nonzero = result
        .get(..cl.wrapping_sub(1))
        .expect("complete data")
        .iter()
        .any(|limb| *limb != 0);
    if guard > 1 || (guard == 1 && data_nonzero) {
        let mut borrow = false;
        for (index, limb) in result.iter_mut().enumerate() {
            let modulus_limb = Limb::from(index == 0 || index == cl.wrapping_sub(1));
            let (after_modulus, modulus_borrow) = limb.overflowing_sub(modulus_limb);
            let (after_borrow, incoming_borrow) = after_modulus.overflowing_sub(Limb::from(borrow));
            *limb = after_borrow;
            borrow = modulus_borrow || incoming_borrow;
        }
        assert!(!borrow, "the oracle residue subtraction must not underflow");
    }
    result
}

pub fn oracle_shift(value: &[Limb], shift: usize, mod_bits: usize) -> Vec<Limb> {
    let period = mod_bits.wrapping_mul(2);
    let reduced_shift = shift.rem_euclid(period);
    let mut result = value.to_vec();
    for _ in 0..reduced_shift {
        result = oracle_add_mod(&result, &result, mod_bits);
    }
    result
}

fn oracle_dft(
    input: &[Vec<Limb>],
    root_shift: usize,
    mod_bits: usize,
    inverse: bool,
) -> Vec<Vec<Limb>> {
    let period = mod_bits.wrapping_mul(2);
    let mut output = Vec::with_capacity(input.len());
    for frequency in 0..input.len() {
        let mut sum = vec![0; ring::SsaRing::coeff_limbs(mod_bits).get()];
        for (index, value) in input.iter().enumerate() {
            let exponent = root_shift
                .wrapping_mul(index)
                .wrapping_mul(frequency)
                .rem_euclid(period);
            let effective_exponent = if inverse {
                period.wrapping_sub(exponent).rem_euclid(period)
            } else {
                exponent
            };
            let term = oracle_shift(value, effective_exponent, mod_bits);
            sum = oracle_add_mod(&sum, &term, mod_bits);
        }
        output.push(sum);
    }
    output
}

fn oracle_bit_reverse(index: usize, transform_log: usize) -> usize {
    let mut reversed = 0_usize;
    for bit in 0..transform_log {
        reversed = reversed.wrapping_shl(1).wrapping_add((index >> bit) & 1);
    }
    reversed
}

fn oracle_coefficients(transform_len: usize, mod_bits: usize) -> Vec<Vec<Vec<Limb>>> {
    let ml = ring::SsaRing::mod_limbs(mod_bits);
    let cl = ring::SsaRing::coeff_limbs(mod_bits).get();
    let zero = vec![0; cl];
    let mut neg_one = vec![0; cl];
    *neg_one.get_mut(ml).expect("complete guard") = 1;
    let mut max = vec![Limb::MAX; cl];
    *max.get_mut(ml).expect("complete guard") = 0;
    let mut mixed = vec![0; cl];
    for (index, limb) in mixed.iter_mut().take(ml).enumerate() {
        let offset = index.wrapping_add(3);
        *limb = Limb::MAX.wrapping_sub(offset);
    }

    let mut alternating = Vec::with_capacity(transform_len);
    for index in 0..transform_len {
        alternating.push(match index % 4 {
            0 => zero.clone(),
            1 => neg_one.clone(),
            2 => max.clone(),
            _ => mixed.clone(),
        });
    }

    let mut ramp = Vec::with_capacity(transform_len);
    for index in 0..transform_len {
        let mut value = vec![0; cl];
        for (limb_index, limb) in value.iter_mut().take(ml).enumerate() {
            let seed = index
                .wrapping_add(1)
                .wrapping_mul(limb_index.wrapping_add(5));
            *limb = seed;
        }
        ramp.push(value);
    }

    vec![alternating, ramp]
}

fn flatten_oracle_coefficients(coefficients: &[Vec<Limb>]) -> Vec<Limb> {
    coefficients
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect()
}

fn canonicalize_oracle_coefficient(value: &mut [Limb], mod_bits: usize) {
    let ml = ring::SsaRing::mod_limbs(mod_bits);
    let guard = *value.last().expect("complete guard");
    assert!(
        guard <= 1,
        "FFT output must be semi-normalized before oracle canonicalization"
    );
    if guard == 0 {
        return;
    }

    let data = value.get_mut(..ml).expect("complete data");
    if data.iter().all(|limb| *limb == 0) {
        // The only canonical residue with guard one is 2^n, representing -1.
        return;
    }

    // For a semi-normalized slot low + 2^n, the Fermat relation gives low - 1.
    // Since low is nonzero here, subtracting one cannot escape the data width,
    // and clearing the guard produces the canonical representative.
    let mut borrow = true;
    for limb in data {
        let (difference, next_borrow) = limb.overflowing_sub(Limb::from(borrow));
        *limb = difference;
        borrow = next_borrow;
    }
    assert!(
        !borrow,
        "nonzero oracle data must absorb the guard correction"
    );
    *value.last_mut().expect("complete guard") = 0;
}

fn canonicalize_transform_matrix(matrix: &mut [Limb], transform_len: usize, mod_bits: usize) {
    let cl = ring::SsaRing::coeff_limbs(mod_bits).get();
    for slot in matrix.chunks_exact_mut(cl).take(transform_len) {
        canonicalize_oracle_coefficient(slot, mod_bits);
    }
}

#[test]
fn ssa_fft_matches_independent_small_fermat_dft() {
    let mod_bits = LIMB_BITS.wrapping_mul(2);
    let period = mod_bits.wrapping_mul(2);
    let shift_period = NonZeroUsize::new(period).expect("positive oracle ring period");
    for transform_len in [1_usize, 2, 4, 8] {
        let transform_log =
            usize::try_from(transform_len.trailing_zeros()).expect("bit count fits usize");
        let root_shift = period.div_euclid(transform_len).rem_euclid(period);
        let cl = ring::SsaRing::coeff_limbs(mod_bits).get();
        for input in oracle_coefficients(transform_len, mod_bits) {
            let expected_forward = oracle_dft(&input, root_shift, mod_bits, false);
            let mut forward_matrix = flatten_oracle_coefficients(&input);
            let mut scratch = vec![0; cl];
            // SAFETY: the matrix has transform_len complete coefficients, the
            // root has the requested transform order, and scratch has one slot.
            unsafe {
                transform::SsaTransform::fft_in_place_with_executor(
                    &mut forward_matrix,
                    transform_len,
                    root_shift,
                    mod_bits,
                    false,
                    transform_len,
                    &SequentialExecutor,
                    &mut scratch,
                );
            }
            canonicalize_transform_matrix(&mut forward_matrix, transform_len, mod_bits);
            for (output_index, actual) in forward_matrix.chunks_exact(cl).enumerate() {
                let frequency = oracle_bit_reverse(output_index, transform_log);
                assert_eq!(
                    actual,
                    expected_forward.get(frequency).expect("frequency fits"),
                    "full DIF output at transform length {transform_len}, slot {output_index}"
                );
            }

            let mut retained_input = input.clone();
            for coefficient in retained_input.iter_mut().skip(transform_len >> 1) {
                coefficient.fill(0);
            }
            let retained_expected = oracle_dft(&retained_input, root_shift, mod_bits, false);
            let mut retained_matrix = flatten_oracle_coefficients(&retained_input);
            // SAFETY: the upper coefficient half is zero, as required by the
            // retained-input DIF specialization; all other layout proofs match above.
            unsafe {
                transform::SsaTransform::fft_in_place_with_executor(
                    &mut retained_matrix,
                    transform_len,
                    root_shift,
                    mod_bits,
                    false,
                    transform_len >> 1,
                    &SequentialExecutor,
                    &mut scratch,
                );
            }
            canonicalize_transform_matrix(&mut retained_matrix, transform_len, mod_bits);
            for (output_index, actual) in retained_matrix.chunks_exact(cl).enumerate() {
                let frequency = oracle_bit_reverse(output_index, transform_log);
                assert_eq!(
                    actual,
                    retained_expected.get(frequency).expect("frequency fits"),
                    "retained DIF output at transform length {transform_len}, slot {output_index}"
                );
            }

            // Feed the independently computed frequencies to the inverse. This
            // keeps a shared forward/inverse defect from satisfying a round trip.
            let expected_inverse = oracle_dft(&expected_forward, root_shift, mod_bits, true);
            let inverse_input = (0..transform_len)
                .map(|output_index| {
                    expected_forward
                        .get(oracle_bit_reverse(output_index, transform_log))
                        .expect("frequency fits")
                        .clone()
                })
                .collect::<Vec<_>>();
            let mut inverse_matrix = flatten_oracle_coefficients(&inverse_input);
            // SAFETY: inverse_input is the forward DIF bit-reversed layout, and
            // the matrix/scratch satisfy the same complete-slot contracts.
            unsafe {
                transform::SsaTransform::fft_in_place_with_executor(
                    &mut inverse_matrix,
                    transform_len,
                    root_shift,
                    mod_bits,
                    true,
                    transform_len,
                    &SequentialExecutor,
                    &mut scratch,
                );
            }
            canonicalize_transform_matrix(&mut inverse_matrix, transform_len, mod_bits);
            for (index, actual) in inverse_matrix.chunks_exact(cl).enumerate() {
                assert_eq!(
                    actual,
                    expected_inverse.get(index).expect("output index fits"),
                    "unscaled inverse output at transform length {transform_len}, slot {index}"
                );
            }

            let inverse_scale =
                ring::SsaRing::reduce_mod_period(period.wrapping_sub(transform_log), shift_period);
            for slot in inverse_matrix.chunks_exact_mut(cl) {
                // SAFETY: canonicalize_transform_matrix established canonical
                // slots; scratch is disjoint and has a complete coefficient slot.
                // The scale is reduced above, so it meets the shift contract
                // even when the unit transform leaves a full-period exponent.
                unsafe {
                    ring::SsaRing::shift_in_place(slot, inverse_scale, mod_bits, &mut scratch);
                }
            }
            assert_eq!(inverse_matrix, flatten_oracle_coefficients(&input));
        }
    }
}
