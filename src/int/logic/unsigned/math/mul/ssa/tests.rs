//! Property tests for recursive Schönhage-Strassen multiplication.

use alloc::{vec, vec::Vec};
use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use proptest::prelude::*;

use super::*;
use crate::{
    int::logic::math::mul::{Schoolbook, ssa::SsaPointwise},
    parallel::{ParallelExecutor, SequentialExecutor},
};

macro_rules! some_or_panic {
    ($value:expr, $message:literal) => {{
        let Some(value) = $value else {
            panic!($message);
        };
        value
    }};
}

macro_rules! ok_or_panic {
    ($value:expr, $message:literal) => {{
        let Ok(value) = $value else {
            panic!($message);
        };
        value
    }};
}

#[derive(Debug, Default)]
struct CountingExecutor {
    joins: AtomicUsize,
}

impl ParallelExecutor for CountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        some_or_panic!(
            NonZeroUsize::new(8),
            "the test executor must have eight logical workers"
        )
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _ = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

#[derive(Debug, Default)]
struct OneSlotCountingExecutor {
    joins: AtomicUsize,
}

impl ParallelExecutor for OneSlotCountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _previous = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

fn operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 1..=4),
            proptest::collection::vec(any::<Limb>(), 1..=4),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 5),
            proptest::collection::vec(any::<Limb>(), 5),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 9),
            proptest::collection::vec(any::<Limb>(), 7),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 17),
            proptest::collection::vec(any::<Limb>(), 17),
        ),
    ]
}

#[test]
fn parallel_grain_is_based_on_work_per_worker() {
    let executor = CountingExecutor::default();
    let workers = executor.parallelism().get();
    let threshold = SSA_PARALLEL_MIN_LIMB_WORK;

    assert!(!transform::SsaTransform::has_parallel_work(
        2, threshold, &executor
    ));
    assert!(transform::SsaTransform::has_parallel_work(
        workers, threshold, &executor
    ));
}

#[test]
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn direct_fermat_requires_the_measured_wide_pool() {
    assert_eq!(Ssa::direct_fermat_threshold(1), None);
    assert_eq!(Ssa::direct_fermat_threshold(2), None);
    assert_eq!(Ssa::direct_fermat_threshold(7), None);
    assert_eq!(Ssa::direct_fermat_threshold(8), Some(1_048_576));
    assert_eq!(Ssa::direct_fermat_threshold(16), Some(1_048_576));
}

#[test]
fn wide_negated_shift_from_matches_in_place_shift() {
    for ml in [65_usize, 128, 256, 1_024] {
        let mod_bits = ml.wrapping_mul(LIMB_BITS);
        let mut source = (0..=ml)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                    .rotate_left(13)
            })
            .collect::<Vec<_>>();
        *some_or_panic!(source.last_mut(), "Fermat coefficient must be nonempty") = 0;
        for shift in [
            mod_bits.wrapping_add(1),
            mod_bits.wrapping_add(31),
            mod_bits.wrapping_add(63),
            mod_bits.wrapping_add(LIMB_BITS).wrapping_add(31),
            mod_bits.wrapping_mul(2).wrapping_sub(1),
        ] {
            let mut expected = source.clone();
            let mut scratch = vec![Limb::MIN; ml.wrapping_add(1)];
            let mut actual = vec![Limb::MIN; ml.wrapping_add(1)];

            // SAFETY: all coefficients and scratch contain exactly ml + 1
            // limbs, source has a canonical zero guard, and the spans are
            // independently allocated.
            unsafe {
                SsaRing::shift_in_place(&mut expected, shift, mod_bits, &mut scratch);
                SsaRing::shift_from(&mut actual, &source, shift, mod_bits);
            }
            assert_eq!(
                actual, expected,
                "wide negated shift differs at ml={ml}, shift={shift}"
            );
        }
    }
}

fn transform_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    let base_limbs = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS);
    let minimum_len = base_limbs.div_euclid(2).wrapping_add(1);
    let maximum_len = minimum_len.wrapping_add(2);
    (
        proptest::collection::vec(any::<Limb>(), minimum_len..=maximum_len),
        proptest::collection::vec(any::<Limb>(), minimum_len..=maximum_len),
    )
}

fn oracle_limb(value: u128) -> Limb {
    ok_or_panic!(Limb::try_from(value), "oracle value must fit in one limb")
}

fn oracle_add_mod(lhs: &[Limb], rhs: &[Limb], mod_bits: usize) -> Vec<Limb> {
    let cl = ring::SsaRing::coeff_limbs(mod_bits);
    let mut result = vec![0; cl];
    let mut carry = 0_u128;
    for ((result_limb, lhs_limb), rhs_limb) in result.iter_mut().zip(lhs).zip(rhs) {
        let sum = ok_or_panic!(u128::try_from(*lhs_limb), "a limb must fit in u128")
            .wrapping_add(ok_or_panic!(
                u128::try_from(*rhs_limb),
                "a limb must fit in u128"
            ))
            .wrapping_add(carry);
        *result_limb = oracle_limb(
            sum & ok_or_panic!(u128::try_from(Limb::MAX), "a limb maximum must fit in u128"),
        );
        carry = sum >> Limb::BITS;
    }
    assert_eq!(
        carry, 0,
        "the fixed oracle slot must hold a sum of residues"
    );

    // Each input is below q = 2^n + 1, so the sum is below 2q and one
    // subtraction of q is sufficient. q has a one in data limb zero and the
    // guard limb; this representation also works for every Limb width.
    let guard = some_or_panic!(result.last().copied(), "oracle slot must have a guard limb");
    let data_nonzero = some_or_panic!(
        result.get(..cl.wrapping_sub(1)),
        "oracle slot must have data limbs"
    )
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

fn oracle_shift(value: &[Limb], shift: usize, mod_bits: usize) -> Vec<Limb> {
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
        let mut sum = vec![0; ring::SsaRing::coeff_limbs(mod_bits)];
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
    let cl = ring::SsaRing::coeff_limbs(mod_bits);
    let zero = vec![0; cl];
    let mut neg_one = vec![0; cl];
    *some_or_panic!(neg_one.get_mut(ml), "oracle slot must have a guard limb") = 1;
    let mut max = vec![Limb::MAX; cl];
    *some_or_panic!(max.get_mut(ml), "oracle slot must have a guard limb") = 0;
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
    let guard = *some_or_panic!(value.last(), "oracle slot must have a guard limb");
    assert!(
        guard <= 1,
        "FFT output must be semi-normalized before oracle canonicalization"
    );
    if guard == 0 {
        return;
    }

    let data = some_or_panic!(
        value.get_mut(..ml),
        "oracle slot must have the complete data width"
    );
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
    *some_or_panic!(value.last_mut(), "oracle slot must have a guard limb") = 0;
}

fn canonicalize_transform_matrix(matrix: &mut [Limb], transform_len: usize, mod_bits: usize) {
    let cl = ring::SsaRing::coeff_limbs(mod_bits);
    for slot in matrix.chunks_exact_mut(cl).take(transform_len) {
        canonicalize_oracle_coefficient(slot, mod_bits);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the oracle test keeps full, retained, inverse, and scaling assertions together"
)]
#[test]
fn ssa_fft_matches_independent_small_fermat_dft() {
    let mod_bits = LIMB_BITS.wrapping_mul(2);
    let period = mod_bits.wrapping_mul(2);
    for transform_len in [1_usize, 2, 4, 8] {
        let transform_log = ok_or_panic!(
            usize::try_from(transform_len.trailing_zeros()),
            "a usize bit count must fit in usize"
        );
        let root_shift = period.div_euclid(transform_len).rem_euclid(period);
        let cl = ring::SsaRing::coeff_limbs(mod_bits);
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
                    some_or_panic!(
                        expected_forward.get(frequency),
                        "oracle frequency must be in range"
                    ),
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
                    some_or_panic!(
                        retained_expected.get(frequency),
                        "oracle frequency must be in range"
                    ),
                    "retained DIF output at transform length {transform_len}, slot {output_index}"
                );
            }

            // Feed the independently computed frequencies to the inverse. This
            // keeps a shared forward/inverse defect from satisfying a round trip.
            let expected_inverse = oracle_dft(&expected_forward, root_shift, mod_bits, true);
            let inverse_input = (0..transform_len)
                .map(|output_index| {
                    some_or_panic!(
                        expected_forward.get(oracle_bit_reverse(output_index, transform_log)),
                        "oracle frequency must be in range"
                    )
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
                    some_or_panic!(
                        expected_inverse.get(index),
                        "oracle output index must be in range"
                    ),
                    "unscaled inverse output at transform length {transform_len}, slot {index}"
                );
            }

            let inverse_scale = mod_bits.wrapping_mul(2).wrapping_sub(transform_log);
            for slot in inverse_matrix.chunks_exact_mut(cl) {
                // SAFETY: canonicalize_transform_matrix established canonical
                // slots; scratch is disjoint and has a complete coefficient slot.
                unsafe {
                    ring::SsaRing::shift(slot, inverse_scale, mod_bits, &mut scratch);
                }
            }
            assert_eq!(inverse_matrix, flatten_oracle_coefficients(&input));
        }
    }
}

#[test]
fn regression_single_limb_fermat_product_uses_scalar_basecase() {
    let modulus_bits = LIMB_BITS;
    let left = [Limb::MAX, 0];
    let right = [Limb::MAX.wrapping_sub(1), 0];
    let mut actual = [Limb::MAX; 2];
    let mut scratch = vec![Limb::MIN; SsaPointwise::fermat_basecase_scratch_len(modulus_bits)];

    // SAFETY: each coefficient has cl = 2 limbs, both operands are canonical
    // ordinary residues, and scratch has the exact required product width.
    unsafe {
        SsaPointwise::fermat_basecase_mul_into(
            &mut actual,
            &left,
            &right,
            modulus_bits,
            &mut scratch,
        );
    }

    // Limb::MAX represents -2 and Limb::MAX-1 represents -3 modulo B+1,
    // where B = 2^LIMB_BITS, so their product is the canonical residue 6.
    assert_eq!(actual, [6, 0]);
}

#[test]
fn fft_mul_mod_slices_accepts_short_implicit_zero_operands() {
    let modulus_bits = 512;
    let cl = SsaRing::coeff_limbs(modulus_bits);
    let plan = FftPlan::new(modulus_bits);
    let left = [3];
    let right = [5];

    let mut short_result = vec![0; cl];
    let mut short_scratch = vec![0; plan.transform_mul_scratch()];
    // SAFETY: both operands omit the guard and carry their exact nonzero widths;
    // the destination and scratch satisfy the plan's validated transform layout.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut short_result,
            &left,
            &right,
            modulus_bits,
            Some((2, 3)),
            true,
            Some(&plan),
            &crate::parallel::SequentialExecutor,
            &mut short_scratch,
        );
    }

    let mut guarded_left = vec![0; cl];
    let mut guarded_right = vec![0; cl];
    *some_or_panic!(
        guarded_left.first_mut(),
        "coefficient geometry must provide a data limb"
    ) = 3;
    *some_or_panic!(
        guarded_right.first_mut(),
        "coefficient geometry must provide a data limb"
    ) = 5;
    let mut guarded_result = vec![0; cl];
    let mut guarded_scratch = vec![0; plan.transform_mul_scratch()];
    // SAFETY: both operands include the complete guard and all buffers match the plan.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut guarded_result,
            &guarded_left,
            &guarded_right,
            modulus_bits,
            None,
            true,
            Some(&plan),
            &crate::parallel::SequentialExecutor,
            &mut guarded_scratch,
        );
    }

    assert_eq!(short_result, guarded_result);
    assert_eq!(
        some_or_panic!(
            short_result.first().copied(),
            "coefficient geometry must provide a non-empty result"
        ),
        15
    );

    let mut exact_result = [Limb::MAX; 1];
    let mut exact_scratch = vec![0; plan.transform_mul_scratch()];
    // SAFETY: 3 and 5 have exact significant widths two and three, so their
    // five-bit product fits the guard-free one-limb destination.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut exact_result,
            &left,
            &right,
            modulus_bits,
            Some((2, 3)),
            true,
            Some(&plan),
            &crate::parallel::SequentialExecutor,
            &mut exact_scratch,
        );
    }
    assert_eq!(exact_result, [15]);
}

#[test]
fn internal_parallel_policy_matches_sequential_fft_product() {
    let modulus_bits = 512;
    let ml = SsaRing::mod_limbs(modulus_bits);
    let cl = SsaRing::coeff_limbs(modulus_bits);
    let plan = FftPlan::new(modulus_bits);
    let mut left = vec![0; ml];
    let mut right = vec![0; ml];
    *some_or_panic!(
        left.first_mut(),
        "modulus geometry must provide a non-empty limb slice"
    ) = 7;
    *some_or_panic!(
        right.first_mut(),
        "modulus geometry must provide a non-empty limb slice"
    ) = 11;

    let mut parallel_result = vec![0; cl];
    let parallel = CountingExecutor::default();
    let parallel_slots = plan.parallel_slots(parallel.parallelism().get());
    let mut parallel_scratch = vec![0; plan.transform_mul_scratch_for_slots(parallel_slots)];
    // SAFETY: both operands omit the guard and carry their exact nonzero widths;
    // the destination and scratch satisfy the plan's validated transform layout.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut parallel_result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &parallel,
            &mut parallel_scratch,
        );
    }

    let mut sequential_result = vec![0; cl];
    let mut sequential_scratch = vec![0; plan.transform_mul_scratch()];
    // SAFETY: the same validated layout is used for the independent sequential run.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut sequential_result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &crate::parallel::SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_result, sequential_result);
    assert!(parallel.joins.load(Ordering::Relaxed) > 0);
}

#[test]
fn one_slot_executor_avoids_all_fft_joins() {
    let modulus_bits = 512;
    let ml = SsaRing::mod_limbs(modulus_bits);
    let cl = SsaRing::coeff_limbs(modulus_bits);
    let plan = FftPlan::new(modulus_bits);
    let mut left = vec![0; ml];
    let mut right = vec![0; ml];
    *some_or_panic!(left.first_mut(), "the ring must have one data limb") = 7;
    *some_or_panic!(right.first_mut(), "the ring must have one data limb") = 11;

    let executor = OneSlotCountingExecutor::default();
    let mut result = vec![0; cl];
    let mut scratch = vec![0; plan.transform_mul_scratch_for_slots(plan.parallel_slots(1))];
    // SAFETY: both operands carry exact nonzero widths, and the result and
    // scratch spans come directly from this forced plan.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            &mut result,
            &left,
            &right,
            modulus_bits,
            Some((LIMB_BITS, LIMB_BITS)),
            true,
            Some(&plan),
            &executor,
            &mut scratch,
        );
    }

    assert_eq!(
        result.first().copied(),
        Some(77),
        "the one-slot path must compute the same product"
    );
    assert_eq!(
        executor.joins.load(Ordering::Relaxed),
        0,
        "parallelism one must not enter fork-shaped control flow"
    );
}

#[test]
fn internal_parallel_policy_splits_large_fft_ranges_without_changing_results() {
    // Keep the matrix small while making each transform coefficient expensive
    // enough to cross the generated work-per-worker parallelism threshold.
    let mod_bits = 16_384;
    let transform_len = 64;
    let cl = SsaRing::coeff_limbs(mod_bits);
    let root_shift = mod_bits.wrapping_mul(2).div_euclid(transform_len);
    let mut parallel_matrix = vec![0; transform_len.wrapping_mul(cl)];
    for (index, coefficient) in parallel_matrix.chunks_exact_mut(cl).enumerate() {
        *some_or_panic!(
            coefficient.first_mut(),
            "coefficient geometry must provide a non-empty limb slice"
        ) = index + 1;
    }
    let mut sequential_matrix = parallel_matrix.clone();
    let parallel = CountingExecutor::default();
    let mut parallel_scratch = vec![0; cl.wrapping_mul(8)];
    let mut sequential_scratch = vec![0; cl];

    // SAFETY: the matrix has transform_len complete canonical coefficients, the
    // root divides the Fermat period, and parallel scratch has two disjoint slots.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut parallel_matrix,
            transform_len,
            root_shift,
            mod_bits,
            false,
            transform_len,
            &parallel,
            &mut parallel_scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut sequential_matrix,
            transform_len,
            root_shift,
            mod_bits,
            false,
            transform_len,
            &SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_matrix, sequential_matrix);
    // The inverse DIT recursion uses the same range partitioning policy. Run it
    // on both outputs to verify that its recombination order also agrees.
    // SAFETY: the forward outputs remain complete matrices and the same root is
    // valid for the inverse transform.
    unsafe {
        SsaTransform::fft_in_place_with_executor(
            &mut parallel_matrix,
            transform_len,
            root_shift,
            mod_bits,
            true,
            transform_len,
            &parallel,
            &mut parallel_scratch,
        );
        SsaTransform::fft_in_place_with_executor(
            &mut sequential_matrix,
            transform_len,
            root_shift,
            mod_bits,
            true,
            transform_len,
            &SequentialExecutor,
            &mut sequential_scratch,
        );
    }

    assert_eq!(parallel_matrix, sequential_matrix);
    assert!(parallel.joins.load(Ordering::Relaxed) >= 2);
}

/// Exercises the CRT half-widths that are not powers of two.
///
/// The top level rounds the product width up to the smallest half-width whose
/// odd part still fits the `mul_mod_bnm1` basecase, so widths such as
/// `3 * 2^k` and `5 * 2^k` now reach the Fermat transform. Every ring in the
/// recursion is a fresh geometry, so each of these widths is checked against an
/// independent schoolbook product.
#[test]
fn ssa_matches_schoolbook_at_non_power_of_two_widths() {
    // Operand widths chosen so that `a.len() + b.len()` lands on a half-width
    // with an odd multiplier of 3, 5, 7, 9, and 17 respectively, plus the
    // power-of-two widths on either side as controls.
    const WIDTHS: [(usize, usize); 10] = [
        (384, 384),
        (512, 512),
        (640, 640),
        (768, 768),
        (896, 896),
        (1_024, 1_024),
        (1_088, 1_088),
        (1_280, 1_280),
        (1_536, 1_536),
        (768, 512),
    ];

    for (len_a, len_b) in WIDTHS {
        let a: Vec<Limb> = (0..len_a)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                    .rotate_left(7)
            })
            .collect();
        let b: Vec<Limb> = (0..len_b)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x85EB_CA6B))
                    .rotate_left(11)
            })
            .collect();

        let result_len = len_a.wrapping_add(len_b);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let executor = SequentialExecutor;
        assert!(
            Ssa::try_mul_with_executor(
                &mut actual,
                &a,
                &b,
                TransformChoice::FORCED,
                None,
                &executor,
            ),
            "SSA declined a {len_a} x {len_b} limb product"
        );
        assert_eq!(
            actual, expected,
            "SSA disagreed with schoolbook at {len_a} x {len_b} limbs"
        );
    }
}

/// The tuner keeps this operand-bound plan and its exact executor-sized arena
/// across samples, so this exercises the infallible path independently of the
/// fallible production entry point that constructs a fresh plan per call.
#[test]
fn prepared_ssa_product_reuses_exact_parallel_scratch() {
    let len = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS).wrapping_add(1);
    let mut a: Vec<Limb> = (0..len)
        .map(|index| Limb::MAX.wrapping_sub(index.wrapping_mul(0x9E37_79B9)))
        .collect();
    let mut b: Vec<Limb> = (0..len)
        .map(|index| Limb::MAX.wrapping_sub(index.wrapping_mul(0x85EB_CA6B)))
        .collect();
    let high_bit = Limb::from(1_u8).wrapping_shl(Limb::BITS.wrapping_sub(1));
    *some_or_panic!(a.last_mut(), "prepared operand must be nonempty") |= high_bit;
    *some_or_panic!(b.last_mut(), "prepared operand must be nonempty") |= high_bit;

    let executor = CountingExecutor::default();
    let plan = some_or_panic!(
        SsaMultiplicationPlan::try_new(&a, &b, TransformChoice::FORCED, executor.parallelism()),
        "the forced prepared geometry must be valid"
    );
    let mut scratch = vec![Limb::MAX; plan.scratch_len()];
    let mut expected = vec![Limb::MIN; plan.destination_len()];
    let mut first = vec![Limb::MIN; plan.destination_len()];
    let mut second = vec![Limb::MIN; plan.destination_len()];
    Schoolbook::mul(&mut expected, &a, &b);

    // SAFETY: both destinations and the reused arena have the exact widths
    // reported by this operand-bound plan, and the same executor constructed it.
    unsafe {
        plan.run_with_scratch(&mut first, &mut scratch, &executor);
    }
    // SAFETY: the identical plan-owned spans remain valid for a second run.
    unsafe {
        plan.run_with_scratch(&mut second, &mut scratch, &executor);
    }

    assert_eq!(first, expected);
    assert_eq!(second, expected);
}

#[test]
fn supplied_ssa_scratch_is_executor_sized_and_never_replaced() {
    let len = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS).wrapping_add(1);
    let mut a = vec![Limb::MAX; len];
    let mut b = vec![Limb::MAX.wrapping_sub(1); len];
    let high_bit = Limb::from(1_u8).wrapping_shl(Limb::BITS.wrapping_sub(1));
    *some_or_panic!(a.last_mut(), "test operand must be nonempty") |= high_bit;
    *some_or_panic!(b.last_mut(), "test operand must be nonempty") |= high_bit;

    let executor = CountingExecutor::default();
    let sequential_len = Ssa::mul_scratch_len_for_parallelism(len, len, 1);
    let parallel_len = Ssa::mul_scratch_len_for_parallelism(len, len, executor.parallelism().get());
    assert!(parallel_len > sequential_len);

    let mut undersized = vec![Limb::MIN; sequential_len];
    let mut untouched = vec![Limb::MAX; len.wrapping_mul(2)];
    assert!(!Ssa::try_mul_with_executor(
        &mut untouched,
        &a,
        &b,
        TransformChoice::FORCED,
        Some(&mut undersized),
        &executor,
    ));
    assert!(untouched.iter().all(|limb| *limb == Limb::MAX));

    let mut exact = vec![Limb::MIN; parallel_len];
    let mut actual = vec![Limb::MIN; len.wrapping_mul(2)];
    let mut expected = vec![Limb::MIN; actual.len()];
    Schoolbook::mul(&mut expected, &a, &b);
    assert!(Ssa::try_mul_with_executor(
        &mut actual,
        &a,
        &b,
        TransformChoice::FORCED,
        Some(&mut exact),
        &executor,
    ));
    assert_eq!(actual, expected);

    let sequential_square_len = Ssa::sqr_scratch_len_for_parallelism(len, 1);
    let parallel_square_len =
        Ssa::sqr_scratch_len_for_parallelism(len, executor.parallelism().get());
    assert!(parallel_square_len > sequential_square_len);

    let mut undersized_square = vec![Limb::MIN; sequential_square_len];
    untouched.fill(Limb::MAX);
    assert!(!Ssa::try_sqr_with_executor(
        &mut untouched,
        &a,
        TransformChoice::FORCED,
        Some(&mut undersized_square),
        &executor,
    ));
    assert!(untouched.iter().all(|limb| *limb == Limb::MAX));

    let mut exact_square = vec![Limb::MIN; parallel_square_len];
    let mut actual_square = vec![Limb::MIN; len.wrapping_mul(2)];
    let mut expected_square = vec![Limb::MIN; actual_square.len()];
    Schoolbook::mul(&mut expected_square, &a, &a);
    assert!(Ssa::try_sqr_with_executor(
        &mut actual_square,
        &a,
        TransformChoice::FORCED,
        Some(&mut exact_square),
        &executor,
    ));
    assert_eq!(actual_square, expected_square);
}

/// The prepared square plan keeps the operand-bound geometry and reuses the
/// exact CRT workspace for every run without re-entering the fallible entry
/// point.
#[test]
fn prepared_ssa_square_reuses_exact_scratch() {
    let len = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS).wrapping_add(1);
    let a: Vec<Limb> = (0..len)
        .map(|index| Limb::MAX.wrapping_sub(index.wrapping_mul(0x9E37_79B9)))
        .collect();
    let executor = SequentialExecutor;
    let plan = some_or_panic!(
        SsaSquaringPlan::try_new(&a, TransformChoice::FORCED, 1),
        "the forced prepared square geometry must be valid"
    );
    let mut scratch = vec![Limb::MAX; plan.scratch_len()];
    let mut expected = vec![Limb::MIN; len.wrapping_mul(2)];
    let mut first = vec![Limb::MIN; expected.len()];
    let mut second = vec![Limb::MIN; expected.len()];
    Schoolbook::mul(&mut expected, &a, &a);

    // SAFETY: the destination and workspace have the exact widths reported by
    // this operand-bound plan, and the executor width is the planned width.
    unsafe {
        plan.run_with_scratch(&mut first, &mut scratch, &executor);
        plan.run_with_scratch(&mut second, &mut scratch, &executor);
    }

    assert_eq!(first, expected);
    assert_eq!(second, expected);
}

/// The squaring transform has its own orchestration: one forward transform,
/// pointwise squares, and an *in-place* inverse untwist. Nothing else in the
/// suite reaches it, because the tower only selects it above `SSA_THRESHOLD`
/// and no other test operand is that wide.
#[test]
fn ssa_square_matches_schoolbook() {
    // A power of two, a width whose half is odd-multiplied, and one either
    // side of the direct-shift threshold in the in-place untwist.
    const WIDTHS: [usize; 6] = [64, 192, 256, 384, 512, 640];

    for len in WIDTHS {
        let a: Vec<Limb> = (0..len)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0xC2B2_AE3D))
                    .rotate_left(13)
            })
            .collect();

        let result_len = len.wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &a);
        let executor = SequentialExecutor;
        assert!(
            Ssa::try_sqr_with_executor(&mut actual, &a, TransformChoice::FORCED, None, &executor,),
            "SSA declined a {len}-limb square"
        );
        assert_eq!(
            actual, expected,
            "SSA squaring disagreed with schoolbook at {len} limbs"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Cross-checks the squaring transform against the multiplication
    /// transform, which the tests above pin to schoolbook.
    #[test]
    fn prop_ssa_square_matches_ssa_product(
        a in proptest::collection::vec(any::<Limb>(), 32..=200),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut squared = vec![Limb::MAX; result_len];
        let mut multiplied = vec![Limb::MAX; result_len];
        let executor = SequentialExecutor;
        prop_assert!(Ssa::try_sqr_with_executor(
            &mut squared,
            &a,
            TransformChoice::FORCED,
            None,
            &executor,
        ));
        prop_assert!(Ssa::try_mul_with_executor(
            &mut multiplied,
            &a,
            &a,
            TransformChoice::FORCED,
            None,
            &executor,
        ));
        prop_assert_eq!(squared, multiplied);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_ssa_matches_basecase((a, b) in operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let executor = SequentialExecutor;
        prop_assert!(Ssa::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            TransformChoice::PLANNED,
            None,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ssa_transform_matches_basecase((a, b) in transform_operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let executor = SequentialExecutor;
        prop_assert!(Ssa::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            TransformChoice::PLANNED,
            None,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_fermat_shift_inverse_roundtrip(
        mod_bits_choice in prop_oneof![Just(512_usize), Just(1024_usize)],
        limbs in prop_oneof![
            proptest::collection::vec(Just(Limb::MIN), 1..=17),
            proptest::collection::vec(any::<Limb>(), 1..=17),
        ],
        shift in 0_usize..2_048,
    ) {
        let ml = ring::SsaRing::mod_limbs(mod_bits_choice);
        let cl = ring::SsaRing::coeff_limbs(mod_bits_choice);
        let mut expected = vec![0; cl];
        let copy_len = limbs.len().min(ml);
        // SAFETY: copy_len <= ml < cl, so both copied ranges are in bounds.
        unsafe {
            expected
                .get_unchecked_mut(..copy_len)
                .copy_from_slice(limbs.get_unchecked(..copy_len));
        }

        let full_period = mod_bits_choice.wrapping_mul(2);
        let reduced_shift = shift.rem_euclid(full_period);
        let inverse_shift = full_period.wrapping_sub(reduced_shift).rem_euclid(full_period);
        let mut actual = expected.clone();
        let mut out_of_place = vec![0; cl];
        let mut scratch = vec![0; cl];
        // SAFETY: actual and scratch both have exactly cl limbs; the two
        // exponents are additive inverses modulo the order 2 * mod_bits.
        unsafe {
            ring::SsaRing::shift(&mut actual, reduced_shift, mod_bits_choice, &mut scratch);
            ring::SsaRing::shift_from(
                &mut out_of_place,
                &expected,
                reduced_shift,
                mod_bits_choice,
            );
        }
        prop_assert_eq!(&out_of_place, &actual);
        // SAFETY: actual and scratch are disjoint cl-limb coefficients and
        // actual is canonical after the first shift.
        unsafe {
            ring::SsaRing::shift(&mut actual, inverse_shift, mod_bits_choice, &mut scratch);
        }
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_fermat_fft_inverse_roundtrip(
        mod_bits_choice in prop_oneof![Just(512_usize), Just(1024_usize)],
        (transform_len, values) in prop_oneof![
            (Just(2_usize), proptest::collection::vec(any::<Limb>(), 2)),
            (Just(4_usize), proptest::collection::vec(any::<Limb>(), 4)),
            (Just(8_usize), proptest::collection::vec(any::<Limb>(), 8)),
            (Just(16_usize), proptest::collection::vec(any::<Limb>(), 16)),
        ],
        zero_padded in any::<bool>(),
    ) {
        let cl = ring::SsaRing::coeff_limbs(mod_bits_choice);
        let mut matrix = vec![0; transform_len.wrapping_mul(cl)];
        let active_coefficients = if zero_padded {
            transform_len >> 1
        } else {
            transform_len
        };
        for (index, value) in values.iter().take(active_coefficients).enumerate() {
            let offset = index.wrapping_mul(cl);
            // SAFETY: index < transform_len and every coefficient has cl limbs.
            unsafe {
                *matrix.get_unchecked_mut(offset) = *value;
            }
        }
        let expected = matrix.clone();
        let mut scratch = vec![0; cl];
        let root_shift = mod_bits_choice
            .wrapping_mul(2)
            .div_euclid(transform_len);
        let active_chunks = if zero_padded {
            transform_len >> 1
        } else {
            transform_len
        };
        // SAFETY: matrix has transform_len * cl limbs, scratch has cl limbs,
        // and each selected transform_len is a power of two dividing 2n.
        unsafe {
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                active_chunks,
                &SequentialExecutor,
                &mut scratch,
            );
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                transform_len,
                &SequentialExecutor,
                &mut scratch,
            );
        }

        // Normalize the semi-normalized inverse output so fermat_shift's
        // guard=1 invariant (only 2^n has a set guard) is satisfied.
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset selects one complete coefficient and scratch has cl limbs.
            unsafe {
                ring::SsaRing::normalize(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    mod_bits_choice,
                );
            }
        }
        let transform_log = ok_or_panic!(
            usize::try_from(transform_len.trailing_zeros()),
            "a usize bit count must represent its trailing-zero count"
        );
        let inverse_scale = mod_bits_choice.wrapping_mul(2).wrapping_sub(transform_log);
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset identifies coefficient index < transform_len and
            // both the coefficient and scratch contain cl limbs.
            unsafe {
                ring::SsaRing::shift(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    inverse_scale,
                    mod_bits_choice,
                    &mut scratch,
                );
            }
        }
        prop_assert_eq!(matrix, expected);
    }

    #[test]
    fn prop_fermat_fft_2d_roundtrip(
        mod_bits_choice in prop_oneof![Just(512_usize), Just(1024_usize)],
        transform_len in prop_oneof![Just(256_usize)],
        values in proptest::collection::vec(any::<Limb>(), 256),
    ) {
        let cl = ring::SsaRing::coeff_limbs(mod_bits_choice);
        let mut matrix = vec![0; transform_len.wrapping_mul(cl)];
        for (index, value) in values.iter().take(transform_len).enumerate() {
            let offset = index.wrapping_mul(cl);
            // SAFETY: index < transform_len and every coefficient has cl limbs.
            unsafe {
                *matrix.get_unchecked_mut(offset) = *value;
            }
        }
        let expected = matrix.clone();
        let mut scratch = vec![0; cl];
        let root_shift = mod_bits_choice
            .wrapping_mul(2)
            .div_euclid(transform_len);
        // SAFETY: matrix has transform_len * cl limbs, scratch has cl limbs,
        // and transform_len is 256 (power of two >= 256).
        unsafe {
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                transform_len,
                &SequentialExecutor,
                &mut scratch,
            );
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                transform_len,
                &SequentialExecutor,
                &mut scratch,
            );
        }
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset selects one complete coefficient and scratch has cl limbs.
            unsafe {
                ring::SsaRing::normalize(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    mod_bits_choice,
                );
            }
        }
        let transform_log = ok_or_panic!(
            usize::try_from(transform_len.trailing_zeros()),
            "a usize bit count must represent its trailing-zero count"
        );
        let inverse_scale = mod_bits_choice.wrapping_mul(2).wrapping_sub(transform_log);
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset identifies coefficient index < transform_len and
            // both the coefficient and scratch contain cl limbs.
            unsafe {
                ring::SsaRing::shift(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    inverse_scale,
                    mod_bits_choice,
                    &mut scratch,
                );
            }
        }
        prop_assert_eq!(matrix, expected);
    }

    #[test]
    fn prop_fermat_fft_full_coefficients_roundtrip(
        (mod_bits_choice, transform_len, values) in prop_oneof![
            Just((8_192_usize, 8_usize)),
            Just((16_384_usize, 8_usize)),
        ]
        .prop_flat_map(|(modulus_bits, transform_length)| {
            let coefficient_count = modulus_bits.div_euclid(LIMB_BITS);
            let value_count = transform_length.wrapping_mul(coefficient_count);
            (
                Just(modulus_bits),
                Just(transform_length),
                proptest::collection::vec(any::<Limb>(), value_count),
            )
        }),
    ) {
        let ml = ring::SsaRing::mod_limbs(mod_bits_choice);
        let cl = ring::SsaRing::coeff_limbs(mod_bits_choice);
        let mut matrix = vec![0; transform_len.wrapping_mul(cl)];
        for index in 0..transform_len {
            let source_start = index.wrapping_mul(ml);
            let destination_start = index.wrapping_mul(cl);
            // SAFETY: values contains transform_len * ml limbs and matrix has
            // transform_len coefficients of cl = ml + 1 limbs. Leaving every
            // guard limb zero makes each generated coefficient canonical.
            unsafe {
                matrix
                    .get_unchecked_mut(destination_start..destination_start.wrapping_add(ml))
                    .copy_from_slice(
                        values.get_unchecked(source_start..source_start.wrapping_add(ml)),
                    );
            }
        }
        let expected = matrix.clone();
        let mut scratch = vec![0; cl];
        let root_shift = mod_bits_choice
            .wrapping_mul(2)
            .div_euclid(transform_len);
        // SAFETY: matrix contains transform_len complete coefficients, scratch
        // contains one coefficient, and transform_len divides 2*mod_bits.
        unsafe {
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                transform_len,
                &SequentialExecutor,
                &mut scratch,
            );
            transform::SsaTransform::fft_in_place_with_executor(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                transform_len,
                &SequentialExecutor,
                &mut scratch,
            );
        }

        // Normalize the semi-normalized inverse output before applying the
        // final inverse-scale shift so fermat_shift sees canonical input.
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset selects one complete coefficient and scratch has cl limbs.
            unsafe {
                ring::SsaRing::normalize(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    mod_bits_choice,
                );
            }
        }
        let transform_log = ok_or_panic!(
            usize::try_from(transform_len.trailing_zeros()),
            "a usize bit count must represent its trailing-zero count"
        );
        let inverse_scale = mod_bits_choice.wrapping_mul(2).wrapping_sub(transform_log);
        for index in 0..transform_len {
            let offset = index.wrapping_mul(cl);
            // SAFETY: offset selects one complete coefficient and scratch has cl limbs.
            unsafe {
                ring::SsaRing::shift(
                    matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    inverse_scale,
                    mod_bits_choice,
                    &mut scratch,
                );
            }
        }
        prop_assert_eq!(matrix, expected);
    }
}
