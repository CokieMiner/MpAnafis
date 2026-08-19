//! Property tests for exact NTT/CRT multiplication.

use alloc::{vec, vec::Vec};
use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use proptest::prelude::*;

use crate::{
    int::logic::{math::mul::Schoolbook, unsigned::math::mul::ntt::plan::MODULI},
    parallel::{DefaultExecutor, ParallelExecutor, SequentialExecutor},
};

use super::*;

fn operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 1),
            proptest::collection::vec(any::<Limb>(), 1),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 17),
            proptest::collection::vec(any::<Limb>(), 13),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 65),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 193),
        ),
    ]
}

// These helpers deliberately do not call the NTT implementation's modular
// arithmetic. They are a scalar O(n^2) oracle for the transform tests below.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "The reference product is intentionally reduced modulo the test prime"
)]
fn reference_mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    left.wrapping_mul(right) % modulus
}

fn reference_pow_mod(mut base: u64, mut exponent: u32, modulus: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = reference_mul_mod(result, base, modulus);
        }
        base = reference_mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

#[allow(
    clippy::arithmetic_side_effects,
    reason = "The reference radix is intentionally reduced modulo the test prime"
)]
fn reference_montgomery_radix(modulus: Modulus) -> u64 {
    1_u64.wrapping_shl(32) % u64::from(modulus.prime)
}

fn reference_to_montgomery(value: u32, modulus: Modulus) -> u32 {
    let encoded = reference_mul_mod(
        u64::from(value),
        reference_montgomery_radix(modulus),
        u64::from(modulus.prime),
    );
    u32::try_from(encoded).expect("Montgomery residue fits in u32")
}

fn reference_from_montgomery(value: u32, modulus: Modulus) -> u64 {
    let prime = u64::from(modulus.prime);
    let radix_inverse = reference_pow_mod(
        reference_montgomery_radix(modulus),
        modulus.prime.wrapping_sub(2),
        prime,
    );
    reference_mul_mod(u64::from(value), radix_inverse, prime)
}

#[allow(
    clippy::arithmetic_side_effects,
    reason = "The scalar DFT accumulates products modulo its fixed test prime"
)]
fn reference_dft(values: &[u32], modulus: Modulus, inverse: bool) -> Vec<u64> {
    let length = values.len();
    let modulus_u64 = u64::from(modulus.prime);
    let length_u32 = u32::try_from(length).expect("reference lengths fit in u32");
    let exponent = modulus.prime.wrapping_sub(1).div_euclid(length_u32);
    let mut root = reference_pow_mod(u64::from(modulus.primitive_root), exponent, modulus_u64);
    if inverse {
        root = reference_pow_mod(root, modulus.prime.wrapping_sub(2), modulus_u64);
    }
    let inverse_length = inverse.then(|| {
        reference_pow_mod(
            u64::try_from(length).expect("reference lengths fit in u64"),
            modulus.prime.wrapping_sub(2),
            modulus_u64,
        )
    });

    let mut result = Vec::with_capacity(length);
    for frequency in 0..length {
        let mut sum = 0_u64;
        for (index, &value) in values.iter().enumerate() {
            let power = index.wrapping_mul(frequency) % length;
            let power_u32 = u32::try_from(power).expect("reference powers fit in u32");
            let twiddle = reference_pow_mod(root, power_u32, modulus_u64);
            let term = reference_mul_mod(u64::from(value), twiddle, modulus_u64);
            sum = sum.wrapping_add(term) % modulus_u64;
        }
        let value = inverse_length.map_or(sum, |scale| reference_mul_mod(sum, scale, modulus_u64));
        result.push(value);
    }
    result
}

fn reverse_bits(mut value: usize, bit_count: u32) -> usize {
    let mut reversed = 0_usize;
    for _ in 0..bit_count {
        reversed = reversed.wrapping_shl(1) | (value & 1);
        value >>= 1;
    }
    reversed
}

#[allow(
    clippy::arithmetic_side_effects,
    reason = "Boundary vectors intentionally exercise modular wrapping values"
)]
fn boundary_vector(length: usize, modulus: Modulus) -> Vec<u32> {
    let prime = modulus.prime;
    let boundary = [
        0,
        1,
        prime.wrapping_sub(1),
        prime.wrapping_sub(2),
        prime.div_euclid(2),
        prime.div_euclid(2).wrapping_add(1),
    ];
    (0..length)
        .map(|index| {
            boundary.get(index).copied().unwrap_or_else(|| {
                let index_u32 = u32::try_from(index).expect("reference lengths fit in u32");
                prime.wrapping_sub(1).wrapping_mul(index_u32) % prime
            })
        })
        .collect()
}

fn alternating_boundary_vector(length: usize, modulus: Modulus) -> Vec<u32> {
    (0..length)
        .map(|index| {
            if index & 1 == 0 {
                modulus.prime.wrapping_sub(1)
            } else {
                0
            }
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_ntt_matches_basecase((a, b) in operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = Ntt::choose_transform_plan(a.len(), b.len()).expect("NTT plan available");
        let executor = SequentialExecutor;
        prop_assert!(Ntt::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_two_prime_digit_widths_match_basecase(
        digit_bits in 15_u32..=20,
        a in proptest::collection::vec(any::<Limb>(), 1..=65),
        b in proptest::collection::vec(any::<Limb>(), 1..=65),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = TransformPlan {
            digit_bits,
            modulus_count: 2,
        };
        let executor = SequentialExecutor;
        prop_assert!(Ntt::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_goldilocks_digit_widths_match_basecase(
        digit_bits in 15_u32..=23,
        a in proptest::collection::vec(any::<Limb>(), 1..=65),
        b in proptest::collection::vec(any::<Limb>(), 1..=65),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = TransformPlan {
            digit_bits,
            modulus_count: 1,
        };
        let executor = SequentialExecutor;
        prop_assert!(Ntt::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_sqr_matches_basecase(a in proptest::collection::vec(any::<Limb>(), 1..=129)) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &a);
        let plan = Ntt::choose_transform_plan(a.len(), a.len()).expect("NTT plan available");
        let executor = SequentialExecutor;
        prop_assert!(Ntt::try_sqr_with_executor(&mut actual, &a, plan, &executor));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_sqr_plans_match_basecase(
        digit_bits in 15_u32..=23,
        modulus_count in 1_usize..=2,
        a in proptest::collection::vec(any::<Limb>(), 1..=65),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &a);
        let plan = TransformPlan {
            digit_bits,
            modulus_count,
        };
        let executor = SequentialExecutor;
        prop_assert!(Ntt::try_sqr_with_executor(
            &mut actual,
            &a,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_parallel_matches_basecase((a, b) in operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = TransformPlan {
            digit_bits: 20,
            modulus_count: 2,
        };
        let executor = DefaultExecutor::default();
        prop_assert!(Ntt::try_mul_with_executor(
            &mut actual,
            &a,
            &b,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_sqr_parallel_matches_basecase(a in proptest::collection::vec(any::<Limb>(), 1..=65)) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &a);
        let plan = TransformPlan {
            digit_bits: 20,
            modulus_count: 2,
        };
        let executor = DefaultExecutor::default();
        prop_assert!(Ntt::try_sqr_with_executor(
            &mut actual,
            &a,
            plan,
            &executor,
        ));
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn regression_large_operands_match_schoolbook() {
    let a = vec![
        3_645_259_609,
        14_192_047_018_431_305_843,
        17_082_396_274_771_505_672,
        7_672_253_880_205_652_847,
        1_403_577_356_775_997_821,
        8_702_350_110_464_529_810,
        1_548_541_154_931_731_014,
        16_818_180_227_945_921_799,
        12_397_238_276_841_967_470,
        1_710_859_323_141_775_272,
        16_222_976_508_995_699_259,
        18_323_120_023_738_921_005,
        7_871_209_398_086_087_938,
        11_947_192_283_083_749_974,
        18_020_088_290_108_633_591,
        14_548_343_904_972_907_133,
        6_908_468_413_114_702_221,
    ];
    let b = vec![
        8_692_232_873_886_580_094,
        14_140_204_644_575_942_440,
        14_191_168_129_848_251_264,
        1_217_800_449_762_447_465,
        15_997_780_826_018_844_661,
        3_051_821_571_201_654_201,
        2_700_405_962_614_297_146,
        6_904_466_803_078_494_216,
        16_206_426_699_489_862_766,
        7_790_111_604_710_368_623,
        14_230_154_172_762_914_826,
        2_388_327_797_176_312_811,
        13_932_956_851_768_545_661,
    ];
    let result_len = a.len() + b.len();
    let mut expected = vec![0; result_len];
    let mut actual = vec![Limb::MAX; result_len];
    Schoolbook::mul(&mut expected, &a, &b);
    let plan = Ntt::choose_transform_plan(a.len(), b.len()).expect("NTT plan available");
    let executor = SequentialExecutor;
    let ok = Ntt::try_mul_with_executor(&mut actual, &a, &b, plan, &executor);
    assert!(ok, "NTT rejected the regression operand lengths");
    assert_eq!(
        actual, expected,
        "NTT result differs from schoolbook for regression operands"
    );
}

#[test]
fn scalar_forward_dif_matches_dft_in_bit_reversed_montgomery_form() {
    for modulus in MODULI {
        for length in [1_usize, 2, 4, 8, 16] {
            let bit_count = length.trailing_zeros();
            for values in [
                boundary_vector(length, modulus),
                alternating_boundary_vector(length, modulus),
            ] {
                let expected = reference_dft(&values, modulus, false);
                let mut actual = values;
                let mut twiddle_buf = vec![0_u32; length.div_euclid(2)];
                let executor = SequentialExecutor;
                Ntt::forward_transform_single_with_executor(
                    &mut actual,
                    modulus,
                    &mut twiddle_buf,
                    &executor,
                );

                // DIF emits frequency k at bit_reverse(k), and every output
                // is Montgomery encoded (x * R mod p), not the canonical x.
                for (frequency, &expected_value) in expected.iter().enumerate() {
                    let output_index = reverse_bits(frequency, bit_count);
                    let encoded = actual
                        .get(output_index)
                        .copied()
                        .expect("bit-reversed index is inside the transform");
                    let decoded = reference_from_montgomery(encoded, modulus);
                    assert_eq!(
                        decoded, expected_value,
                        "forward mismatch: p={}, n={}, frequency={}, output_index={}",
                        modulus.prime, length, frequency, output_index,
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_inverse_dit_matches_independent_scaled_dft() {
    for modulus in MODULI {
        for length in [1_usize, 2, 4, 8, 16] {
            let bit_count = length.trailing_zeros();
            for spectrum in [
                boundary_vector(length, modulus),
                alternating_boundary_vector(length, modulus),
            ] {
                let expected = reference_dft(&spectrum, modulus, true);
                let mut actual = vec![0_u32; length];
                for (frequency, &value) in spectrum.iter().enumerate() {
                    let output_index = reverse_bits(frequency, bit_count);
                    let slot = actual
                        .get_mut(output_index)
                        .expect("bit-reversed index is inside the transform");
                    *slot = reference_to_montgomery(value, modulus);
                }
                let mut twiddle_buf = vec![0_u32; length.div_euclid(2)];
                let executor = SequentialExecutor;
                Ntt::inverse_transform_with_executor(
                    &mut actual,
                    modulus,
                    &mut twiddle_buf,
                    &executor,
                );

                // The inverse includes multiplication by n^-1 and its final
                // Montgomery multiplication converts the result to canonical
                // representation, unlike the forward transform output.
                for (index, &expected_value) in expected.iter().enumerate() {
                    let canonical = actual
                        .get(index)
                        .copied()
                        .expect("inverse output index is inside the transform");
                    assert_eq!(
                        u64::from(canonical),
                        expected_value,
                        "inverse mismatch: p={}, n={}, index={}, spectrum={spectrum:?}",
                        modulus.prime,
                        length,
                        index,
                    );
                }
            }
        }
    }
}

#[derive(Default)]
struct CountingExecutor {
    joins: AtomicUsize,
}

impl ParallelExecutor for CountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).expect("constant is nonzero")
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

#[test]
fn custom_executor_drives_exact_multi_prime_result() {
    let a = vec![Limb::MAX; 80];
    let b = vec![Limb::MAX - 1; 73];
    let mut expected = vec![0; a.len() + b.len()];
    let mut actual = vec![0; a.len() + b.len()];
    Schoolbook::mul(&mut expected, &a, &b);

    let executor = CountingExecutor::default();
    let plan = TransformPlan {
        digit_bits: 31,
        modulus_count: 3,
    };
    assert!(Ntt::try_mul_with_executor(
        &mut actual,
        &a,
        &b,
        plan,
        &executor,
    ));
    assert_eq!(actual, expected);
    assert!(executor.joins.load(Ordering::Relaxed) >= 2);
}

#[test]
fn custom_executor_drives_one_prime_multiply_and_square() {
    let a = vec![Limb::MAX; 192];
    let b = vec![Limb::MAX - 3; 160];
    let mut expected_product = vec![0; a.len() + b.len()];
    let mut actual_product = vec![0; a.len() + b.len()];
    Schoolbook::mul(&mut expected_product, &a, &b);
    let mut expected_square = vec![0; a.len() * 2];
    let mut actual_square = vec![0; a.len() * 2];
    Schoolbook::mul(&mut expected_square, &a, &a);

    let executor = CountingExecutor::default();
    let plan = TransformPlan {
        digit_bits: 15,
        modulus_count: 1,
    };
    assert!(Ntt::try_mul_with_executor(
        &mut actual_product,
        &a,
        &b,
        plan,
        &executor,
    ));
    assert_eq!(actual_product, expected_product);
    assert!(Ntt::try_sqr_with_executor(
        &mut actual_square,
        &a,
        plan,
        &executor,
    ));
    assert_eq!(actual_square, expected_square);
    assert!(executor.joins.load(Ordering::Relaxed) > 0);
}

#[test]
fn custom_executor_splits_single_large_transform_stage() {
    let modulus = MODULI[0];
    let mut parallel_values: Vec<u32> = (0..4_096_u32).map(|value| value % modulus.prime).collect();
    let mut sequential_values = parallel_values.clone();
    let mut parallel_twiddles = vec![0_u32; parallel_values.len().div_euclid(2)];
    let mut sequential_twiddles = parallel_twiddles.clone();
    let executor = CountingExecutor::default();

    Ntt::forward_transform_single_with_executor(
        &mut parallel_values,
        modulus,
        &mut parallel_twiddles,
        &executor,
    );
    Ntt::forward_transform_single_with_executor(
        &mut sequential_values,
        modulus,
        &mut sequential_twiddles,
        &SequentialExecutor,
    );

    assert_eq!(parallel_values, sequential_values);
    assert!(
        executor.joins.load(Ordering::Relaxed) > 0,
        "large one-block stages must expose disjoint butterfly ranges to the executor"
    );
}

#[test]
fn invalid_and_overflowing_plan_inputs_are_rejected() {
    let mut output = [Limb::MAX; 2];
    let input = [Limb::from(3_u8)];
    let executor = SequentialExecutor;

    for plan in [
        TransformPlan {
            digit_bits: 0,
            modulus_count: 1,
        },
        TransformPlan {
            digit_bits: 32,
            modulus_count: 1,
        },
        TransformPlan {
            digit_bits: 15,
            modulus_count: 0,
        },
        TransformPlan {
            digit_bits: 15,
            modulus_count: 4,
        },
    ] {
        assert!(!plan.is_valid());
        assert!(!Ntt::try_mul_with_executor(
            &mut output,
            &input,
            &input,
            plan,
            &executor,
        ));
    }

    assert!(Ntt::estimated_transform_len(usize::MAX, usize::MAX, 1).is_none());
    assert!(!Ntt::coefficient_range_fits(usize::MAX, 128, 1));
    assert!(!Ntt::coefficient_range_fits(1, 15, 0));
    assert!(!Ntt::coefficient_range_fits(1, 15, MODULI.len() + 1));
}

#[test]
fn prepared_ntt_product_reuses_exact_workspace_for_every_modulus_family() {
    let left: Vec<Limb> = (0..65_usize)
        .map(|index| index.wrapping_mul(0x9e37_79b9).wrapping_add(1))
        .collect();
    let right: Vec<Limb> = (0..65_usize)
        .map(|index| index.wrapping_mul(0x85eb_ca6b).wrapping_add(3))
        .collect();
    let mut expected = vec![Limb::MIN; left.len().saturating_add(right.len())];
    Schoolbook::mul(&mut expected, &left, &right);

    for transform in [
        TransformPlan {
            digit_bits: 15,
            modulus_count: 1,
        },
        TransformPlan {
            digit_bits: 20,
            modulus_count: 2,
        },
        TransformPlan {
            digit_bits: 31,
            modulus_count: 3,
        },
    ] {
        let plan = NttMultiplicationPlan::try_new(&left, &right, transform)
            .expect("test geometry fits the fixed roots and CRT range");
        let mut scratch_u32 = vec![u32::MAX; plan.scratch_u32_len()];
        let mut scratch_u64 = vec![u64::MAX; plan.scratch_u64_len()];
        let mut actual = vec![Limb::MAX; plan.destination_len()];

        for _ in 0..2 {
            // SAFETY: every span has the exact width reported by this immutable
            // plan, and the sequential executor has no external state.
            unsafe {
                plan.run_with_scratch(
                    &mut actual,
                    &mut scratch_u32,
                    &mut scratch_u64,
                    &SequentialExecutor,
                );
            }
            assert_eq!(actual, expected);
            actual.fill(Limb::MAX);
        }
    }
}

#[test]
fn sixteen_bit_digit_packing_matches_little_endian_reference() {
    let limbs = [Limb::MAX, Limb::from(0x1234_u16), Limb::from(0x8001_u16)];
    let mut actual = vec![0_u32; limbs.len().saturating_mul(4).saturating_add(1)];
    // SAFETY: `actual` has four output slots per 64-bit input limb.
    let count = unsafe { Ntt::limbs_to_digits_into(&mut actual, &limbs, 16) };

    let mut expected = Vec::new();
    for &limb in &limbs {
        let mut value = limb;
        let chunks = Limb::BITS.div_euclid(16);
        for _ in 0..chunks {
            expected.push(
                (value & Limb::from(u16::MAX))
                    .try_into()
                    .expect("16-bit digit"),
            );
            value >>= 16;
        }
    }
    while expected.last() == Some(&0) {
        let _ = expected.pop();
    }

    assert_eq!(count, expected.len());
    assert_eq!(
        actual.get(..count).expect("digit count fits output"),
        expected.as_slice()
    );
}

#[test]
fn digit_unpacking_ignores_zero_high_padding_limb() {
    let mut digits = vec![u32::MAX; 316];
    *digits
        .get_mut(315)
        .expect("regression digit index is in range") = 1;
    let sentinel = Limb::MAX.wrapping_sub(Limb::from(42_u8));
    let mut storage = vec![Limb::MIN; 154];
    *storage
        .last_mut()
        .expect("destination plus guard has one final limb") = sentinel;
    let (limbs, guard) = storage.split_at_mut(153);

    // SAFETY: all nonzero represented bits fit in 153 64-bit limbs; the
    // nominal 316th 31-bit digit has only zero padding above that width.
    unsafe {
        Ntt::digits_to_limbs(limbs, &digits, 31);
    }
    assert_ne!(limbs.last(), Some(&0));
    assert_eq!(
        guard.first(),
        Some(&sentinel),
        "zero high padding must not write a limb beyond the destination"
    );
}
