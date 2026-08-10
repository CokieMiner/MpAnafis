//! Property tests for recursive Schönhage-Strassen multiplication.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{Schoolbook, ssa::SsaPointwise};

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

fn transform_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    let base_limbs = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS);
    let minimum_len = base_limbs.div_euclid(2).wrapping_add(1);
    let maximum_len = minimum_len.wrapping_add(2);
    (
        proptest::collection::vec(any::<Limb>(), minimum_len..=maximum_len),
        proptest::collection::vec(any::<Limb>(), minimum_len..=maximum_len),
    )
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
        assert!(
            Ssa::try_mul(&mut actual, &a, &b, TransformChoice::FORCED, None),
            "SSA declined a {len_a} x {len_b} limb product"
        );
        assert_eq!(
            actual, expected,
            "SSA disagreed with schoolbook at {len_a} x {len_b} limbs"
        );
    }
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
        assert!(
            Ssa::try_sqr(&mut actual, &a, TransformChoice::FORCED, None),
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
        prop_assert!(Ssa::try_sqr(&mut squared, &a, TransformChoice::FORCED, None));
        prop_assert!(Ssa::try_mul(&mut multiplied, &a, &a, TransformChoice::FORCED, None));
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
        prop_assert!(Ssa::try_mul(&mut actual, &a, &b, TransformChoice::PLANNED, None));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ssa_transform_matches_basecase((a, b) in transform_operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        prop_assert!(Ssa::try_mul(&mut actual, &a, &b, TransformChoice::PLANNED, None));
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
        // SAFETY: matrix has transform_len * cl limbs, scratch has cl limbs,
        // and each selected transform_len is a power of two dividing 2n.
        unsafe {
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                zero_padded,
                &mut scratch,
                &mut [],
            );
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                false,
                &mut scratch,
                &mut [],
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
        let transform_log = usize::try_from(transform_len.trailing_zeros())
            .expect("a usize bit count always represents its trailing-zero count");
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
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                false,
                &mut scratch,
                &mut [],
            );
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                false,
                &mut scratch,
                &mut [],
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
        let transform_log = usize::try_from(transform_len.trailing_zeros())
            .expect("a usize bit count always represents its trailing-zero count");
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
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                false,
                false,
                &mut scratch,
                &mut [],
            );
            transform::SsaTransform::fft_in_place(
                &mut matrix,
                transform_len,
                root_shift,
                mod_bits_choice,
                true,
                false,
                &mut scratch,
                &mut [],
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
        let transform_log = usize::try_from(transform_len.trailing_zeros())
            .expect("a usize bit count always represents its trailing-zero count");
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
