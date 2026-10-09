//! ADX fixed rows and two-row initializers against safe recurrences.

#![expect(
    unsafe_code,
    reason = "The tests detect ADX and BMI2 and provide the fixed widths encoded by each kernel"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::{logic::unsigned::math::arch::tests::oracle::Oracle, types::Limb};

use super::super::{x86_64_adx as adx, x86_64_adx_tail as tail};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn fixed_accumulation_rows_match_widened_recurrence(
        source in collection::vec(any::<Limb>(), 17..=17),
        destination in collection::vec(any::<Limb>(), 17..=17), scalar in any::<Limb>(),
    ) {
        if std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2") {
            check_rows(&source, &destination, scalar);
        }
    }

    #[test]
    fn fixed_initializers_write_the_complete_two_row_product(
        source in collection::vec(any::<Limb>(), 13..=13),
        scalars in (any::<Limb>(), any::<Limb>()), dirty in any::<Limb>(),
    ) {
        if std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2") {
            check_initializers(&source, scalars, dirty);
        }
    }
}

#[test]
fn fixed_rows_cover_extreme_scalars_and_full_carry_chains() {
    if !(std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2"))
    {
        return;
    }
    for source in [vec![0; 17], vec![Limb::MAX; 17]] {
        for destination in [vec![0; 17], vec![Limb::MAX; 17]] {
            for scalar in [0, 1, Limb::MAX] {
                check_rows(&source, &destination, scalar);
            }
        }
        for scalars in [
            (0, 0),
            (0, Limb::MAX),
            (Limb::MAX, 0),
            (Limb::MAX, Limb::MAX),
        ] {
            check_initializers(&source, scalars, 37);
        }
    }
}

fn check_rows(source_seed: &[Limb], destination_seed: &[Limb], scalar: Limb) {
    type Kernel = unsafe fn(*mut Limb, *const Limb, Limb) -> Limb;
    let kernels: &[(usize, Kernel)] = &[
        (4, adx::add_mul_4_limbs_unchecked),
        (5, adx::add_mul_5_limbs_unchecked),
        (6, adx::add_mul_6_limbs_unchecked),
        (7, adx::add_mul_7_limbs_unchecked),
        (8, adx::add_mul_8_limbs_unchecked),
        (9, adx::add_mul_9_limbs_unchecked),
        (10, adx::add_mul_10_limbs_unchecked),
        (11, adx::add_mul_11_limbs_unchecked),
        (12, adx::add_mul_12_limbs_unchecked),
        (13, adx::add_mul_13_limbs_unchecked),
        (14, tail::add_mul_14_limbs_unchecked),
        (15, tail::add_mul_15_limbs_unchecked),
        (16, tail::add_mul_16_limbs_unchecked),
        (17, tail::add_mul_17_limbs_unchecked),
    ];
    for &(len, kernel) in kernels {
        let source = source_seed.get(..len).expect("fixed source span");
        let destination = destination_seed.get(..len).expect("fixed destination span");
        let mut expected = destination.to_vec();
        let expected_carry = Oracle::add_mul(&mut expected, source, scalar);
        let end = len.checked_add(1).expect("fixed result end");
        let mut actual = vec![37; len.checked_add(2).expect("fixed guard width")];
        actual
            .get_mut(1..end)
            .expect("fixed destination")
            .copy_from_slice(destination);
        // SAFETY: the caller detected ADX and BMI2; the table encodes len
        // initialized disjoint limbs in the source and writable destination.
        let carry = unsafe { kernel(actual.as_mut_ptr().add(1), source.as_ptr(), scalar) };
        assert_eq!(
            (carry, actual.get(1..end).expect("fixed destination")),
            (expected_carry, expected.as_slice())
        );
        assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
    }
}

fn check_initializers(source_seed: &[Limb], scalars: (Limb, Limb), dirty: Limb) {
    type Kernel = unsafe fn(*mut Limb, *const Limb, Limb, Limb);
    let kernels: &[(usize, Kernel)] = &[
        (4, adx::mul_2x4_limbs_unchecked),
        (5, adx::mul_2x5_limbs_unchecked),
        (6, adx::mul_2x6_limbs_unchecked),
        (7, adx::mul_2x7_limbs_unchecked),
        (8, adx::mul_2x8_limbs_unchecked),
        (9, adx::mul_2x9_limbs_unchecked),
        (10, adx::mul_2x10_limbs_unchecked),
        (11, adx::mul_2x11_limbs_unchecked),
        (12, adx::mul_2x12_limbs_unchecked),
        (13, adx::mul_2x13_limbs_unchecked),
    ];
    for &(len, kernel) in kernels {
        let source = source_seed.get(..len).expect("fixed source span");
        let expected = Oracle::product(source, &<[Limb; 2]>::from(scalars));
        let width = len.checked_add(2).expect("fixed product width");
        let mut uninitialized = vec![MaybeUninit::<Limb>::uninit(); width];
        // SAFETY: the caller detected ADX and BMI2; the table encodes len
        // readable source limbs and the disjoint writable output has len+2.
        unsafe {
            kernel(
                uninitialized.as_mut_ptr().cast(),
                source.as_ptr(),
                scalars.0,
                scalars.1,
            );
        }
        for (actual, expected_limb) in uninitialized.iter().zip(&expected) {
            // SAFETY: each initializer writes all len+2 output limbs.
            assert_eq!(unsafe { actual.assume_init() }, *expected_limb);
        }
        let mut guarded = vec![dirty; width.checked_add(2).expect("fixed guard width")];
        // SAFETY: the initialized window has the same complete disjoint bounds,
        // and the caller detected ADX and BMI2.
        unsafe {
            kernel(
                guarded.as_mut_ptr().add(1),
                source.as_ptr(),
                scalars.0,
                scalars.1,
            );
        }
        let end = width.checked_add(1).expect("fixed result end");
        assert_eq!(guarded.get(1..end).expect("fixed product window"), expected);
        assert_eq!(
            (guarded.first(), guarded.last()),
            (Some(&dirty), Some(&dirty))
        );
    }
}
