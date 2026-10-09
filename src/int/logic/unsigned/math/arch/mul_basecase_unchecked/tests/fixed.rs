//! Fixed complete products against an independent schoolbook recurrence.

#![expect(
    unsafe_code,
    reason = "Fixed-width tables establish input bounds, writable spans, and CPU prerequisites"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::{
    logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle},
    types::Limb,
};

use super::super::{mul_4x4_unchecked, mul_8x8_unchecked, portable};

type Kernel = unsafe fn(*mut Limb, *const Limb, *const Limb);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]
    #[test]
    fn fixed_products_initialize_and_overwrite_the_full_destination(
        left in collection::vec(any::<Limb>(), 8..=8),
        right in collection::vec(any::<Limb>(), 8..=8), dirty in any::<Limb>(),
    ) {
        check(&left, &right, dirty);
    }
}

#[test]
fn fixed_products_cover_full_carries_and_zero_rows() {
    let patterns = row_patterns(8);
    for left in &patterns {
        for right in &patterns {
            check(left, right, 37);
        }
    }
}

fn check(left_seed: &[Limb], right_seed: &[Limb], dirty: Limb) {
    let kernels: &[(usize, Option<Kernel>)] = &[
        (2, Some(portable::mul_2x2_portable_unchecked)),
        (3, Some(portable::mul_3x3_portable_unchecked)),
        #[cfg(not(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx",
            target_feature = "bmi2"
        )))]
        (4, Some(portable::mul_4x4_portable_unchecked)),
        #[cfg(not(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx",
            target_feature = "bmi2"
        )))]
        (8, Some(portable::mul_8x8_portable_unchecked)),
        (4, Some(mul_4x4_unchecked)),
        (8, Some(mul_8x8_unchecked)),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        (
            4,
            (std::arch::is_x86_feature_detected!("adx")
                && std::arch::is_x86_feature_detected!("bmi2"))
            .then_some::<Kernel>(super::super::x86_64_adx::mul_4x4_adx_unchecked),
        ),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        (
            8,
            (std::arch::is_x86_feature_detected!("adx")
                && std::arch::is_x86_feature_detected!("bmi2"))
            .then_some::<Kernel>(super::super::x86_64_adx::mul_8x8_adx_unchecked),
        ),
    ];
    for &(len, candidate) in kernels {
        let Some(kernel) = candidate else {
            continue;
        };
        let left = left_seed.get(..len).expect("fixed left span");
        let right = right_seed.get(..len).expect("fixed right span");
        let width = len.checked_mul(2).expect("fixed product width");
        let expected = Oracle::product(left, right);
        let mut uninitialized = vec![MaybeUninit::<Limb>::uninit(); width];
        // SAFETY: the table encodes both input widths and CPU prerequisites;
        // the disjoint, aligned output has exactly twice that width.
        unsafe {
            kernel(
                uninitialized.as_mut_ptr().cast(),
                left.as_ptr(),
                right.as_ptr(),
            );
        }
        for (actual, expected_limb) in uninitialized.iter().zip(&expected) {
            // SAFETY: each fixed-product kernel initializes every output limb.
            assert_eq!(unsafe { actual.assume_init() }, *expected_limb);
        }
        let mut guarded = vec![dirty; width.checked_add(2).expect("fixed guard width")];
        // SAFETY: the initialized window has the same disjoint complete-product
        // bounds, and the table establishes CPU prerequisites.
        unsafe {
            kernel(guarded.as_mut_ptr().add(1), left.as_ptr(), right.as_ptr());
        }
        let end = width.checked_add(1).expect("fixed result end");
        assert_eq!(guarded.get(1..end).expect("fixed product window"), expected);
        assert_eq!(
            (guarded.first(), guarded.last()),
            (Some(&dirty), Some(&dirty))
        );
    }
}
