//! Single-row multiply-subtract contracts and carry/borrow extremes.

#![expect(
    unsafe_code,
    reason = "Tests pass guarded disjoint spans and detect all required CPU features"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, SubMulKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn multiply_subtract_matches_widened_recurrence(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80), scalar in any::<Limb>()) {
        let (initial, source): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&initial, &source, scalar);
    }
}

#[test]
fn carry_and_borrow_extremes_cover_every_unrolled_tail() {
    for len in 0..=if cfg!(miri) { 9 } else { 40 } {
        let patterns = row_patterns(len);
        for initial in &patterns {
            for source in &patterns {
                for scalar in [0, 1, 2, Limb::MAX >> 1, Limb::MAX - 1, Limb::MAX] {
                    check(initial, source, scalar);
                }
            }
        }
    }
}

fn check(initial: &[Limb], source: &[Limb], scalar: Limb) {
    let len = source.len();
    let mut expected = initial.to_vec();
    let expected_carries = Oracle::sub_mul(&mut expected, source, scalar);
    let kernels: &[Option<SubMulKernel>] = &[
        Some(ArchKernels::sub_mul_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        Some(super::x86_64::sub_mul_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        std::arch::is_x86_feature_detected!("bmi2")
            .then_some::<SubMulKernel>(super::x86_64_bmi2::sub_mul_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        (std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2"))
            .then_some::<SubMulKernel>(super::x86_64_adx::sub_mul_limbs_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
        actual
            .get_mut(1..=len)
            .expect("destination window")
            .copy_from_slice(initial);
        // SAFETY: source and the initialized destination window cover len
        // disjoint limbs. The table guards every ADX/BMI2 feature requirement.
        let carries = unsafe { kernel(actual.as_mut_ptr().add(1), source.as_ptr(), len, scalar) };
        assert_eq!(carries, expected_carries);
        assert_eq!(actual.get(1..=len).expect("result window"), expected);
        assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
        if len == 0 {
            assert_eq!(
                // SAFETY: empty rows access neither pointer; the table establishes
                // the CPU prerequisites.
                unsafe { kernel(core::ptr::null_mut(), core::ptr::null(), 0, scalar) },
                (0, 0)
            );
        }
    }
}
