//! Interleaved multiply-add rows and their separate carries.

#![expect(
    unsafe_code,
    reason = "Tests pass a len+1 destination and detect BMI2 before direct calls"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::Limb;

type Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn paired_rows_match_interleaved_recurrence(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80), high in any::<Limb>(), scalars in (any::<Limb>(), any::<Limb>())) {
        let (mut initial, source): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        initial.push(high);
        check(&initial, &source, scalars);
    }
}

#[test]
fn paired_row_carries_cover_block_boundaries() {
    for len in 0..=if cfg!(miri) { 9 } else { 40 } {
        for source in row_patterns(len) {
            for initial_limb in [0, Limb::MAX] {
                let initial = vec![initial_limb; len.checked_add(1).expect("bounded row width")];
                for low in [0, 1, Limb::MAX] {
                    for high in [0, 1, Limb::MAX] {
                        check(&initial, &source, (low, high));
                    }
                }
            }
        }
    }
}

fn check(initial: &[Limb], source: &[Limb], scalars: (Limb, Limb)) {
    let len = source.len();
    let mut expected = initial.to_vec();
    let carries = Oracle::add_mul_two(&mut expected, source, scalars.0, scalars.1);
    let kernels: &[Option<Kernel>] = &[
        #[cfg(not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        )))]
        Some(super::kernel()),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        Some(super::x86_64::add_mul_2_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        std::arch::is_x86_feature_detected!("bmi2")
            .then_some::<Kernel>(super::x86_64_bmi2::add_mul_2_limbs_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        let end = len.checked_add(2).expect("bounded row end");
        let mut actual = vec![37; end.checked_add(1).expect("bounded guard width")];
        actual
            .get_mut(1..end)
            .expect("destination window")
            .copy_from_slice(initial);
        // SAFETY: the initialized destination has len+1 writable limbs,
        // source has len readable limbs, and their allocations are disjoint.
        // The table admits BMI2 only after detection.
        let actual_carries = unsafe {
            kernel(
                actual.as_mut_ptr().add(1),
                source.as_ptr(),
                len,
                scalars.0,
                scalars.1,
            )
        };
        assert_eq!(actual_carries, carries);
        assert_eq!(actual.get(1..end).expect("result window"), expected);
        assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
        if len == 0 {
            assert_eq!(
                // SAFETY: an empty row accesses no limb; the table establishes
                // the CPU prerequisites.
                unsafe {
                    kernel(
                        core::ptr::null_mut(),
                        core::ptr::null(),
                        0,
                        scalars.0,
                        scalars.1,
                    )
                },
                (0, 0)
            );
        }
    }
}
