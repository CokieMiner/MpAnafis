//! Shifted-source subtraction with an incoming binary borrow.

#![expect(
    unsafe_code,
    reason = "Tests prove source/destination spans, shift bounds, and CPU prerequisites"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, SubShiftedHighKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn shifted_subtraction_matches_binary_borrow_recurrence(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=129), shift in 1..Limb::BITS, borrow in any::<bool>()) {
        let (initial, source): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&initial, &source, shift, borrow);
    }
}

#[test]
fn borrow_crosses_shift_and_block_boundaries() {
    for len in 0..=if cfg!(miri) { 9 } else { 40 } {
        for source in row_patterns(len) {
            for initial_limb in [0, Limb::MAX] {
                for shift in [1, Limb::BITS >> 1, Limb::BITS - 1] {
                    for borrow in [false, true] {
                        check(&vec![initial_limb; len], &source, shift, borrow);
                    }
                }
            }
        }
    }
}

fn check(initial: &[Limb], source: &[Limb], shift: u32, incoming: bool) {
    let len = source.len();
    let mut shifted = source.to_vec();
    let _ = Oracle::rshift(
        &mut shifted,
        Limb::BITS.checked_sub(shift).expect("valid shift count"),
    );
    let mut expected = initial.to_vec();
    let mut borrow = incoming;
    for (output, input) in expected.iter_mut().zip(&shifted) {
        let (value, next_borrow) = output.borrowing_sub(*input, borrow);
        *output = value;
        borrow = next_borrow;
    }
    let kernels: &[Option<SubShiftedHighKernel>] = &[
        Some(ArchKernels::selected_sub_shifted_high_limbs_unchecked()),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "bmi2")
        ))]
        Some(super::fallback::sub_shifted_high_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("bmi2").then_some::<SubShiftedHighKernel>(
            super::x86_64_bmi2::sub_shifted_high_limbs_unchecked,
        ),
    ];
    for kernel in kernels.iter().flatten() {
        let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
        actual
            .get_mut(1..=len)
            .expect("destination window")
            .copy_from_slice(initial);
        // SAFETY: the destination window and source contain len disjoint
        // initialized limbs. Shift and binary borrow bounds hold; the table
        // guards BMI2 execution.
        let actual_borrow = unsafe {
            kernel(
                actual.as_mut_ptr().add(1),
                source.as_ptr(),
                len,
                shift,
                Limb::from(incoming),
            )
        };
        assert_eq!(actual_borrow, Limb::from(borrow));
        assert_eq!(actual.get(1..=len).expect("result window"), expected);
        assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
        if len == 0 {
            assert_eq!(
                // SAFETY: the empty kernel returns the binary incoming borrow
                // without pointer access; shift is in 1..Limb::BITS and the table
                // establishes the CPU prerequisites.
                unsafe {
                    kernel(
                        core::ptr::null_mut(),
                        core::ptr::null(),
                        0,
                        shift,
                        Limb::from(incoming),
                    )
                },
                Limb::from(incoming)
            );
        }
    }
}
