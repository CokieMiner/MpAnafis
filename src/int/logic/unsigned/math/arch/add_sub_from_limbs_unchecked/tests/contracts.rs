//! Shared-source sum and difference, including the permitted destination alias.

#![expect(
    unsafe_code,
    reason = "Tests provide proven spans, permitted aliases, and detected CPU prerequisites"
)]

use core::mem::MaybeUninit;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::oracle::Oracle;

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64"
))]
use super::vector;
use super::{
    super::{super::ArchKernels, AddSubFromKernel},
    Limb,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn shared_source_matches_recurrences_and_preserves_exact_alias(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=129)) {
        let (left, right): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&left, &right);
    }
}

#[test]
fn paired_carries_cross_all_block_and_tail_boundaries() {
    for len in 0..=129 {
        let mut one = vec![0; len];
        if let Some(low) = one.first_mut() {
            *low = 1;
        }
        for (left, right) in [(Limb::MAX, 0), (0, Limb::MAX), (Limb::MAX, Limb::MAX)] {
            check(&vec![left; len], &vec![right; len]);
        }
        check(&vec![Limb::MAX; len], &one);
        check(&vec![0; len], &one);
    }
}

fn check(left: &[Limb], right: &[Limb]) {
    let len = left.len();
    let mut expected_sum = left.to_vec();
    let mut expected_difference = left.to_vec();
    let carries = (
        Oracle::add(&mut expected_sum, right),
        Oracle::sub(&mut expected_difference, right),
    );
    let kernels: &[Option<AddSubFromKernel>] = &[
        Some(ArchKernels::selected_add_sub_from_limbs_unchecked()),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("adx")
            .then_some::<AddSubFromKernel>(super::super::x86_64_adx::add_sub_from_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "adx")
        ))]
        Some(super::super::fallback::add_sub_from_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx2")
            .then_some::<AddSubFromKernel>(vector::add_sub_from_limbs_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        for aliased in [false, true] {
            let guard_len = len.checked_add(2).expect("bounded guard width");
            let mut sum = vec![37; guard_len];
            let mut difference = vec![MaybeUninit::<Limb>::uninit(); guard_len];
            *difference.first_mut().expect("low guard") = MaybeUninit::new(73);
            *difference.last_mut().expect("high guard") = MaybeUninit::new(73);
            sum.get_mut(1..=len)
                .expect("sum window")
                .copy_from_slice(left);
            if aliased {
                for (slot, &limb) in difference
                    .get_mut(1..=len)
                    .expect("source window")
                    .iter_mut()
                    .zip(right)
                {
                    *slot = MaybeUninit::new(limb);
                }
            }
            // SAFETY: guard_len = len+2 makes both offset pointers valid for len
            // writable limbs. Sum and source are initialized; the difference
            // is initialized only when it supplies the exactly aliased source.
            // Allocations are otherwise disjoint. The table guards CPU features.
            let actual = unsafe {
                let difference_ptr = difference.as_mut_ptr().cast::<Limb>().add(1);
                let source_ptr = if aliased {
                    difference_ptr.cast_const()
                } else {
                    right.as_ptr()
                };
                kernel(sum.as_mut_ptr().add(1), difference_ptr, source_ptr, len)
            };
            // SAFETY: the two guards were initialized before the call, and
            // the kernel contract initializes every intervening output limb.
            // MaybeUninit<Limb> has Limb's size and alignment; the allocation
            // remains live and is borrowed immutably after the writes finish.
            let difference_result = unsafe {
                core::slice::from_raw_parts(difference.as_ptr().cast::<Limb>(), guard_len)
            };
            assert_eq!(actual, carries);
            assert_eq!(sum.get(1..=len).expect("sum result"), expected_sum);
            assert_eq!(
                difference_result.get(1..=len).expect("difference result"),
                expected_difference
            );
            assert_eq!((sum.first(), sum.last()), (Some(&37), Some(&37)));
            assert_eq!(
                (difference_result.first(), difference_result.last()),
                (Some(&73), Some(&73))
            );
        }
        if len == 0 {
            assert_eq!(
                // SAFETY: empty kernels access none of the three pointers;
                // the table establishes the CPU prerequisites.
                unsafe {
                    kernel(
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        core::ptr::null(),
                        0,
                    )
                },
                (0, 0)
            );
        }
    }
}
