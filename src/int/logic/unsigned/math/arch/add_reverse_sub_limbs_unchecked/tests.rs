//! Simultaneous addition and reversed subtraction.

#![expect(
    unsafe_code,
    reason = "Tests provide initialized disjoint destinations and guard CPU-specific calls"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::oracle::Oracle;

use super::{super::ArchKernels, Limb};

type Kernel = unsafe fn(*mut Limb, *mut Limb, usize) -> (Limb, Limb);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn paired_arithmetic_matches_independent_recurrences(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=129)) {
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
        check(&vec![Limb::MAX; len], &one);
        check(&one, &vec![0; len]);
        check(&vec![Limb::MAX; len], &vec![Limb::MAX; len]);
    }
}

fn check(left: &[Limb], right: &[Limb]) {
    let len = left.len();
    let mut expected_sum = left.to_vec();
    let mut expected_difference = right.to_vec();
    let expected_carry = Oracle::add(&mut expected_sum, right);
    let expected_borrow = Oracle::sub(&mut expected_difference, left);
    let kernels: &[Option<Kernel>] = &[
        Some(ArchKernels::add_reverse_sub_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("adx")
            .then_some::<Kernel>(super::x86_64_adx::add_reverse_sub_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "adx")
        ))]
        Some(super::fallback::add_reverse_sub_limbs_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        let guard_len = len.checked_add(2).expect("bounded guard width");
        let mut sum = vec![37; guard_len];
        let mut difference = vec![73; guard_len];
        sum.get_mut(1..=len)
            .expect("sum window")
            .copy_from_slice(left);
        difference
            .get_mut(1..=len)
            .expect("difference window")
            .copy_from_slice(right);
        // SAFETY: both windows cover len initialized writable limbs and are
        // disjoint. The table admits ADX only after runtime feature detection.
        let carries =
            unsafe { kernel(sum.as_mut_ptr().add(1), difference.as_mut_ptr().add(1), len) };
        assert_eq!(carries, (expected_carry, expected_borrow));
        assert_eq!(sum.get(1..=len).expect("sum result"), expected_sum);
        assert_eq!(
            difference.get(1..=len).expect("difference result"),
            expected_difference
        );
        assert_eq!((sum.first(), sum.last()), (Some(&37), Some(&37)));
        assert_eq!(
            (difference.first(), difference.last()),
            (Some(&73), Some(&73))
        );
        if len == 0 {
            assert_eq!(
                // SAFETY: empty kernels access neither pointer; the table
                // establishes the CPU prerequisites.
                unsafe { kernel(core::ptr::null_mut(), core::ptr::null_mut(), 0) },
                (0, 0)
            );
        }
    }
}
