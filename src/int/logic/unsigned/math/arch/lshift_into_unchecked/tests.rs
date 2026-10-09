//! Write-only left shifts across scalar and vector tiers.

#![expect(
    unsafe_code,
    reason = "Tests provide disjoint readable/writable spans and guard SIMD features"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, LshiftIntoKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn shifts_initialize_outputs_and_match_recurrence(source in collection::vec(any::<Limb>(), 0..=129), shift in 1..Limb::BITS) { check(&source, shift); }
}

#[test]
fn vector_and_multi_block_boundaries_cover_extreme_bits() {
    for len in [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024,
        1025,
    ] {
        if cfg!(miri) && len > 17 {
            continue;
        }
        for source in row_patterns(len) {
            for shift in [1, Limb::BITS >> 1, Limb::BITS - 1] {
                check(&source, shift);
            }
        }
    }
}

fn check(source: &[Limb], shift: u32) {
    let len = source.len();
    let mut expected = source.to_vec();
    let expected_carry = Oracle::lshift(&mut expected, shift);
    let kernels: &[Option<LshiftIntoKernel>] = &[
        Some(ArchKernels::lshift_into_unchecked),
        Some(ArchKernels::lshift_into_small_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        Some(super::x86_64::lshift_into_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx2")
            .then_some::<LshiftIntoKernel>(super::x86_64_avx2::lshift_into_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx512f")
            .then_some::<LshiftIntoKernel>(super::x86_64_avx512::lshift_into_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        let mut output = vec![MaybeUninit::<Limb>::uninit(); len];
        // SAFETY: the source and aligned write-only output cover len disjoint
        // limbs. The shift range and table establish shift and CPU prerequisites.
        let carry = unsafe { kernel(output.as_mut_ptr().cast(), source.as_ptr(), len, shift) };
        assert_eq!(carry, expected_carry);
        for (word, value) in output.iter().zip(&expected) {
            // SAFETY: the write-only contract initializes every output limb.
            assert_eq!(unsafe { word.assume_init() }, *value);
        }
        let mut guarded = vec![37; len.checked_add(2).expect("bounded guard width")];
        assert_eq!(
            // SAFETY: the len-limb window is writable and disjoint from source;
            // the same valid shift and CPU-feature proof applies.
            unsafe { kernel(guarded.as_mut_ptr().add(1), source.as_ptr(), len, shift) },
            expected_carry
        );
        assert_eq!(guarded.get(1..=len).expect("result window"), expected);
        assert_eq!((guarded.first(), guarded.last()), (Some(&37), Some(&37)));
        if len == 0 {
            assert_eq!(
                // SAFETY: empty shifts access neither pointer; the table
                // establishes the CPU prerequisites and shift is in 1..Limb::BITS.
                unsafe { kernel(core::ptr::null_mut(), core::ptr::null(), 0, shift) },
                0
            );
        }
    }
}
