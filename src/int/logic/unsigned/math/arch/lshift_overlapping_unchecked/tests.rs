//! Left shifts into overlapping suffixes, with complete-buffer preservation.

#![expect(
    unsafe_code,
    reason = "Tests prove the full overlapping span and detect SIMD prerequisites"
)]

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, LshiftOverlappingKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn overlapping_shifts_match_snapshotted_source(source in collection::vec(any::<Limb>(), 0..=129), offset in 0_usize..=64, dirty in any::<Limb>(), shift in 1..Limb::BITS) {
        check(&source, offset, dirty, shift);
    }
}

#[test]
fn vector_boundaries_cover_exact_alias_and_cross_block_overlap() {
    for len in [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        if cfg!(miri) && len > 9 {
            continue;
        }
        for offset in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63] {
            if cfg!(miri) && offset > 2 {
                continue;
            }
            for source in row_patterns(len) {
                for shift in [1, Limb::BITS >> 1, Limb::BITS - 1] {
                    check(&source, offset, Limb::MAX, shift);
                }
            }
        }
    }
}

fn check(source: &[Limb], offset: usize, dirty: Limb, shift: u32) {
    let len = source.len();
    let mut shifted = source.to_vec();
    let expected_carry = Oracle::lshift(&mut shifted, shift);
    let span = len.checked_add(offset).expect("bounded overlapping span");
    let mut initial = vec![37; span.checked_add(2).expect("bounded guard width")];
    initial
        .get_mut(1..=len)
        .expect("source window")
        .copy_from_slice(source);
    let extension_start = len.checked_add(1).expect("bounded extension start");
    let end = span.checked_add(1).expect("bounded destination end");
    initial
        .get_mut(extension_start..end)
        .expect("extension fits")
        .fill(dirty);
    let start = offset.checked_add(1).expect("bounded destination start");
    let mut expected = initial.clone();
    expected
        .get_mut(start..end)
        .expect("destination window")
        .copy_from_slice(&shifted);
    let kernels: &[Option<LshiftOverlappingKernel>] = &[
        Some(ArchKernels::lshift_overlapping_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        Some(super::x86_64::lshift_overlapping_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx2")
            .then_some::<LshiftOverlappingKernel>(super::x86_64_avx2::lshift_overlapping_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx512f").then_some::<LshiftOverlappingKernel>(
            super::x86_64_avx512::lshift_overlapping_unchecked,
        ),
    ];
    for kernel in kernels.iter().flatten() {
        let mut actual = initial.clone();
        // SAFETY: the guarded region contains offset+len initialized writable
        // limbs. The generated shift is valid; the table proves CPU prerequisites.
        let carry = unsafe { kernel(actual.as_mut_ptr().add(1), len, offset, shift) };
        assert_eq!((carry, actual), (expected_carry, expected.clone()));
        if len == 0 {
            // SAFETY: len=offset=0 requires no pointer access.
            assert_eq!(unsafe { kernel(core::ptr::null_mut(), 0, 0, shift) }, 0);
        }
    }
}
