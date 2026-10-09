//! In-place right shifts, vector boundaries, and shifted-out low bits.

#![expect(
    unsafe_code,
    reason = "Tests pass guarded initialized spans and detect SIMD features"
)]

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::{super::ArchKernels, Limb, RshiftKernel};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn shifts_match_recurrence_and_preserve_guards(source in collection::vec(any::<Limb>(), 0..=129), shift in 1..Limb::BITS) { check(&source, shift); }
}

#[test]
fn vector_boundaries_cover_extreme_bits() {
    for len in [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
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
    let expected_carry = Oracle::rshift(&mut expected, shift);
    let kernels: &[Option<RshiftKernel>] = &[
        Some(ArchKernels::rshift_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        Some(super::x86_64::rshift_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64"
        ))]
        std::arch::is_x86_feature_detected!("avx2")
            .then_some::<RshiftKernel>(super::x86_64_avx2::rshift_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
        actual
            .get_mut(1..=len)
            .expect("destination window")
            .copy_from_slice(source);
        // SAFETY: the window contains len initialized writable limbs and the
        // shift lies in 1..Limb::BITS. The table guards AVX2 execution.
        let carry = unsafe { kernel(actual.as_mut_ptr().add(1), len, shift) };
        assert_eq!(carry, expected_carry);
        assert_eq!(actual.get(1..=len).expect("result window"), expected);
        assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
        if len == 0 {
            // SAFETY: the empty contract returns without accessing the pointer.
            assert_eq!(unsafe { kernel(core::ptr::null_mut(), 0, shift) }, 0);
        }
    }
}
