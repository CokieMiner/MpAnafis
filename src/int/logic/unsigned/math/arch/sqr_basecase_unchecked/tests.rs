//! Complete-square initialization and fixed-width dispatch boundaries.

#![expect(
    unsafe_code,
    reason = "The tests provide complete disjoint square spans and guard backend CPU prerequisites"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::{
    logic::unsigned::math::arch::{
        ArchKernels,
        tests::{cases::row_patterns, oracle::Oracle},
    },
    types::Limb,
};

type Kernel = unsafe fn(*mut Limb, *const Limb, usize);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]
    #[test]
    fn squares_initialize_every_limb_and_ignore_destination_contents(
        source in collection::vec(any::<Limb>(), 0..=64), dirty in any::<Limb>(),
    ) {
        check(&source, dirty);
    }
}

#[test]
fn dense_squares_cross_fixed_width_and_doubling_boundaries() {
    for len in [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        if cfg!(miri) && len > 9 {
            continue;
        }
        for source in row_patterns(len) {
            check(&source, 37);
        }
    }
}

fn check(source: &[Limb], dirty: Limb) {
    let len = source.len();
    let width = len.checked_mul(2).expect("bounded square width");
    let expected = Oracle::product(source, source);
    let kernels: &[Option<Kernel>] = &[
        Some(ArchKernels::sqr_basecase_unchecked),
        Some(super::direct::sqr_basecase_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            any(
                all(target_feature = "adx", target_feature = "bmi2"),
                not(target_feature = "bmi2")
            )
        ))]
        (len > 8
            && std::arch::is_x86_feature_detected!("adx")
            && std::arch::is_x86_feature_detected!("bmi2"))
        .then_some::<Kernel>(super::x86_64_adx::sqr_basecase_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        (len > 8 && std::arch::is_x86_feature_detected!("bmi2"))
            .then_some::<Kernel>(super::x86_64_bmi2::sqr_basecase_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        if len == 0 {
            // SAFETY: the empty square performs no pointer accesses; the table
            // establishes any CPU features required by the backend.
            unsafe {
                kernel(core::ptr::null_mut(), core::ptr::null(), 0);
            }
            continue;
        }
        let mut uninitialized = vec![MaybeUninit::<Limb>::uninit(); width];
        // SAFETY: source has len readable limbs; the disjoint output has 2*len
        // aligned writable limbs, all initialized by the square contract.
        // The table establishes CPU prerequisites.
        unsafe {
            kernel(uninitialized.as_mut_ptr().cast(), source.as_ptr(), len);
        }
        for (actual, expected_limb) in uninitialized.iter().zip(&expected) {
            // SAFETY: every square output limb was initialized by the kernel.
            assert_eq!(unsafe { actual.assume_init() }, *expected_limb);
        }
        let mut guarded = vec![dirty; width.checked_add(2).expect("bounded guard width")];
        // SAFETY: the initialized window has the same complete disjoint bounds;
        // the table establishes CPU prerequisites.
        unsafe {
            kernel(guarded.as_mut_ptr().add(1), source.as_ptr(), len);
        }
        let end = width.checked_add(1).expect("bounded result end");
        assert_eq!(guarded.get(1..end).expect("square window"), expected);
        assert_eq!(
            (guarded.first(), guarded.last()),
            (Some(&dirty), Some(&dirty))
        );
    }
}
