//! Complete two-row products and destination initialization.

#![expect(
    unsafe_code,
    reason = "Tests provide disjoint uninitialized output spans and guard BMI2 calls"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::{cases::row_patterns, oracle::Oracle};

use super::Limb;

type Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn paired_product_initializes_all_output_limbs(source in collection::vec(any::<Limb>(), 0..=80), scalars in (any::<Limb>(), any::<Limb>())) {
        check(&source, scalars);
    }
}

#[test]
fn paired_product_carries_cover_block_boundaries() {
    for len in 0..=if cfg!(miri) { 9 } else { 40 } {
        for source in row_patterns(len) {
            for low in [0, 1, Limb::MAX] {
                for high in [0, 1, Limb::MAX] {
                    check(&source, (low, high));
                }
            }
        }
    }

    #[cfg(all(not(miri), target_arch = "x86", target_pointer_width = "32"))]
    for scalar in [1, 7, Limb::MAX] {
        let source = [core::hint::black_box(5)];
        let shared_scalar = core::hint::black_box(scalar);
        let expected = Oracle::product(&source, &[shared_scalar; 2]);
        let mut actual = [MaybeUninit::<Limb>::uninit(); 3];
        // SAFETY: one initialized source limb and three aligned, disjoint
        // writable outputs satisfy the one-limb path. The shared scalar remains
        // live until the second MUL, so EAX must be an early output.
        unsafe {
            super::x86::mul_2_limbs_unchecked(
                actual.as_mut_ptr().cast(),
                source.as_ptr(),
                1,
                shared_scalar,
                shared_scalar,
            );
        }
        for (output, value) in actual.iter().zip(&expected) {
            // SAFETY: the kernel initializes all three output limbs.
            assert_eq!(unsafe { output.assume_init() }, *value);
        }
    }
}

fn check(source: &[Limb], scalars: (Limb, Limb)) {
    let len = source.len();
    let expected = Oracle::product(source, &<[Limb; 2]>::from(scalars));
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
        Some(super::x86_64::mul_2_limbs_unchecked),
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))]
        std::arch::is_x86_feature_detected!("bmi2")
            .then_some::<Kernel>(super::x86_64_bmi2::mul_2_limbs_unchecked),
    ];
    for kernel in kernels.iter().flatten() {
        if len == 0 {
            // SAFETY: empty rows return without accessing either pointer.
            unsafe {
                kernel(
                    core::ptr::null_mut(),
                    core::ptr::null(),
                    0,
                    scalars.0,
                    scalars.1,
                );
            }
            continue;
        }
        let width = len.checked_add(2).expect("bounded product width");
        let mut actual = vec![MaybeUninit::<Limb>::uninit(); width];
        // SAFETY: source has len readable limbs and the disjoint output has
        // len+2 aligned writable limbs. Every output is initialized by contract;
        // the table establishes CPU prerequisites.
        unsafe {
            kernel(
                actual.as_mut_ptr().cast(),
                source.as_ptr(),
                len,
                scalars.0,
                scalars.1,
            );
        }
        for (output, value) in actual.iter().zip(&expected) {
            // SAFETY: the complete-product contract initializes all len+2 limbs.
            assert_eq!(unsafe { output.assume_init() }, *value);
        }
        let mut poisoned = vec![37; width.checked_add(2).expect("bounded guard width")];
        // SAFETY: this initialized len+2 window is also disjoint from source.
        unsafe {
            kernel(
                poisoned.as_mut_ptr().add(1),
                source.as_ptr(),
                len,
                scalars.0,
                scalars.1,
            );
        }
        let end = width.checked_add(1).expect("bounded result end");
        assert_eq!(poisoned.get(1..end).expect("product window"), expected);
        assert_eq!((poisoned.first(), poisoned.last()), (Some(&37), Some(&37)));
    }
}
