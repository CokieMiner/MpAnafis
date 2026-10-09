//! Truncated prefixes, recursive boundaries, aliasing, and spare initialization.

#![expect(
    unsafe_code,
    reason = "The low-product writer initializes the active prefix; disjoint suffix sentinels are initialized before the call"
)]

use core::mem::MaybeUninit;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::KARATSUBA_THRESHOLD;

use super::super::{
    LOW_PRODUCT_FULL_THRESHOLD, LOW_PRODUCT_RECURSIVE_THRESHOLD, Limb, LimbOutput, LowProduct,
    MulScratch, Schoolbook,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 8 }))]

    #[test]
    fn low_prefixes_cover_recursive_widths_carries_aliasing_and_dirty_reuse(
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { LOW_PRODUCT_RECURSIVE_THRESHOLD + 1 } else { 800 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { LOW_PRODUCT_RECURSIVE_THRESHOLD + 1 } else { 800 }),
        random_width in 0_usize..=if cfg!(miri) { LOW_PRODUCT_RECURSIVE_THRESHOLD + 1 } else { 800 },
        dirty in any::<Limb>(),
    ) {
        let mut scratch = MulScratch::default();
        for width in [0, 1, 2, 4, 5, 15, 31, 32, 33, 50, 100, 127, 128, 255, 256, 511, 640, 800,
            KARATSUBA_THRESHOLD * 4 - 1, KARATSUBA_THRESHOLD * 4, KARATSUBA_THRESHOLD * 4 + 1,
            LOW_PRODUCT_RECURSIVE_THRESHOLD - 1, LOW_PRODUCT_RECURSIVE_THRESHOLD, LOW_PRODUCT_RECURSIVE_THRESHOLD + 1,
            LOW_PRODUCT_RECURSIVE_THRESHOLD * 2 - 1, LOW_PRODUCT_RECURSIVE_THRESHOLD * 2, LOW_PRODUCT_RECURSIVE_THRESHOLD * 2 + 1,
            LOW_PRODUCT_FULL_THRESHOLD.saturating_sub(1), LOW_PRODUCT_FULL_THRESHOLD, LOW_PRODUCT_FULL_THRESHOLD.checked_add(1).expect("full crossover fits"), random_width] {
            if width > left_words.len() || width > right_words.len() { continue; }
            let left = left_words.get(..width).expect("generated prefix fits");
            let right = right_words.get(..width).expect("generated prefix fits");
            let maximum = vec![Limb::MAX; width];
            let near_left: Vec<Limb> = left.iter().map(|word| !(*word & Limb::from(u8::MAX))).collect();
            let near_right: Vec<Limb> = right.iter().map(|word| !(*word & Limb::from(u8::MAX))).collect();
            for (a, b) in [(left, right), (right, left), (left, left), (maximum.as_slice(), maximum.as_slice()), (near_left.as_slice(), near_right.as_slice())] {
                let mut full = vec![0; width.checked_mul(2).expect("test product fits")];
                Schoolbook::mul(&mut full, a, b);
                let mut output = vec![dirty; width.checked_add(2).expect("guards fit")];
                for _ in 0..2 {
                    output.fill(dirty);
                    scratch.buf.fill(Limb::MAX);
                    let (_, after_prefix) = output.split_at_mut(1);
                    LowProduct::mul(after_prefix, a, b, width, &mut scratch);
                    prop_assert_eq!(after_prefix.get(..width).expect("active prefix fits"), full.get(..width).expect("low product fits"));
                    prop_assert_eq!(output.first(), Some(&dirty));
                    prop_assert_eq!(output.last(), Some(&dirty));
                }
            }
        }
    }
}

#[test]
fn spare_storage_initializes_exactly_the_low_prefix() {
    let mut scratch = MulScratch::default();
    for width in [
        0_usize,
        1,
        2,
        LOW_PRODUCT_RECURSIVE_THRESHOLD - 1,
        LOW_PRODUCT_RECURSIVE_THRESHOLD,
        LOW_PRODUCT_RECURSIVE_THRESHOLD + 1,
        LOW_PRODUCT_RECURSIVE_THRESHOLD * 2 - 1,
        LOW_PRODUCT_RECURSIVE_THRESHOLD * 2,
        LOW_PRODUCT_RECURSIVE_THRESHOLD * 2 + 1,
        LOW_PRODUCT_FULL_THRESHOLD.saturating_sub(1),
        LOW_PRODUCT_FULL_THRESHOLD,
        LOW_PRODUCT_FULL_THRESHOLD
            .checked_add(1)
            .expect("full crossover fits"),
        800,
    ] {
        if cfg!(miri) && width > LOW_PRODUCT_RECURSIVE_THRESHOLD + 1 {
            continue;
        }
        let mut output = vec![MaybeUninit::uninit(); width.checked_add(2).expect("guards fit")];
        let pointer = output.as_ptr();
        for seed in [0, Limb::MAX, Limb::MAX.div_euclid(3)] {
            let left = vec![seed; width];
            let right: Vec<Limb> = left.iter().map(|word| word.rotate_left(1)).collect();
            for rhs in [&left, &right] {
                let mut full = vec![0; width.checked_mul(2).expect("test product fits")];
                Schoolbook::mul(&mut full, &left, rhs);
                output.fill(MaybeUninit::uninit());
                let (_, sentinels) = output.split_at_mut(width);
                sentinels.copy_from_slice(&[MaybeUninit::new(17), MaybeUninit::new(29)]);
                scratch.buf.fill(Limb::MAX);
                LowProduct::mul(&mut output, &left, rhs, width, &mut scratch);
                let (written, suffix) = output.split_at(width);
                // SAFETY: the writer initializes width prefix limbs; the
                // disjoint suffix retains its two initialized sentinel limbs.
                let (low, guards) = unsafe {
                    (
                        LimbOutput::assume_init(written),
                        LimbOutput::assume_init(suffix),
                    )
                };
                assert_eq!(low, full.get(..width).expect("low product fits"));
                assert_eq!(guards, &[17, 29]);
                assert_eq!(output.as_ptr(), pointer);
            }
        }
    }
}
