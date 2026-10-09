//! Rectangular residue products, uninitialized output, and shrinking multiplier reuse.

#![expect(
    unsafe_code,
    reason = "Disjoint operands and workspaces supply the guarded product's exact writable width and initialized sentinels"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::prelude::*;

use crate::int::logic::InternalMpUint;

use super::super::{Limb, LowProduct, MulScratch, ScratchBuffer};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]

    #[test]
    fn guarded_low_products_match_full_multiplication_and_preserve_sentinels(
        divisor in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 260 }),
        mut multiplier in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 0..=if cfg!(miri) { 8 } else { 260 },
        ),
    ) {
        multiplier.truncate(divisor.len());
        let width = divisor.len().checked_add(1).expect("bounded residue width");
        let mut storage = vec![MaybeUninit::uninit(); width.checked_add(2).expect("two guards")];
        let (low, upper) = storage.split_at_mut(1);
        let (output, high) = upper.split_at_mut(width);
        let low_guard = low.first_mut().expect("low sentinel").write(Limb::MAX);
        let high_guard = high.first_mut().expect("high sentinel").write(Limb::MAX);
        let mut padded = ScratchBuffer::acquire(0);
        let mut cross_product = ScratchBuffer::acquire(0);
        let mut scratch = MulScratch::default();
        let denominator = InternalMpUint::from_limbs_slice(&divisor);
        // The same workspace receives shorter products down to an empty multiplier.
        // Every call must overwrite its complete residue span.
        loop {
            let expected = denominator.mul(&InternalMpUint::from_limbs_slice(&multiplier));
            let mut low_digits = expected.limbs().get(..width.min(expected.limbs().len()))
                .expect("bounded product prefix").to_vec();
            low_digits.resize(width, 0);
            output.fill(MaybeUninit::uninit());
            // SAFETY: divisor is nonempty and multiplier was truncated to
            // its width. Inputs, width=divisor.len()+1 writable output limbs,
            // and all three workspaces are pairwise disjoint.
            let product = unsafe {
                LowProduct::mul_with_guard(
                    &divisor, &multiplier, output, &mut padded, &mut cross_product, &mut scratch,
                )
            };
            prop_assert_eq!(product, &low_digits);
            prop_assert_eq!((*low_guard, *high_guard), (Limb::MAX, Limb::MAX));
            if multiplier.is_empty() {
                break;
            }
            multiplier.truncate(multiplier.len().div_euclid(2));
        }
    }
}
