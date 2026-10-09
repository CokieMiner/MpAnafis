//! Reciprocal-prefix quotient estimates and implicit leading correction digits.

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, LIMB_BITS, Limb, ScratchBuffer};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]

    #[test]
    fn reciprocal_prefix_products_preserve_quotient_and_guard(
        high in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 8 } else { 140 }),
        inverse_low in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 140 }),
        guarded in any::<bool>(),
    ) {
        let mut scratch = DivScratch::default();
        let minimum_width = 1_usize.checked_add(usize::from(guarded)).expect("prefix and optional guard");
        for width in [high.len(), minimum_width] {
            let block = width;
            let mut inverse = inverse_low.clone();
            inverse.resize(block, 0);
            inverse.push(1);
            let mut prefix = high.get(..width).expect("bounded dividend prefix").to_vec();
            // H<B^width/2 and V<2B^block imply H*V<B^(width+block),
            // so the complete product's last guard is zero.
            *prefix.last_mut().expect("nonempty prefix") &= Limb::MAX >> 1;
            let product = InternalMpUint::from_limbs_slice(&prefix)
                .mul(&InternalMpUint::from_limbs_slice(&inverse));
            let product_width = width.checked_add(inverse.len()).expect("bounded full product");
            let mut expected = product.limbs().to_vec();
            expected.resize(product_width, 0);
            let guard = Division::newton_quotient_estimate(&prefix, &inverse, &mut scratch);
            let quotient_width = width.checked_add(usize::from(!guarded)).expect("quotient and high guard");
            let quotient_start = scratch.v_padded.len().checked_sub(quotient_width).expect("initialized quotient width");
            let expected_start = block.checked_add(usize::from(guarded)).expect("reference quotient offset");
            prop_assert_eq!(
                scratch.v_padded.get(quotient_start..).expect("initialized quotient suffix"),
                expected.get(expected_start..).expect("reference quotient suffix"),
            );
            prop_assert_eq!(guard, *expected.get(block).expect("scaled estimate digit"));
            prop_assert_eq!(scratch.v_padded.last(), Some(&0));
        }
    }

    #[test]
    fn reciprocal_corrections_preserve_implicit_leading_digits(
        inverse_low in proptest::collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 8 } else { 90 }),
        mut error_low in proptest::collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 8 } else { 90 }),
        leading_one in any::<bool>(),
    ) {
        let k = inverse_low.len();
        if leading_one {
            error_low.resize(k, 0);
            error_low.push(1);
        } else {
            error_low.truncate(k);
        }
        let mut inverse = inverse_low.clone();
        inverse.push(1);
        let shift = k.checked_add(1).and_then(|width| width.checked_mul(LIMB_BITS)).expect("bounded correction shift");
        let expected = InternalMpUint::from_limbs_slice(&error_low)
            .mul(&InternalMpUint::from_limbs_slice(&inverse))
            .shr(shift);
        let capacity = error_low.len().checked_add(k).and_then(|width| width.checked_add(1)).expect("bounded correction product");
        let mut output = ScratchBuffer::acquire(capacity);
        let mut full = ScratchBuffer::acquire(0);
        let mut scratch = DivScratch::default();
        let start = Division::newton_correction_product(
            &error_low, &inverse_low, &mut output, &mut full, &mut scratch.mul_scratch,
        );
        prop_assert_eq!(
            InternalMpUint::from_limbs_slice(output.get(start..).expect("initialized correction suffix")),
            expected,
        );
    }
}
