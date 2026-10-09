//! Quotient buffers across scalar, stack, recursive and reciprocal dispatch.

use proptest::prelude::*;

use crate::int::logic::unsigned::math::div::quotient::truncated_quotient_into;

use super::{
    BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, Limb,
    NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_THRESHOLD,
};

#[test]
#[cfg_attr(
    miri,
    ignore = "Reciprocal allocation retention requires production Newton dispatch sizes; bounded reciprocal and scratch-reuse properties run under Miri"
)]
fn division_outputs_preserve_reciprocal_and_product_allocations() {
    let width = NEWTON_RAPHSON_THRESHOLD.max(NEWTON_QUOTIENT_THRESHOLD);
    let capacity = width
        .checked_mul(4)
        .and_then(|n| n.checked_add(8))
        .expect("bounded scratch");
    let mut scratch = DivScratch::default();
    scratch.dummy_rem.reserve(capacity);
    scratch.dummy_quot.reserve(capacity);
    scratch.u_norm.reserve(capacity);
    scratch.v_padded.reserve(capacity);
    let reciprocal_pointer = scratch.dummy_rem.limbs().as_ptr();
    let reciprocal_divisor_pointer = scratch.dummy_quot.limbs().as_ptr();
    let reciprocal_numerator_pointer = scratch.u_norm.as_ptr();
    let product_pointer = scratch.v_padded.as_ptr();
    let mut quotient = InternalMpUint::zero();
    let mut remainder = InternalMpUint::zero();
    for n in [width, width.checked_add(1).expect("adjacent width"), width] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX.wrapping_sub(17); n]);
        let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX.wrapping_sub(31); n]);
        let residue = divisor.sub(&InternalMpUint::one());
        let numerator = divisor.mul(&expected).add(&residue);
        Division::div_into::<false, false>(&numerator, &divisor, &mut quotient, &mut scratch);
        assert_eq!(quotient, expected);
        assert!(
            !scratch.dummy_rem.is_zero(),
            "the reciprocal remains in reusable scratch"
        );
        assert_eq!(scratch.dummy_rem.limbs().as_ptr(), reciprocal_pointer);
        assert_eq!(scratch.dummy_rem.capacity(), capacity);
        assert_eq!(
            scratch.dummy_quot.limbs().as_ptr(),
            reciprocal_divisor_pointer
        );
        assert_eq!(scratch.dummy_quot.capacity(), capacity);
        assert_eq!(scratch.u_norm.as_ptr(), reciprocal_numerator_pointer);
        assert_eq!(scratch.u_norm.capacity(), capacity);

        Division::rem_into(&numerator, &divisor, &mut remainder, &mut scratch);
        assert_eq!(remainder, residue);
        assert!(
            !scratch.v_padded.is_empty(),
            "the block product remains in reusable scratch"
        );
        assert_eq!(scratch.v_padded.as_ptr(), product_pointer);
        assert_eq!(scratch.v_padded.capacity(), capacity);
        assert_eq!(
            scratch.dummy_quot.limbs().as_ptr(),
            reciprocal_divisor_pointer
        );
        assert_eq!(scratch.u_norm.as_ptr(), reciprocal_numerator_pointer);
    }
}

#[test]
fn truncated_quotient_retains_output_capacity_and_supplied_scratch() {
    let mut output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 256]);
    let capacity = output.capacity();
    let pointer = output.limbs().as_ptr();
    let mut scratch = DivScratch::default();
    for width in [1_usize, 4, 31, 32, 33, 47, 48, 49, 95, 96, 97, 129] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX >> 1; (width + 2) * 2]);
        let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let product = divisor.mul(&expected);
        let before_exact = output.clone();
        if !truncated_quotient_into::<true>(&product, &divisor, &mut output, &mut scratch) {
            assert_eq!(output, before_exact);
            Division::div_into::<false, true>(&product, &divisor, &mut output, &mut scratch);
        }
        assert_eq!(output, expected);
        assert_eq!(output.capacity(), capacity);
        assert_eq!(output.limbs().as_ptr(), pointer);
        for residue in [InternalMpUint::zero(), divisor.shr(1)] {
            let numerator = product.add(&residue);
            let prior = output.clone();
            if !truncated_quotient_into::<false>(&numerator, &divisor, &mut output, &mut scratch) {
                assert_eq!(output, prior);
                Division::div_into::<false, false>(&numerator, &divisor, &mut output, &mut scratch);
            }
            assert_eq!(output, expected);
            assert_eq!(output.capacity(), capacity);
            assert_eq!(output.limbs().as_ptr(), pointer);
        }
    }
    let normalization_capacity = scratch.u_norm.capacity();
    let normalization_pointer = scratch.u_norm.as_ptr();
    for value in [InternalMpUint::zero(), InternalMpUint::one()] {
        assert!(truncated_quotient_into::<false>(
            &value,
            &InternalMpUint::one(),
            &mut output,
            &mut scratch
        ));
        assert_eq!(output, value);
        assert_eq!(output.capacity(), capacity);
        assert_eq!(scratch.u_norm.capacity(), normalization_capacity);
        assert_eq!(scratch.u_norm.as_ptr(), normalization_pointer);
    }
}

#[test]
fn reusable_quotient_crosses_every_division_tier() {
    let mut scratch = DivScratch::default();
    let mut output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 80]);
    let mut reference_scratch = DivScratch::default();
    for width in [
        1,
        3,
        4,
        5,
        63,
        64,
        65,
        BURNIKEL_ZIEGLER_THRESHOLD - 1,
        BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_ZIEGLER_THRESHOLD + 1,
        NEWTON_RAPHSON_THRESHOLD - 1,
        NEWTON_RAPHSON_THRESHOLD,
        NEWTON_RAPHSON_THRESHOLD + 1,
    ] {
        if cfg!(miri) && width > 65 {
            continue;
        }
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX.div_euclid(3); width]);
        for num_width in [width, width + 1, 2 * width + 1] {
            let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; num_width]);
            let mut expected = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let _ = Division::algorithm_d::<true, true, false, false>(
                numerator.limbs(),
                divisor.limbs(),
                &mut expected,
                &mut remainder,
                &mut reference_scratch,
            );
            Division::div_into::<true, false>(&numerator, &divisor, &mut output, &mut scratch);
            assert_eq!(output, expected, "{num_width}/{width} limbs");
            assert_eq!(numerator.div(&divisor), expected);
            let mut assigned = numerator;
            assigned.div_assign(&divisor);
            assert_eq!(assigned, expected);
        }
    }
    for (num, den, expected) in [(0, 3, 0), (2, 3, 0), (3, 3, 1), (19, 1, 19)] {
        Division::div_into::<true, false>(
            &InternalMpUint::from_limb(num),
            &InternalMpUint::from_limb(den),
            &mut output,
            &mut scratch,
        );
        assert_eq!(output, InternalMpUint::from_limb(expected));
    }
}

proptest! {
    #[test]
    fn prop_reusable_quotient_matches_full_division(
        num in proptest::collection::vec(any::<Limb>(), 0..=140),
        mut den in proptest::collection::vec(any::<Limb>(), 1..=70),
    ) {
        *den.last_mut().expect("nonempty divisor") |= 1;
        let numerator = InternalMpUint::from_limbs(num);
        let divisor = InternalMpUint::from_limbs(den);
        let mut output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 150]);
        Division::div_into::<true, false>(&numerator, &divisor, &mut output, &mut DivScratch::default());
        prop_assert_eq!(output, numerator.div_rem(&divisor).0);
    }
}
