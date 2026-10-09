//! Certified high products, omitted carries, and reusable output storage.

use proptest::prelude::*;

use crate::int::logic::unsigned::math::{DivScratch, InternalMpUint, LIMB_BITS};

use super::{HighProduct, KARATSUBA_THRESHOLD, Limb, ScratchBuffer};

#[test]
fn high_product_resolves_carry_from_maximal_omitted_columns() {
    for width in [
        4,
        5,
        KARATSUBA_THRESHOLD - 1,
        KARATSUBA_THRESHOLD,
        KARATSUBA_THRESHOLD + 1,
    ] {
        let a = alloc::vec![Limb::MAX; width];
        let b = a.clone();
        let mut output = ScratchBuffer::acquire(width * 2 + 2);
        let pointer = output.as_ptr();
        let capacity = output.capacity();
        let mut carry_product = ScratchBuffer::acquire(0);
        let mut scratch = DivScratch::default();
        let expected = InternalMpUint::from_limbs_slice(&a)
            .mul(&InternalMpUint::from_limbs_slice(&b))
            .shr(width * LIMB_BITS);
        let product = HighProduct::mul(
            &a,
            &b,
            width,
            &mut output,
            &mut carry_product,
            &mut scratch.mul_scratch,
        );
        assert_eq!(InternalMpUint::from_limbs_slice(product), expected);
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output.capacity(), capacity);
        if (3..KARATSUBA_THRESHOLD).contains(&width) {
            assert_eq!(
                carry_product.len(),
                width,
                "maximal discarded columns require only their omitted triangle"
            );
            // For maximal limbs, E=sum((B-1)^2*(j+1)*B^j,j<c)
            // equals c*B^(c+1)-(c+1)*B^c+1, where c=width-2.
            let cut = width - 2;
            let omitted = InternalMpUint::from_limb(cut)
                .shl((cut + 1) * LIMB_BITS)
                .sub(&InternalMpUint::from_limb(cut + 1).shl(cut * LIMB_BITS))
                .add(&InternalMpUint::one());
            assert_eq!(InternalMpUint::from_limbs_slice(&carry_product), omitted);
        }
    }
}

#[test]
fn recursive_high_products_certify_guards_and_preserve_workspace() {
    let mut output = ScratchBuffer::acquire(8_194);
    let pointer = output.as_ptr();
    let capacity = output.capacity();
    let mut work = ScratchBuffer::acquire(0);
    let mut scratch = DivScratch::default();
    for width in [
        18_usize, 19, 36, 72, 258, 259, 313, 512, 1_024, 3_072, 3_073, 4_096,
    ] {
        if cfg!(miri) && width > 72 {
            continue;
        }
        for (left, right) in [
            (0, 0),
            (Limb::MAX, Limb::MAX),
            (Limb::MAX >> 1, Limb::MAX >> 2),
        ] {
            let a = alloc::vec![left; width];
            let b = alloc::vec![right; width];
            let shift = width
                .checked_mul(LIMB_BITS)
                .expect("bounded high-product shift");
            let expected = InternalMpUint::from_limbs_slice(&a)
                .mul(&InternalMpUint::from_limbs_slice(&b))
                .shr(shift);
            let high = HighProduct::mul(
                &a,
                &b,
                width,
                &mut output,
                &mut work,
                &mut scratch.mul_scratch,
            );
            assert_eq!(
                InternalMpUint::from_limbs_slice(high),
                expected,
                "recursive product at {width} limbs"
            );
            let total = width.checked_mul(2).expect("bounded product width");
            if width >= HighProduct::RECURSIVE_THRESHOLD {
                if left == 0 {
                    assert!(
                        output.len() < total,
                        "zero guards certify the omitted low blocks"
                    );
                } else if left == Limb::MAX && right == Limb::MAX {
                    assert_eq!(
                        output.len(),
                        total,
                        "maximal guards require exact carry reconstruction"
                    );
                }
            }
            assert_eq!(output.as_ptr(), pointer);
            assert_eq!(output.capacity(), capacity);
        }
    }
}

#[test]
fn rectangular_high_products_cover_every_retained_width() {
    let threshold = HighProduct::RECURSIVE_THRESHOLD;
    let mut output = ScratchBuffer::acquire(0);
    let mut work = ScratchBuffer::acquire(0);
    let mut scratch = DivScratch::default();
    for small in [threshold, threshold.checked_add(1).expect("adjacent width")] {
        for large in [
            small.checked_add(1).expect("adjacent rectangular width"),
            small
                .checked_mul(3)
                .and_then(|width| width.checked_add(1))
                .expect("bounded rectangular width"),
        ] {
            let a = alloc::vec![Limb::MAX >> 1; large];
            let b = alloc::vec![Limb::MAX >> 2; small];
            let complete =
                InternalMpUint::from_limbs_slice(&a).mul(&InternalMpUint::from_limbs_slice(&b));
            let total = large.checked_add(small).expect("complete product width");
            for skip in large..total {
                let shift = skip.checked_mul(LIMB_BITS).expect("retained product shift");
                let expected = complete.shr(shift);
                for (left, right) in [(&a, &b), (&b, &a)] {
                    let high = HighProduct::mul(
                        left,
                        right,
                        skip,
                        &mut output,
                        &mut work,
                        &mut scratch.mul_scratch,
                    );
                    assert_eq!(InternalMpUint::from_limbs_slice(high), expected);
                }
            }
        }
    }
}

#[test]
fn trimmed_high_products_certify_guards_and_preserve_carry_storage() {
    let mut work = ScratchBuffer::acquire(0);
    let mut scratch = DivScratch::default();
    for smaller in [KARATSUBA_THRESHOLD, KARATSUBA_THRESHOLD + 1, 36, 72] {
        let larger = smaller
            .checked_mul(3)
            .and_then(|width| width.checked_add(1))
            .expect("bounded rectangular width");
        let total = larger.checked_add(smaller).expect("full product width");
        let mut output = ScratchBuffer::acquire(total.checked_add(1).expect("high carry slot"));
        let pointer = output.as_ptr();
        let capacity = output.capacity();
        for (left_digit, right_digit) in [(0, 0), (Limb::MAX, Limb::MAX), (1, Limb::MAX >> 1)] {
            let a = alloc::vec![left_digit; larger];
            let b = alloc::vec![right_digit; smaller];
            let complete =
                InternalMpUint::from_limbs_slice(&a).mul(&InternalMpUint::from_limbs_slice(&b));
            for skip in [
                larger,
                larger
                    .checked_add(smaller.div_euclid(2))
                    .expect("middle retained width"),
                total.checked_sub(1).expect("one retained limb"),
            ] {
                let shift = skip.checked_mul(LIMB_BITS).expect("retained product shift");
                let expected = complete.shr(shift);
                for (left, right) in [(&a, &b), (&b, &a)] {
                    output.resize(
                        total.checked_add(1).expect("dirty product and guard"),
                        Limb::MAX,
                    );
                    let product = HighProduct::mul(
                        left,
                        right,
                        skip,
                        &mut output,
                        &mut work,
                        &mut scratch.mul_scratch,
                    );
                    assert_eq!(InternalMpUint::from_limbs_slice(product), expected);
                    assert_eq!(output.as_ptr(), pointer);
                    assert_eq!(output.capacity(), capacity);
                    if left_digit == 0 {
                        assert!(
                            output.len() < total,
                            "zero guards certify the shortened product"
                        );
                    } else if left_digit == Limb::MAX {
                        assert_eq!(
                            output.len(),
                            total,
                            "maximal guards require the complete product"
                        );
                    }
                    let carry = output
                        .spare_capacity_mut()
                        .first_mut()
                        .expect("the high-product consumer has one reserved carry limb")
                        .write(Limb::MAX);
                    assert_eq!(*carry, Limb::MAX);
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]

    #[test]
    fn certified_high_products_match_full_multiplication(
        a in proptest::collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 40 } else { 640 }),
        b in proptest::collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 40 } else { 640 }),
        skip_seed in 0_usize..=1_280,
    ) {
        let total = a.len().checked_add(b.len()).expect("bounded product width");
        let skip = skip_seed.min(total.checked_sub(1).expect("nonempty product"));
        let shift = skip.checked_mul(LIMB_BITS).expect("bounded product shift");
        let expected = InternalMpUint::from_limbs_slice(&a)
            .mul(&InternalMpUint::from_limbs_slice(&b))
            .shr(shift);
        let mut expected_digits = expected.limbs().to_vec();
        expected_digits.resize(total.checked_sub(skip).expect("retained product width"), 0);
        let output_width = total.checked_add(2).expect("bounded product guards");
        let mut output = ScratchBuffer::acquire(output_width);
        let pointer = output.as_ptr();
        let capacity = output.capacity();
        let mut full = ScratchBuffer::acquire(0);
        let mut scratch = DivScratch::default();
        let high = HighProduct::mul(
            &a, &b, skip, &mut output, &mut full, &mut scratch.mul_scratch,
        );
        prop_assert_eq!(high, expected_digits.as_slice());
        let swapped_high = HighProduct::mul(
            &b, &a, skip, &mut output, &mut full, &mut scratch.mul_scratch,
        );
        prop_assert_eq!(swapped_high, expected_digits.as_slice());
        prop_assert_eq!(output.as_ptr(), pointer);
        prop_assert_eq!(output.capacity(), capacity);
    }

}
