//! Double-width products preserve low residues, high words, and precision.

use proptest::prelude::{Just, any, prop_assert, prop_assert_eq, prop_oneof, proptest};

use crate::{MpError, MpInt, MpUint, Precision, int::tests::support::nz};

proptest! {
    #[test]
    fn unsigned_products_match_native_word_splits(
        left in prop_oneof![Just(0_u64), Just(u64::MAX), any::<u64>()],
        right in prop_oneof![Just(0_u64), Just(u64::MAX), any::<u64>()],
        carry1 in any::<u64>(), carry2 in any::<u64>(),
        left_bits in 1_usize..=64, right_bits in 1_usize..=64,
        first_bits in 1_usize..=64, second_bits in 1_usize..=64,
    ) {
        let a = MpUint::with_precision_wrapping(left, nz(left_bits));
        let b = MpUint::with_precision_wrapping(right, nz(right_bits));
        let c = MpUint::with_precision_wrapping(carry1, nz(first_bits));
        let d = MpUint::with_precision_wrapping(carry2, nz(second_bits));
        // Four operands of at most 64 bits produce at most 128 result bits.
        let product = a.to_u128().expect("fits") * b.to_u128().expect("fits");
        let with_carry = product + c.to_u128().expect("fits");
        let with_two = with_carry + d.to_u128().expect("fits");
        let product_bits = left_bits.max(right_bits);
        let carrying_bits = product_bits.max(first_bits);
        for ((lower, upper), exact, bits) in [
            (a.widening_mul(&b), product, product_bits),
            (a.try_widening_mul(&b).expect("bounded operands"), product, product_bits),
            (a.carrying_mul(&b, &c), with_carry, carrying_bits),
            (a.try_carrying_mul(&b, &c).expect("bounded operands"), with_carry, carrying_bits),
            (a.carrying_mul_add(&b, &c, &d), with_two, carrying_bits.max(second_bits)),
        ] {
            let mask = (1_u128 << bits) - 1;
            prop_assert_eq!(lower.to_u128(), Some(exact & mask));
            prop_assert_eq!(upper.to_u128(), Some(exact >> bits));
            prop_assert_eq!(lower.precision(), Precision::Bounded(nz(bits)));
            prop_assert_eq!(upper.precision(), Precision::Bounded(nz(bits)));
        }
        let unlimited_a = MpUint::zero() + &a;
        let unlimited_b = MpUint::zero() + &b;
        let unlimited_c = MpUint::zero() + &c;
        let unlimited_d = MpUint::zero() + &d;
        for (x, y, z) in [(&unlimited_a, &b, &c), (&a, &unlimited_b, &c), (&a, &b, &unlimited_c)] {
            let (lower, upper) = x.carrying_mul(y, z);
            prop_assert_eq!(lower.to_u128(), Some(with_carry));
            prop_assert!(upper.is_zero());
            prop_assert_eq!(lower.precision(), Precision::Unlimited);
            prop_assert_eq!(upper.precision(), Precision::Unlimited);
            prop_assert_eq!(x.try_carrying_mul(y, z), Err(MpError::WidthRequired));
            let (sum_lower, sum_upper) = x.carrying_mul_add(y, z, &d);
            prop_assert_eq!(sum_lower.to_u128(), Some(with_two));
            prop_assert!(sum_upper.is_zero());
        }
        let (sum_lower, sum_upper) = a.carrying_mul_add(&b, &c, &unlimited_d);
        prop_assert_eq!(sum_lower.to_u128(), Some(with_two));
        prop_assert!(sum_upper.is_zero());
        for (x, y) in [(&unlimited_a, &b), (&a, &unlimited_b)] {
            let (lower, upper) = x.widening_mul(y);
            prop_assert_eq!(lower.to_u128(), Some(product));
            prop_assert!(upper.is_zero());
            prop_assert_eq!(x.try_widening_mul(y), Err(MpError::WidthRequired));
        }
    }

    #[test]
    fn signed_products_reconstruct_exact_values_and_resolve_precision(
        left in any::<i128>(), right in any::<i128>(), carry1 in any::<i128>(), carry2 in any::<i128>(),
        left_bits in 1_usize..=128, right_bits in 1_usize..=128,
        first_bits in 1_usize..=128, second_bits in 1_usize..=128,
    ) {
        let a = MpInt::with_precision_wrapping(left, nz(left_bits));
        let b = MpInt::with_precision_wrapping(right, nz(right_bits));
        let c = MpInt::with_precision_wrapping(carry1, nz(first_bits));
        let d = MpInt::with_precision_wrapping(carry2, nz(second_bits));
        let exact_a = MpInt::zero() + &a;
        let exact_b = MpInt::zero() + &b;
        let exact_c = MpInt::zero() + &c;
        let exact_d = MpInt::zero() + &d;
        let product = &exact_a * &exact_b;
        let with_carry = &product + &exact_c;
        let with_two = &with_carry + &exact_d;
        let product_bits = left_bits.max(right_bits);
        let carrying_bits = product_bits.max(first_bits);
        for ((lower, upper), exact, bits) in [
            (a.widening_mul(&b), &product, product_bits),
            (a.try_widening_mul(&b).expect("bounded operands"), &product, product_bits),
            (a.carrying_mul(&b, &c), &with_carry, carrying_bits),
            (a.try_carrying_mul(&b, &c).expect("bounded operands"), &with_carry, carrying_bits),
            (a.carrying_mul_add(&b, &c, &d), &with_two, carrying_bits.max(second_bits)),
        ] {
            prop_assert_eq!(&reconstruct_signed_words(&lower, &upper, bits), exact);
            prop_assert_eq!(lower.precision(), Precision::Bounded(nz(bits)));
            prop_assert_eq!(upper.precision(), Precision::Bounded(nz(bits)));
        }
        for (x, y, z) in [(&exact_a, &b, &c), (&a, &exact_b, &c), (&a, &b, &exact_c)] {
            let (lower, upper) = x.carrying_mul(y, z);
            prop_assert_eq!(&lower, &with_carry);
            prop_assert!(upper.is_zero());
            prop_assert_eq!(lower.precision(), Precision::Unlimited);
            prop_assert_eq!(upper.precision(), Precision::Unlimited);
            prop_assert_eq!(x.try_carrying_mul(y, z), Err(MpError::WidthRequired));
            let (sum_lower, sum_upper) = x.carrying_mul_add(y, z, &d);
            prop_assert_eq!(&sum_lower, &with_two);
            prop_assert!(sum_upper.is_zero());
        }
        let (sum_lower, sum_upper) = a.carrying_mul_add(&b, &c, &exact_d);
        prop_assert_eq!(&sum_lower, &with_two);
        prop_assert!(sum_upper.is_zero());
        for (x, y) in [(&exact_a, &b), (&a, &exact_b)] {
            let (lower, upper) = x.widening_mul(y);
            prop_assert_eq!(&lower, &product);
            prop_assert!(upper.is_zero());
            prop_assert_eq!(x.try_widening_mul(y), Err(MpError::WidthRequired));
        }
    }
}

#[test]
fn extreme_words_reconstruct_products_across_inline_and_heap_widths() {
    for bits in [1, 2, 8, 64, 65, 128, 256, 257] {
        let minimum = MpInt::min_for_precision(bits);
        let maximum = MpInt::max_for_precision(bits);
        let one = MpInt::with_precision_wrapping(1_i8, nz(bits));
        let minus_one = MpInt::with_precision_checked(-1_i8, nz(bits)).expect("minus one fits");
        for (a, b) in [
            (&minimum, &one),
            (&minus_one, &one),
            (&minimum, &minus_one),
            (&maximum, &maximum),
            (&minimum, &minimum),
        ] {
            let product = (MpInt::zero() + a) * (MpInt::zero() + b);
            let (lower, upper) = a.widening_mul(b);
            assert_eq!(reconstruct_signed_words(&lower, &upper, bits), product);
            assert_eq!(a.try_widening_mul(b), Ok((lower, upper)));
            let expected = product + (MpInt::zero() + &maximum) + (MpInt::zero() + &minus_one);
            let (sum_lower, sum_upper) = a.carrying_mul_add(b, &maximum, &minus_one);
            assert_eq!(
                reconstruct_signed_words(&sum_lower, &sum_upper, bits),
                expected
            );
        }
        let unsigned_maximum = MpUint::max_for_precision(bits);
        let (lower, upper) = unsigned_maximum.carrying_mul_add(
            &unsigned_maximum,
            &unsigned_maximum,
            &unsigned_maximum,
        );
        // (2^bits - 1)^2 + 2(2^bits - 1) = 2^(2bits) - 1.
        assert_eq!(lower, unsigned_maximum);
        assert_eq!(upper, unsigned_maximum);
        assert_eq!(lower.precision(), unsigned_maximum.precision());
        assert_eq!(upper.precision(), unsigned_maximum.precision());
    }
}

fn reconstruct_signed_words(lower: &MpInt, upper: &MpInt, bits: usize) -> MpInt {
    let radix = MpInt::one() << bits;
    let mut residue = MpInt::zero() + lower;
    if lower.is_negative() {
        residue += &radix;
    }
    (MpInt::zero() + upper) * radix + residue
}
