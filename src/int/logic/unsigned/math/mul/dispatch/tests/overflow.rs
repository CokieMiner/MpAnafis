//! Overflow rejection for virtual workspace widths.

use crate::int::logic::unsigned::math::mul::Multiplication;

#[test]
#[should_panic(expected = "Karatsuba local workspace overflows usize")]
fn karatsuba_product_sizing_rejects_overflow() {
    let width = usize::MAX
        .div_euclid(2)
        .checked_add(1)
        .expect("half plus one fits");
    let _ = Multiplication::karatsuba_mul_scratch_len(width, width);
}

#[test]
#[should_panic(expected = "Karatsuba local square workspace overflows usize")]
fn karatsuba_square_sizing_rejects_overflow() {
    let _ = Multiplication::karatsuba_sqr_scratch_len(usize::MAX);
}
