//! Tour of the `MpAnafis` API across precision, bitwise,
//! roots, division modes, serialization, signed arithmetic,
//! and modular arithmetic.
//!
//! Run with: `cargo run --release --example api_tour`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::many_single_char_names,
    clippy::panic,
    clippy::print_stdout,
    clippy::similar_names,
    reason = "example binary prints results and uses assertions to verify documented behavior"
)]

use mp_anafis::{BoundedPrecision, MpInt, MpUint, Precision};

fn main() {
    arithmetic_semantics();
    roots_and_serialization();
    modular_and_misc();
}

fn arithmetic_semantics() {
    let bp = BoundedPrecision::new(16).expect("16-bit precision");
    let max16 = MpUint::with_precision_checked(65_535_u64, bp).expect("u16 max");
    let one = MpUint::with_precision_wrapping(1_u64, bp);

    assert!(
        max16.checked_add(&one).is_none(),
        "checked_add detects bounded overflow"
    );
    assert_eq!(
        max16.wrapping_add(&one),
        MpUint::with_precision_wrapping(0_u64, bp),
        "wrapping_add wraps to 0",
    );
    assert_eq!(
        max16.saturating_add(&one),
        max16,
        "saturating_add clamps to u16::MAX"
    );
    println!("precision:          [OK] checked/wrapping/saturating");

    let word = MpUint::from(0x1234_5678_u64);
    let rev = word.reverse_bits(32).expect("32-bit reverse");
    assert_eq!(rev, MpUint::from(0x1E6A_2C48_u64), "reverse 32 bits");
    assert_eq!(
        rev.reverse_bits(32).expect("reverse back"),
        word,
        "reverse_bits is involutive",
    );
    println!("bitwise:            [OK] reverse_bits");

    let rot = MpUint::from(0xFF00_00FF_u64)
        .rotate_left(8, 32)
        .expect("32-bit rotate");
    assert_eq!(rot, MpUint::from(0x0000_FFFF_u64), "left-rotate 8 bits");
    println!("bitwise:            [OK] rotate_left");

    let sbp = BoundedPrecision::new(8).expect("8-bit precision");
    let neg = MpInt::with_precision_checked(-7_i64, sbp).expect("-7 fits in i8");
    let three = MpInt::from(3_i64);

    assert_eq!(
        neg.div_trunc(&three),
        MpInt::from(-2_i64),
        "div_trunc rounds toward zero"
    );
    assert_eq!(
        neg.div_floor(&three),
        MpInt::from(-3_i64),
        "div_floor rounds toward -inf"
    );
    assert_eq!(
        neg.div_ceil(&three),
        MpInt::from(-2_i64),
        "div_ceil toward +inf"
    );
    assert_eq!(neg.div_euclid(&three), MpInt::from(-3_i64), "div_euclid");
    assert_eq!(
        neg.rem_euclid(&three),
        MpInt::from(2_i64),
        "rem_euclid non-negative"
    );
    println!("signed:             [OK] floor/trunc/ceil/euclid division");

    let wrapped = MpInt::with_precision_wrapping(130_i64, sbp);
    assert_eq!(
        wrapped,
        MpInt::from(-126_i64),
        "i8 wrapping: 130 wraps to -126"
    );
    println!("signed:             [OK] wrapping overflow");
}

fn roots_and_serialization() {
    let big = MpUint::from(1_000_000_u64);
    let s = big.isqrt().expect("non-negative input");
    assert_eq!(s, MpUint::from(1_000_u64), "isqrt of 1e6");
    let (sqrt_root, sqrt_rem) = big.sqrt_rem().expect("sqrt_rem");
    assert_eq!(sqrt_root, s, "sqrt_rem root matches isqrt");
    assert!(sqrt_rem.is_zero(), "perfect square");
    println!("roots:              [OK] isqrt, sqrt_rem");

    let p = MpUint::from(1_000_000_u64)
        .next_prime()
        .expect("prime in range");
    assert!(p.is_prime(), "next_prime returns a prime");
    assert_eq!(p, MpUint::from(1_000_003_u64), "next prime after 1e6");
    println!("primes:             [OK] next_prime, is_prime");

    let orig = MpUint::from(0xDEAD_BEEF_CAFE_u64);
    assert_eq!(
        orig,
        MpUint::from_le_bytes(&orig.to_le_bytes()),
        "LE round-trip"
    );
    assert_eq!(
        orig,
        MpUint::from_be_bytes(&orig.to_be_bytes()),
        "BE round-trip"
    );
    println!("serialization:      [OK] LE/BE bytes");

    let n = MpUint::from(100_u64);
    let d = MpUint::from(7_u64);
    let (q, r) = n.div_rem(&d).expect("div_rem");
    assert_eq!(q, MpUint::from(14_u64), "quotient");
    assert_eq!(r, MpUint::from(2_u64), "remainder");
    assert_eq!(n.div_ceil(&d), MpUint::from(15_u64), "div_ceil");
    println!("division:           [OK] div_rem, div_ceil");
}

fn modular_and_misc() {
    let mod7 = MpUint::from(7_u64);
    let mx = MpUint::from(3_u64);
    let my = MpUint::from(5_u64);

    assert_eq!(
        mx.add_mod(&my, &mod7).expect("add_mod"),
        MpUint::from(1_u64),
        "3 + 5 == 1 (mod 7)",
    );
    assert_eq!(
        mx.mul_mod(&my, &mod7).expect("mul_mod"),
        MpUint::from(1_u64),
        "3 * 5 == 1 (mod 7)",
    );
    let inv = mx.invert(&mod7).expect("inverse exists");
    assert_eq!(
        mx.mul_mod(&inv, &mod7).expect("mul_mod"),
        MpUint::from(1_u64),
        "inv(3) * 3 == 1 (mod 7)",
    );
    println!("modular:            [OK] add/mul/inverse");

    assert!(
        mx.montgomery_mul(&my, &mod7).is_some(),
        "Montgomery mul accepted odd modulus",
    );
    println!("montgomery:         [OK] accepted odd modulus");

    let f = MpUint::factorial(200, Precision::Unlimited);
    assert_eq!(f.significant_bits(), 1_246, "200! has 1246 bits");
    println!("factorial:          [OK] 200! = 1246 bits");

    println!("\nAll API tour checks passed.");
}
