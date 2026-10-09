//! Unsigned arithmetic, bounded construction, radix conversion, and factorials.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::print_stdout,
    reason = "example binary prints results and uses assertions to verify documented behavior"
)]

use mp_anafis::{BoundedPrecision, MpUint, Precision};

fn main() {
    // Add a native integer to a value larger than u64::MAX.
    let a = MpUint::from(1_000_000_000_000_u64);
    let b = MpUint::from(2_u64).pow(80);
    let sum = &a + &b;
    println!("2^80 + 10^12:  {} bits", sum.significant_bits());

    // Wrapping reduces modulo 2^16; saturation clamps to 2^16 - 1.
    let bp = BoundedPrecision::new(16).expect("16-bit precision");
    let x = MpUint::with_precision_wrapping(100_000_u64, bp);
    assert_eq!(
        x,
        MpUint::from(34_464_u64),
        "u16 wrapping: 100_000 mod 2^16"
    );
    let y = MpUint::with_precision_saturating(100_000_u64, bp);
    assert_eq!(
        y,
        MpUint::from(65_535_u64),
        "u16 saturating: 100_000 capped at 2^16-1"
    );
    println!("u16 wrapping:   100000 -> {x}");
    println!("u16 saturating: 100000 -> {y}");

    // Parse decimal and hexadecimal strings.
    let dec = MpUint::from_str_radix("12345678901234567890", 10).expect("valid decimal");
    let hex = MpUint::from_str_radix("DEADBEEFCAFE", 16).expect("valid hex");
    println!("decimal parse:  {dec}");
    println!("hex parse:      {hex:#x}");

    // Evaluate 100! with unlimited precision.
    let fact = MpUint::factorial(100, Precision::Unlimited);
    println!("100!:           {} bits", fact.significant_bits());

    println!("[OK] quickstart passed");
}
