//! Sequential-product oracles for leaf, storage, and prime-swing boundaries.

use super::super::InternalMpUint;

#[test]
fn factorial_matches_sequential_products() {
    let mut product = InternalMpUint::one();
    for n in 0..=if cfg!(miri) { 96 } else { 256 } {
        if n != 0 {
            product.mul_assign(&InternalMpUint::from_u64(u64::from(n)));
        }
        assert_eq!(InternalMpUint::factorial(n), product, "{n}!");
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Prime-swing boundaries through 5001 require native execution; leaf and inline-to-heap products run under Miri."
)]
fn prime_swings_cover_odd_even_and_prime_power_boundaries() {
    let mut product = InternalMpUint::one();
    for n in 1..=5_001 {
        product.mul_assign(&InternalMpUint::from_u64(u64::from(n)));
        if [
            257, 499, 500, 501, 729, 999, 1_000, 1_001, 2_187, 5_000, 5_001,
        ]
        .contains(&n)
        {
            assert_eq!(InternalMpUint::factorial(n), product, "{n}!");
        }
    }
}
