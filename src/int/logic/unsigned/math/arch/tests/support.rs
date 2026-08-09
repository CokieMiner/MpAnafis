//! Generators and arithmetic oracles for architecture-kernel properties.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::int::types::{DoubleLimb, Limb};

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The oracle widens limbs into DoubleLimb and deliberately keeps each product's low limb"
)]
pub fn reference_add_multiply_limb(dst: &mut [Limb], src: &[Limb], scalar: Limb) -> Limb {
    let mut carry: Limb = 0;
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        let product = (*src_limb as DoubleLimb).wrapping_mul(scalar as DoubleLimb);
        let low_product = product as Limb;
        let high_product = product.wrapping_shr(Limb::BITS) as Limb;

        let (low_with_carry, carry_from_product) = low_product.overflowing_add(carry);
        let (result, carry_from_sum) = dst_limb.overflowing_add(low_with_carry);
        *dst_limb = result;
        carry = high_product
            .wrapping_add(Limb::from(carry_from_product))
            .wrapping_add(Limb::from(carry_from_sum));
    }
    carry
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "The oracle indexes length-proven row spans and splits exact DoubleLimb products into their mathematical base-B limbs"
)]
pub fn reference_add_multiply_two(
    dst: &mut [Limb],
    src: &[Limb],
    low_scalar: Limb,
    high_scalar: Limb,
) -> (Limb, Limb) {
    let mut low_carry = 0;
    let mut high_carry = 0;
    for index in 0..src.len() {
        // A limb product is at most B^2-2B+1. Adding a carry and destination
        // limb contributes at most 2B-2, so each row sum is at most B^2-1 and
        // fits exactly in DoubleLimb on every supported Limb width.
        let source = src[index] as DoubleLimb;
        let low_sum = source
            .wrapping_mul(low_scalar as DoubleLimb)
            .wrapping_add(low_carry as DoubleLimb)
            .wrapping_add(dst[index] as DoubleLimb);
        dst[index] = low_sum as Limb;
        low_carry = low_sum.wrapping_shr(Limb::BITS) as Limb;

        let high_index = index.wrapping_add(1);
        let high_sum = source
            .wrapping_mul(high_scalar as DoubleLimb)
            .wrapping_add(high_carry as DoubleLimb)
            .wrapping_add(dst[high_index] as DoubleLimb);
        dst[high_index] = high_sum as Limb;
        high_carry = high_sum.wrapping_shr(Limb::BITS) as Limb;
    }
    (low_carry, high_carry)
}

#[allow(
    clippy::indexing_slicing,
    reason = "The result has src.len()+2 limbs, which proves both row slices and closing-limb indices are in bounds"
)]
pub fn reference_multiply_two(src: &[Limb], low_scalar: Limb, high_scalar: Limb) -> Vec<Limb> {
    let len = src.len();
    let mut result = vec![0; len.wrapping_add(2)];
    if len == 0 {
        return result;
    }
    let low_carry = reference_add_multiply_limb(&mut result[..len], src, low_scalar);
    result[len] = low_carry;
    let high_end = len.wrapping_add(1);
    let high_carry = reference_add_multiply_limb(&mut result[1..high_end], src, high_scalar);
    result[high_end] = high_carry;
    result
}

pub fn limb_vec(len: core::ops::RangeInclusive<usize>) -> impl Strategy<Value = Vec<Limb>> {
    proptest::collection::vec(any::<Limb>(), len)
}

pub fn equal_length_limb_vecs(
    len: core::ops::RangeInclusive<usize>,
) -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    len.prop_flat_map(|len_value| {
        (
            limb_vec(len_value..=len_value),
            limb_vec(len_value..=len_value),
        )
    })
}

pub fn odd_limb_vec(len: core::ops::RangeInclusive<usize>) -> impl Strategy<Value = Vec<Limb>> {
    limb_vec(len)
        .prop_filter("modulus must be non-empty", |limbs| !limbs.is_empty())
        .prop_map(|mut limbs| {
            if let Some(first_limb) = limbs.first_mut() {
                *first_limb |= 1;
            }
            limbs
        })
}

pub fn equal_length_odd_limb_vecs(
    len: core::ops::RangeInclusive<usize>,
) -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>, Vec<Limb>)> {
    len.prop_flat_map(|len_value| {
        (
            odd_limb_vec(len_value..=len_value),
            odd_limb_vec(len_value..=len_value),
            odd_limb_vec(len_value..=len_value),
        )
    })
}

pub fn montgomery_inverse(modulus_low: Limb) -> Limb {
    let mut inverse = modulus_low;
    let two = Limb::from(2_u8);
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(two.wrapping_sub(modulus_low.wrapping_mul(inverse)));
    }
    inverse.wrapping_neg()
}

#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(miri)
))]
pub fn exact_limb_vec(len: usize) -> impl Strategy<Value = Vec<Limb>> {
    proptest::collection::vec(any::<Limb>(), len)
}
