//! Safe limb recurrences independent of architecture backends.

use alloc::{vec, vec::Vec};

use crate::int::types::{DoubleLimb, Limb};

pub struct Oracle;

impl Oracle {
    pub fn add(dst: &mut [Limb], src: &[Limb]) -> Limb {
        assert_eq!(dst.len(), src.len());
        let mut carry = false;
        for (output, input) in dst.iter_mut().zip(src) {
            let (value, next_carry) = output.carrying_add(*input, carry);
            *output = value;
            carry = next_carry;
        }
        Limb::from(carry)
    }

    pub fn sub(dst: &mut [Limb], src: &[Limb]) -> Limb {
        assert_eq!(dst.len(), src.len());
        let mut borrow = false;
        for (output, input) in dst.iter_mut().zip(src) {
            let (value, next_borrow) = output.borrowing_sub(*input, borrow);
            *output = value;
            borrow = next_borrow;
        }
        Limb::from(borrow)
    }

    #[expect(
        clippy::as_conversions,
        reason = "Widening Limb to DoubleLimb is exact; narrowing extracts the low base-B digit"
    )]
    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "Narrowing extracts the low base-B digit"
        )
    )]
    pub fn add_mul(dst: &mut [Limb], src: &[Limb], scalar: Limb) -> Limb {
        assert_eq!(dst.len(), src.len());
        let mut carry = 0;
        for (output, input) in dst.iter_mut().zip(src) {
            // (B-1)^2 + 2(B-1) = B^2-1 fits DoubleLimb at every pointer width.
            let value = (*input as DoubleLimb)
                .checked_mul(scalar as DoubleLimb)
                .and_then(|product| product.checked_add(*output as DoubleLimb))
                .and_then(|sum| sum.checked_add(carry as DoubleLimb))
                .expect("row sum fits DoubleLimb");
            *output = value as Limb;
            carry = Limb::try_from(value >> Limb::BITS).expect("carry fits a limb");
        }
        carry
    }

    #[expect(
        clippy::as_conversions,
        reason = "The product fits DoubleLimb; casts split its low and high base-B digits"
    )]
    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "Narrowing extracts a base-B digit"
        )
    )]
    pub fn sub_mul(dst: &mut [Limb], src: &[Limb], scalar: Limb) -> (Limb, Limb) {
        assert_eq!(dst.len(), src.len());
        let mut carry = 0;
        let mut borrow = false;
        for (output, input) in dst.iter_mut().zip(src) {
            let product = (*input as DoubleLimb)
                .checked_mul(scalar as DoubleLimb)
                .and_then(|value| value.checked_add(carry as DoubleLimb))
                .expect("product and carry fit DoubleLimb");
            carry = Limb::try_from(product >> Limb::BITS).expect("carry fits a limb");
            let (value, next_borrow) = output.borrowing_sub(product as Limb, borrow);
            *output = value;
            borrow = next_borrow;
        }
        (carry, Limb::from(borrow))
    }

    pub fn product(left: &[Limb], right: &[Limb]) -> Vec<Limb> {
        let len = left
            .len()
            .checked_add(right.len())
            .expect("bounded oracle width");
        let mut result = vec![0; len];
        for (row, scalar) in left.iter().copied().enumerate() {
            let end = row.checked_add(right.len()).expect("row fits result");
            let carry = Self::add_mul(result.get_mut(row..end).expect("row fits"), right, scalar);
            *result.get_mut(end).expect("closing limb fits") = carry;
        }
        result
    }

    pub fn add_mul_two(
        dst: &mut [Limb],
        src: &[Limb],
        low_scalar: Limb,
        high_scalar: Limb,
    ) -> (Limb, Limb) {
        assert_eq!(
            dst.len(),
            src.len().checked_add(1).expect("bounded row width")
        );
        let mut low_carry = 0;
        let mut high_carry = 0;
        for (index, source) in src.iter().copied().enumerate() {
            let next = index.checked_add(1).expect("bounded row index");
            let low_output = dst.get_mut(index).expect("low row fits");
            let (low, low_product_carry) = source.carrying_mul(low_scalar, low_carry);
            let (low_sum, low_sum_carry) = low_output.overflowing_add(low);
            *low_output = low_sum;
            low_carry = low_product_carry
                .checked_add(Limb::from(low_sum_carry))
                .expect("row carry fits a limb");
            let high_output = dst.get_mut(next).expect("shifted row fits");
            let (high, high_product_carry) = source.carrying_mul(high_scalar, high_carry);
            let (high_sum, high_sum_carry) = high_output.overflowing_add(high);
            *high_output = high_sum;
            high_carry = high_product_carry
                .checked_add(Limb::from(high_sum_carry))
                .expect("row carry fits a limb");
        }
        (low_carry, high_carry)
    }

    pub fn lshift(limbs: &mut [Limb], shift: u32) -> Limb {
        assert!((1..Limb::BITS).contains(&shift));
        let complementary_shift = Limb::BITS.checked_sub(shift).expect("valid shift count");
        let mut carry = 0;
        for output in limbs {
            let source = *output;
            *output = (source << shift) | carry;
            carry = source >> complementary_shift;
        }
        carry
    }

    pub fn rshift(limbs: &mut [Limb], shift: u32) -> Limb {
        assert!((1..Limb::BITS).contains(&shift));
        let complementary_shift = Limb::BITS.checked_sub(shift).expect("valid shift count");
        let mut carry = 0;
        for output in limbs.iter_mut().rev() {
            let source = *output;
            *output = (source >> shift) | carry;
            carry = source << complementary_shift;
        }
        carry
    }
}
