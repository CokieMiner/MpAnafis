//! Fermat negation and halving using ordinary multi-limb integer arithmetic.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Nonempty bounded data spans retain an extra limb for each ordinary integer oracle"
)]

use alloc::{vec, vec::Vec};

use super::Limb;

pub fn oracle_negation(data: &[Limb], guard: Limb) -> Vec<Limb> {
    let ml = data.len();
    // q+guard-L is positive. A one-bit guard requires at most one
    // modulus subtraction to obtain the canonical ring residue.
    let mut result = vec![0; ml + 1];
    result[0] = 1 + guard;
    result[ml] = 1;
    let mut borrow = false;
    for (index, digit) in result.iter_mut().enumerate() {
        let source = data.get(index).copied().unwrap_or(0);
        let (difference, first) = digit.overflowing_sub(source);
        let (value, second) = difference.overflowing_sub(Limb::from(borrow));
        *digit = value;
        borrow = first || second;
    }
    assert!(!borrow, "q+guard bounds the unsigned data value");
    if result[ml] != 0 && result[..ml].iter().any(|&limb| limb != 0) {
        let mut correction_borrow = false;
        for (index, digit) in result.iter_mut().enumerate() {
            let modulus_digit = Limb::from(index == 0 || index == ml);
            let (difference, first) = digit.overflowing_sub(modulus_digit);
            let (value, second) = difference.overflowing_sub(Limb::from(correction_borrow));
            *digit = value;
            correction_borrow = first || second;
        }
        assert!(!correction_borrow);
    }
    result
}

pub fn oracle_half(canonical: &[Limb]) -> Vec<Limb> {
    let ml = canonical.len() - 1;
    let mut integer = canonical.to_vec();
    // Odd x uses (x+p)/2 as an ordinary integer; 2p-1 fits the guard.
    if integer[0] & 1 != 0 {
        let mut carry = false;
        for (index, limb) in integer.iter_mut().enumerate() {
            let (sum, first) = limb.overflowing_add(Limb::from(index == 0 || index == ml));
            let (value, second) = sum.overflowing_add(Limb::from(carry));
            *limb = value;
            carry = first || second;
        }
        assert!(!carry);
    }
    let mut incoming = 0;
    for limb in integer.iter_mut().rev() {
        let next = *limb << (Limb::BITS - 1);
        *limb = (*limb >> 1) | incoming;
        incoming = next;
    }
    assert_eq!(incoming, 0, "the ordinary integer dividend is even");
    integer
}
