//! Deterministic limb operands and checked full-product output sizing.
//! A fixed xorshift stream fills each operand; the top limb's high bit is set.

use core::ops::BitXor;

use mp_anafis::tune_api::Limb;

pub fn operands_pair(left_len: usize, right_len: usize) -> (Vec<Limb>, Vec<Limb>, Vec<Limb>) {
    let left = operand(left_len, Limb::MAX.wrapping_sub(0x1234));
    let right = operand(right_len, Limb::MAX.wrapping_sub(0x4321));
    let result_len = left_len
        .checked_add(right_len)
        .expect("configured benchmark lengths must fit in usize");
    let destination = vec![Limb::MIN; result_len];
    (left, right, destination)
}

pub fn operand(len: usize, mut state: Limb) -> Vec<Limb> {
    let mut limbs: Vec<Limb> = (0..len)
        .map(|index| {
            state = BitXor::bitxor(state, state.wrapping_shl(7));
            state = BitXor::bitxor(state, state.wrapping_shr(9));
            state = BitXor::bitxor(state, state.wrapping_shl(8));
            BitXor::bitxor(state, index.rotate_left(5))
        })
        .collect();
    if let Some(top) = limbs.last_mut() {
        let high_bit = Limb::from(1_u8).wrapping_shl(Limb::BITS.wrapping_sub(1));
        *top = core::ops::BitOr::bitor(*top, high_bit);
    }
    limbs
}
