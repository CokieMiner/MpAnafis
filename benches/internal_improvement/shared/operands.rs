//! Exact-limb operand generation shared by every benchmark target.
//!
//! Both bench targets are separate crates, so this file is included into each
//! of them with `#[path]` rather than imported. The generator is deterministic
//! and fills every limb, because a comparison against another library is only
//! meaningful when both arms see identical bit patterns: operands with sparse
//! or zero limbs let a carry-skipping basecase look faster than it is.

use core::ops::BitXor;

use mp_anafis::tune_api::tier::Limb;

pub fn operands(len: usize) -> (Vec<Limb>, Vec<Limb>, Vec<Limb>) {
    operands_pair(len, len)
}

pub fn operands_pair(left_len: usize, right_len: usize) -> (Vec<Limb>, Vec<Limb>, Vec<Limb>) {
    let left = operand(left_len, Limb::MAX.wrapping_sub(0x1234));
    let right = operand(right_len, Limb::MAX.wrapping_sub(0x4321));
    let result_len = left_len
        .checked_add(right_len)
        .expect("configured benchmark lengths fit in usize");
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
