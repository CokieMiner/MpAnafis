//! Deterministic full-width limbs shared by every measured engine.

use mp_ntt::Limb;

pub struct Inputs;

impl Inputs {
    pub fn operand(len: usize, mut state: u64) -> Vec<Limb> {
        let mask = u64::try_from(Limb::MAX).expect("native limbs have at most 64 bits");
        let mut words: Vec<Limb> = (0..len)
            .map(|_| {
                // SplitMix64 evaluates its permutation in the ring Z/(2^64).
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                Limb::try_from((value ^ (value >> 31)) & mask).expect("masked native limb")
            })
            .collect();
        let top = words.last_mut().expect("benchmark operands are nonempty");
        *top |= 1_usize
            .checked_shl(Limb::BITS.checked_sub(1).expect("positive width"))
            .expect("native high bit");
        words
    }
}
