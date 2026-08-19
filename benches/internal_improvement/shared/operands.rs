//! Exact-limb operand generation shared by every benchmark target.
//!
//! Both bench targets are separate crates, so this file is included into each
//! of them with `#[path]` rather than imported. The generator is deterministic
//! and fills every limb, because a comparison against another library is only
//! meaningful when both arms see identical bit patterns: operands with sparse
//! or zero limbs let a carry-skipping basecase look faster than it is.

#![allow(
    unsafe_code,
    reason = "the internal comparison benchmark uses GMP only for untimed reference products"
)]

use core::{fmt::Debug, ops::BitXor};

#[cfg(feature = "_internal-tune")]
use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
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

/// Checks one untimed result against an independent result buffer.
///
/// Keeping this assertion beside the deterministic operand generator makes it
/// harder for a raw benchmark arm to omit validation when a new backend is
/// added. The assertion is called before timing and never from a hot closure.
pub fn assert_same_output<T: Debug + Eq>(expected: &[T], actual: &[T], label: &str) {
    assert_eq!(
        actual, expected,
        "{label} output differs from the reference"
    );
}

/// Runs one untimed operation to warm lazy dispatch and worker state.
///
/// This is deliberately one call: benchmark arms must not hide a substantial
/// warm-up loop in setup, and the operation's own destination/output handling
/// remains the exact same work as the timed closure.
pub fn warm_up_once(operation: impl FnOnce()) {
    operation();
}

/// Warms one candidate call and checks its result before timing begins.
///
/// The probe destination is separate from the timed destination, so this is
/// also safe for backends whose first call initializes dispatch state.
pub fn validate_and_warm_product(
    expected: &[Limb],
    label: &str,
    mut operation: impl FnMut(&mut [Limb]),
) {
    let mut probe = vec![Limb::MIN; expected.len()];
    warm_up_once(|| operation(&mut probe));
    assert_same_output(expected, &probe, label);
}

/// Computes an equal-width GMP reference outside a timed benchmark closure.
#[cfg(feature = "_internal-tune")]
pub fn gmp_equal_reference(left: &[Limb], right: &[Limb]) -> Vec<Limb> {
    assert_eq!(left.len(), right.len(), "equal-width operands differ");
    let result_len = left
        .len()
        .checked_mul(2)
        .expect("configured benchmark lengths fit in usize");
    let mut expected = vec![Limb::MIN; result_len];
    let count = size_t::try_from(left.len()).expect("width fits a GMP size");
    // SAFETY: the output and both inputs are independent, initialized spans of
    // exactly `count` limbs, and the output holds the complete product.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr().cast::<limb_t>(),
            left.as_ptr().cast::<limb_t>(),
            right.as_ptr().cast::<limb_t>(),
            count,
        );
    }
    expected
}

/// Computes an unequal-width GMP reference outside a timed benchmark closure.
#[cfg(feature = "_internal-tune")]
pub fn gmp_pair_reference(larger: &[Limb], smaller: &[Limb]) -> Vec<Limb> {
    assert!(
        larger.len() >= smaller.len() && !smaller.is_empty(),
        "GMP reference requires larger >= smaller >= 1"
    );
    let result_len = larger
        .len()
        .checked_add(smaller.len())
        .expect("configured benchmark lengths fit in usize");
    let mut expected = vec![Limb::MIN; result_len];
    let larger_count = size_t::try_from(larger.len()).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller.len()).expect("width fits a GMP size");
    // SAFETY: the output and both inputs are independent, initialized spans of
    // their exact counts, and the output holds the complete product.
    unsafe {
        let _high_limb = gmp::mpn_mul(
            expected.as_mut_ptr().cast::<limb_t>(),
            larger.as_ptr().cast::<limb_t>(),
            larger_count,
            smaller.as_ptr().cast::<limb_t>(),
            smaller_count,
        );
    }
    expected
}
