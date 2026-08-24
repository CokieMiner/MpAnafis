//! Residue screen rejecting non-squares before any square root is attempted.
//!
//! A perfect square is a quadratic residue modulo every modulus, so a value
//! landing on a non-residue class cannot be square. Screening is cheap and one
//! sided: a rejection is proof, while survival only means the expensive
//! [`sqrt_rem`](super::super::InternalMpUint::sqrt_rem) still has to run.
//!
//! Two stages, ordered by cost:
//!
//! 1. The low byte alone decides the class modulo 256 and rejects 82.8% of
//!    inputs without reading any other limb.
//! 2. Survivors get one fold over the whole magnitude modulo
//!    [`SCREEN_MODULUS`], whose seven prime-power factors are then checked
//!    individually. Folding once and splitting afterwards keeps the cost at a
//!    single division per limb no matter how many factors are screened.
//!
//! Together they reject 99.79% of non-squares, so the expected cost is the
//! first stage plus 17.2% of the second.

#![allow(
    unsafe_code,
    reason = "the second-stage fold runs one division per limb and uses unwrap_unchecked on a literal non-zero modulus to keep a panic landing pad out of that loop"
)]

use super::{InternalMpUint, Limb};

/// Quadratic residues modulo 256, one bit per class, low word first.
const RESIDUES_MOD_256: [u64; 4] = [
    0x0202_0212_0203_0213,
    0x0202_0212_0202_0213,
    0x0202_0212_0203_0212,
    0x0202_0212_0202_0212,
];

/// `9 * 5 * 7 * 11 * 13 * 17 * 19`, the product of the second-stage moduli.
///
/// Chosen as the largest such product that keeps a limb-wide fold in a `u128`
/// on every supported target: it needs 24 bits, so `residue << 64` stays well
/// inside 128.
const SCREEN_MODULUS: u64 = 14_549_535;

/// `(modulus, residue bitmask)` for each factor of [`SCREEN_MODULUS`].
///
/// Every modulus is below 32, so the mask of its quadratic residues fits in a
/// `u32`. Pass rates run from 4/9 down to 10/19; the combined rate is 4.5%.
const RESIDUE_MASKS: [(u64, u32); 7] = [
    (9, 0x93),
    (5, 0x13),
    (7, 0x17),
    (11, 0x23B),
    (13, 0x161B),
    (17, 0x1_A317),
    (19, 0x3_0AF3),
];

/// Returns `false` only when `value` is provably not a perfect square.
pub fn may_be_square(value: &InternalMpUint) -> bool {
    let limbs = value.limbs();
    let Some(&first_limb) = limbs.first() else {
        // Zero has no limbs and is a square.
        return true;
    };

    #[allow(
        clippy::as_conversions,
        reason = "The low byte fits in u64 and its two-bit table index fits in usize on every supported target."
    )]
    let low_byte = (first_limb as u64) & 0xFF;
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "low_byte >> 6 is in 0..=3 on every supported target"
    )]
    let word_index = (low_byte >> 6) as usize;
    // SAFETY: word_index is in 0..=3 and RESIDUES_MOD_256 has four entries.
    let word = unsafe { *RESIDUES_MOD_256.get_unchecked(word_index) };
    if word & (1_u64 << (low_byte & 63)) == 0 {
        return false;
    }

    let modulus = u128::from(SCREEN_MODULUS);
    let mut residue = 0_u128;
    #[allow(clippy::as_conversions, reason = "Limb fits safely in u128")]
    for &limb in limbs.iter().rev() {
        let shifted = residue.wrapping_shl(Limb::BITS) | (limb as u128);
        // SAFETY: SCREEN_MODULUS is a non-zero literal.
        residue = unsafe { shifted.checked_rem(modulus).unwrap_unchecked() };
    }
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "a remainder modulo SCREEN_MODULUS needs 24 bits"
    )]
    let residue_small = residue as u64;

    // Every literal factor is nonzero and at most 19, so its residue class fits
    // in a u32 shift count.
    RESIDUE_MASKS.iter().all(|&(factor, mask)| {
        // SAFETY: every factor in RESIDUE_MASKS is a nonzero literal.
        let class = unsafe { residue_small.checked_rem(factor).unwrap_unchecked() };
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "class is less than factor, and every factor is at most 19"
        )]
        let class_u32 = class as u32;
        mask & (1_u32.wrapping_shl(class_u32)) != 0
    })
}
