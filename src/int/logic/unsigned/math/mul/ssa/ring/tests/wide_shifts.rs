//! Wide negated digit traversal against ordinary integer modular arithmetic.

#![expect(
    unsafe_code,
    clippy::indexing_slicing,
    reason = "Fixed ring widths bound complete disjoint initialized coefficients and reduced negated shifts"
)]

use alloc::{vec, vec::Vec};

use crate::int::logic::unsigned::InternalMpUint;

use super::{LIMB_BITS, Limb, SsaRing};

#[cfg_attr(
    miri,
    ignore = "the fixed 65-, 128-, and 256-limb cases exercise wide shift traversals; smaller arbitrary shifts run under Miri"
)]
#[test]
fn wide_negated_traversals_match_integer_modular_shifts() {
    for ml in [65_usize, 128, 256] {
        let bits = ml * LIMB_BITS;
        let source: Vec<_> = (0..ml)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                    .rotate_left(13)
            })
            .chain([0])
            .collect();
        let value = InternalMpUint::from_limbs_slice(&source[..ml]);
        let one = InternalMpUint::one();
        let modulus = one.shl(bits).add(&one);
        for shift in [
            bits + 1,
            bits + 31,
            bits + 63,
            bits + LIMB_BITS + 31,
            2 * bits - 1,
        ] {
            let positive = value.shl(shift - bits).rem(&modulus);
            let expected = if positive.is_zero() {
                positive
            } else {
                modulus.sub(&positive)
            };
            let mut actual = source.clone();
            let mut separate = vec![Limb::MAX; ml + 1];
            let mut scratch = vec![Limb::MAX; ml + 1];
            // SAFETY: complete disjoint initialized coefficients use a canonical
            // zero guard; every selected negated exponent is below 2*bits.
            unsafe {
                SsaRing::shift_in_place(&mut actual, shift, bits, &mut scratch);
                SsaRing::shift_from(&mut separate, &source, shift, bits);
                let _ = SsaRing::normalize(&mut actual, bits);
                let _ = SsaRing::normalize(&mut separate, bits);
            }
            assert_eq!(actual, separate);
            assert_eq!(InternalMpUint::from_limbs(actual), expected);
        }
    }
}
