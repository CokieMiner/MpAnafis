//! Signed matrix quotients with independent 128-bit carry-chain oracles.

use alloc::vec;

use super::super::{LIMB_BITS, Limb, Toom8};

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    reason = "The signed oracle is encoded modulo 2^128 and extracted one native limb at a time on every supported pointer width"
)]
fn exact_matrix_divisions_preserve_signed_carry_chains() {
    const QUOTIENTS: [i128; 9] = [
        0,
        1,
        -1,
        (1 << 32) - 1,
        -(1 << 32),
        (1 << 64) - 1,
        -(1 << 64),
        (1 << 90) - 1,
        -((1 << 90) - 1),
    ];
    for divisor in [9_u64, 1_020, 2_835, 42_525, 48_070_897_875, 46_591_793_325] {
        for quotient in QUOTIENTS {
            // |q|<2^90 and d<2^36 give |q*d|<2^126.
            let mut encoded = quotient
                .checked_mul(i128::from(divisor))
                .expect("oracle product fits i128") as u128;
            let mut value = vec![0; 128_usize.div_euclid(LIMB_BITS)];
            for limb in &mut value {
                *limb = encoded as Limb;
                encoded >>= LIMB_BITS;
            }
            match divisor {
                9 => Toom8::exact_signed_div_u64::<9>(&mut value),
                1_020 => Toom8::exact_signed_div_u64::<1_020>(&mut value),
                2_835 => Toom8::exact_signed_div_u64::<2_835>(&mut value),
                42_525 => Toom8::exact_signed_div_u64::<42_525>(&mut value),
                48_070_897_875 => Toom8::exact_signed_div_u64::<48_070_897_875>(&mut value),
                _ => Toom8::exact_signed_div_u64::<46_591_793_325>(&mut value),
            }
            let mut expected = quotient as u128;
            for limb in value {
                assert_eq!(
                    limb, expected as Limb,
                    "exact {quotient}*{divisor}/{divisor}"
                );
                expected >>= LIMB_BITS;
            }
        }
    }
}
