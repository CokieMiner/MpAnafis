//! Sparse exponent and large modulus fixtures beyond the campaign scalar limits.

use core::mem::size_of;

use mp_anafis::MpUint;
use rug::{Integer, integer::Order};

use crate::assert_optional;

#[test]
#[cfg_attr(
    miri,
    ignore = "Modular comparisons use GMP native FFI unavailable to Miri"
)]
fn sparse_modular_powers_match_gmp_across_operand_widths() {
    for limbs in [1_usize, 2, 4, 5, 16, 33, 34] {
        let bytes = limbs.checked_mul(size_of::<usize>()).unwrap();
        let modulus_bytes = vec![0xff; bytes];
        let base_bytes = vec![0xf1; bytes.checked_add(6).unwrap()];
        let modulus = MpUint::from_le_bytes(&modulus_bytes);
        let base = MpUint::from_le_bytes(&base_bytes);
        for high in [0_usize, 1, 64, 128, 256] {
            for low in [0, 1, high / 2] {
                let exponent = (MpUint::from(1_u8) << high) + (MpUint::from(1_u8) << low);
                let expected = Integer::from_digits(&base_bytes, Order::Lsf)
                    .pow_mod(
                        &Integer::from_digits(&exponent.to_le_bytes(), Order::Lsf),
                        &Integer::from_digits(&modulus_bytes, Order::Lsf),
                    )
                    .unwrap();
                assert_optional(base.pow_mod(&exponent, &modulus), Some(expected));
            }
        }
    }
}
