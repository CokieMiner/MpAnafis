//! Common-prefix and asymmetric GCD fixtures across algorithm boundaries.

use core::mem::size_of;

use mp_anafis::MpUint;
use rug::{Integer, integer::Order};

use crate::assert_integer;

#[test]
#[cfg_attr(
    miri,
    ignore = "GCD comparisons use GMP native FFI unavailable to Miri"
)]
fn gcd_matches_gmp_for_shared_high_limbs_and_asymmetric_tails() {
    for limbs in [
        1_usize, 2, 3, 4, 5, 63, 64, 65, 122, 123, 124, 127, 128, 129, 177, 178, 179, 257,
    ] {
        let bytes = limbs.checked_mul(size_of::<usize>()).unwrap();
        let mut state = 17_u8;
        let high: Vec<_> = (0..bytes)
            .map(|_| {
                state = state.wrapping_mul(157).wrapping_add(1);
                state
            })
            .collect();
        let mut first = vec![0, 0xff, 0x7f, 1];
        let mut second = vec![1, 0x80, 0xfe, 1];
        first.extend_from_slice(&high);
        second.extend_from_slice(&high);
        let left = MpUint::from_le_bytes(&first);
        for bytes in [second, vec![1], vec![0xff; size_of::<usize>()]] {
            let right = MpUint::from_le_bytes(&bytes);
            let reference = Integer::from_digits(&first, Order::Lsf)
                .gcd(&Integer::from_digits(&bytes, Order::Lsf));
            assert_integer(left.gcd(&right), &reference);
            assert_integer(right.gcd(&left), &reference);
        }
    }
}
