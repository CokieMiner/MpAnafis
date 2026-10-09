//! Unlimited signed arithmetic and absolute-value evaluation against GMP.

use mp_anafis::MpInt;
use rug::Integer;

use crate::assert_integer;

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, operation: u8) {
    match operation % 6 {
        0 => assert_integer(a + b, &Integer::from(ra + rb)),
        1 => assert_integer(a - b, &Integer::from(ra - rb)),
        2 => assert_integer(a * b, &Integer::from(ra * rb)),
        3 => {
            if rb == &0 {
                assert_eq!(a.div_rem(b), None);
                return;
            }
            let (q, r) = a.div_rem(b).unwrap();
            assert_integer(q, &Integer::from(ra / rb));
            assert_integer(r, &Integer::from(ra % rb));
        }
        4 => {
            let mut destination = a.clone();
            destination.assign_add(a, b);
            assert_integer(&destination, &Integer::from(ra + rb));
            destination.assign_sub(a, b);
            assert_integer(&destination, &Integer::from(ra - rb));
            destination.assign_mul(a, b);
            assert_integer(&destination, &Integer::from(ra * rb));
            destination.assign_square(a);
            assert_integer(destination, &Integer::from(ra * ra));
            assert_integer(a.abs(), &ra.clone().abs());
        }
        _ => {
            assert_integer(a.mul_add(a, b), &(Integer::from(ra * ra) + rb));
            assert_integer(a.midpoint(b), &(Integer::from(ra + rb) / 2_u32));
        }
    }
}
