//! Unlimited arithmetic and reusable destination evaluation against GMP.

use mp_anafis::MpUint;
use rug::Integer;

use crate::assert_integer;

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, operation: u8) {
    match operation % 6 {
        0 => assert_integer(a + b, &Integer::from(ra + rb)),
        1 => {
            if ra >= rb {
                assert_integer(a - b, &Integer::from(ra - rb));
            }
        }
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
            let before = destination.clone();
            assert_eq!(destination.assign_sub(a, b), ra < rb);
            assert_integer(
                &destination,
                &if ra < rb {
                    Integer::from(ra + rb)
                } else {
                    Integer::from(ra - rb)
                },
            );
            if ra < rb {
                assert_eq!(destination.precision(), before.precision());
            }
            destination.assign_mul(a, b);
            assert_integer(&destination, &Integer::from(ra * rb));
            destination.assign_square(a);
            assert_integer(destination, &Integer::from(ra * ra));
        }
        _ => {
            assert_integer(a.mul_add(a, b), &(Integer::from(ra * ra) + rb));
            assert_integer(a.midpoint(b), &(Integer::from(ra + rb) / 2_u32));
            assert_integer(a.abs_diff(b), &Integer::from(ra - rb).abs());
        }
    }
}
