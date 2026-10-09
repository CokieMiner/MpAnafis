//! Infinite two's-complement kernels and arithmetic shifts against GMP.

use mp_anafis::MpInt;
use rug::Integer;

use crate::assert_integer;

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, operation: u8, parameter: u16) {
    let shift = usize::from(parameter % 1024);
    match operation % 5 {
        0 => assert_integer(a & b, &Integer::from(ra & rb)),
        1 => assert_integer(a | b, &Integer::from(ra | rb)),
        2 => assert_integer(a ^ b, &Integer::from(ra ^ rb)),
        3 => {
            assert_integer(a << shift, &(ra.clone() << shift));
            assert_integer(a.mul_2exp(shift), &(ra.clone() << shift));
        }
        _ => {
            assert_integer(a >> shift, &(ra.clone() >> shift));
            assert_integer(a.div_2exp(shift), &(ra.clone() >> shift));
        }
    }
}
