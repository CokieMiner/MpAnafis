//! Signed radix conversion on negative operands.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench},
};

paired_bench!(to_string_radix_10, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.to_string_radix(10),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.to_string_radix(10),
);
paired_bench!(from_string_radix_10, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true).to_string_radix(10)] => |a: &String| MpInt::from_str_radix(a, 10),
    rug: |bits| vec![rug_int(bits, 42, true).to_string_radix(10)] => |a: &String| Integer::from_str_radix(a, 10),
);
paired_bench!(to_string_radix_16, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.to_string_radix(16),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.to_string_radix(16),
);
