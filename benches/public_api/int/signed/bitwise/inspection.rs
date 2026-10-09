//! Signed bit inspection. Unlimited negative population counts return None.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench},
};

paired_bench!(count_ones, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.count_ones(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.count_ones(),
);
paired_bench!(trailing_zeros, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.trailing_zeros(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.find_one(0).expect("nonzero input"),
);
paired_bench!(significant_bits, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.significant_bits(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.significant_bits(),
);
