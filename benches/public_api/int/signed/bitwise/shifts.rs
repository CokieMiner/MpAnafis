//! Shift counts measured independently on negative operands.
//! Right shifts in both libraries round toward negative infinity.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench},
};

paired_bench!(shl_3, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a << 3_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a << 3_u32),
);
paired_bench!(shl_31, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a << 31_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a << 31_u32),
);
paired_bench!(shl_137, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a << 137_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a << 137_u32),
);
paired_bench!(shr_3, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a >> 3_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a >> 3_u32),
);
paired_bench!(shr_31, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a >> 31_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a >> 31_u32),
);
paired_bench!(shr_137, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a >> 137_usize,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a >> 137_u32),
);
