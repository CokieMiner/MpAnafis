//! One unsigned shift method per case, with isolated shift-count scenarios.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{mp_uint, paired_bench},
};

paired_bench!(shl_3, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a << 3_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a << 3_u32),
);
paired_bench!(shl_31, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a << 31_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a << 31_u32),
);
paired_bench!(shl_137, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a << 137_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a << 137_u32),
);
paired_bench!(shr_3, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a >> 3_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a >> 3_u32),
);
paired_bench!(shr_31, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a >> 31_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a >> 31_u32),
);
paired_bench!(shr_137, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a >> 137_usize,
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a >> 137_u32),
);
paired_bench!(checked_shl, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.checked_shl(137),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Some(Integer::from(a << 137_u32)),
);
paired_bench!(wrapping_shl, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.wrapping_shl(137),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a << 137_u32),
);
paired_bench!(overflowing_shl, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.overflowing_shl(137),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| (Integer::from(a << 137_u32), false),
);
paired_bench!(saturating_shl, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.saturating_shl(137),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a << 137_u32),
);
paired_bench!(try_shl, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.try_shl(137),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Ok::<_, mp_anafis::MpError>(Integer::from(a << 137_u32)),
);
