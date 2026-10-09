//! One case per primitive conversion, including representable and rejected widths.
//! Float references compose ties-to-even rounding with GMP's truncating export.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use crate::int::support::{mp_uint, paired_bench};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{nearest_f32, nearest_f64, rug_uint};

paired_bench!(to_u64, [32, 64, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_u64(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_u64(),
);
paired_bench!(to_i64, [32, 64, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_i64(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_i64(),
);
paired_bench!(to_u128, [64, 128, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_u128(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_u128(),
);
paired_bench!(to_i128, [64, 128, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_i128(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_i128(),
);
paired_bench!(to_usize, [8, 64, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_usize(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_usize(),
);
paired_bench!(to_isize, [8, 64, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_isize(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| a.to_isize(),
);
paired_bench!(to_f64, [32, 256, 1_024, 2_048],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_f64(),
    rug: |bits| vec![rug_uint(bits, 42)] => nearest_f64,
);
paired_bench!(to_f32, [32, 128, 256],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.to_f32(),
    rug: |bits| vec![rug_uint(bits, 42)] => nearest_f32,
);
paired_bench!(from_u64, [64],
    mp: |_| vec![0xdead_beef_cafe_f00d_u64] => |a: &u64| MpUint::from(*a),
    rug: |_| vec![0xdead_beef_cafe_f00d_u64] => |a: &u64| Integer::from(*a),
);
