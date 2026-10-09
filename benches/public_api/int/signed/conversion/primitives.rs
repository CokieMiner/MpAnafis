//! Signed primitive conversions with representable and rejected widths.
//! Float references compose ties-to-even rounding with GMP's truncating export.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use crate::int::support::{mp_int, paired_bench};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{nearest_f64, rug_int};

paired_bench!(to_i64, [32, 64, 256],
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.to_i64(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a.to_i64(),
);
paired_bench!(to_f64, [32, 256, 1_024, 2_048],
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.to_f64(),
    rug: |bits| vec![rug_int(bits, 42, true)] => nearest_f64,
);
