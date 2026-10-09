//! Signed square root on its nonnegative domain.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench},
};

paired_bench!(checked_isqrt, NARROW,
    mp: |bits| vec![mp_int(bits, 42, false)] => |a: &MpInt| a.checked_isqrt(),
    rug: |bits| vec![rug_int(bits, 42, false)] => |a: &Integer| Some(Integer::from(a.sqrt_ref())),
);
