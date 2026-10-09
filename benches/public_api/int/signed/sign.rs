//! Sign operations on negative operands; each predicate has its own measurement.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench, paired_mutate},
};

paired_bench!(abs, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.abs(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a.abs_ref()),
);
paired_mutate!(abs_assign, NARROW,
    mp: |bits| mp_int(bits, 42, true) => |a: &mut MpInt| a.abs_assign(),
    rug: |bits| rug_int(bits, 42, true) => |a: &mut Integer| a.abs_mut(),
);
paired_bench!(checked_abs, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.checked_abs(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Some(Integer::from(a.abs_ref())),
);
paired_bench!(neg, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| -a,
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(-a),
);
paired_bench!(signum, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.signum(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a.signum_ref()),
);
paired_bench!(unsigned_abs, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.unsigned_abs(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a.abs_ref()),
);
paired_bench!(abs_sub, NARROW,
    mp: |bits| vec![(mp_int(bits, 42, false), mp_int(bits, 1_337, true))] => |(a, b): &(MpInt, MpInt)| a.abs_sub(b),
    rug: |bits| vec![(rug_int(bits, 42, false), rug_int(bits, 1_337, true))] => |(a, b): &(Integer, Integer)| Integer::from(a - b).max(Integer::ZERO),
);
paired_bench!(abs_diff, NARROW,
    mp: |bits| vec![(mp_int(bits, 42, true), mp_int(bits, 1_337, false))] => |(a, b): &(MpInt, MpInt)| a.abs_diff(b),
    rug: |bits| vec![(rug_int(bits, 42, true), rug_int(bits, 1_337, false))] => |(a, b): &(Integer, Integer)| Integer::from(a - b).abs(),
);
paired_bench!(is_negative, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.is_negative(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a < &0,
);
paired_bench!(is_positive, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.is_positive(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a > &0,
);
paired_bench!(is_minus_one, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.is_minus_one(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| a == &-1,
);
