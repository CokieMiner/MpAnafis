//! Unsigned arithmetic helpers with exact composed Rug equivalents.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::Pow};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint, unsigned_words};
use crate::int::{
    ladders::{BALANCED, NARROW},
    support::{bounded_mp_uint, mp_uint, paired_bench},
};

paired_bench!(mul_add, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), mp_uint(bits, 1_337), mp_uint(bits, 9_999))] =>
        |(a, b, c): &(MpUint, MpUint, MpUint)| a.mul_add(b, c),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337), rug_uint(bits, 9_999))] =>
        |(a, b, c): &(Integer, Integer, Integer)| Integer::from(a * b) + c,
);
paired_bench!(square, BALANCED,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.square(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a.square_ref()),
);
paired_bench!(pow, [256, 1_024],
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.pow(17),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from(a.pow(17)),
);
paired_bench!(midpoint, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), mp_uint(bits, 1_337))] => |(a, b): &(MpUint, MpUint)| a.midpoint(b),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337))] => |(a, b): &(Integer, Integer)| Integer::from(a + b) >> 1_u32,
);
paired_bench!(abs_diff, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), mp_uint(bits, 1_337))] => |(a, b): &(MpUint, MpUint)| a.abs_diff(b),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337))] => |(a, b): &(Integer, Integer)| Integer::from(a - b).abs(),
);
paired_bench!(checked_next_power_of_two, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.checked_next_power_of_two(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Some(Integer::from(1) << Integer::from(a - 1_u32).significant_bits()),
);
paired_bench!(widening_mul, NARROW,
    mp: |bits| vec![(bounded_mp_uint(bits, 42), bounded_mp_uint(bits, 1_337), bits)] =>
        |(a, b, _): &(MpUint, MpUint, usize)| a.widening_mul(b),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337), bits)] =>
        |(a, b, bits): &(Integer, Integer, usize)| unsigned_words(Integer::from(a * b), *bits),
);
paired_bench!(carrying_mul, NARROW,
    mp: |bits| vec![(bounded_mp_uint(bits, 42), bounded_mp_uint(bits, 1_337), bounded_mp_uint(bits, 9_999), bits)] =>
        |(a, b, c, _): &(MpUint, MpUint, MpUint, usize)| a.carrying_mul(b, c),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337), rug_uint(bits, 9_999), bits)] =>
        |(a, b, c, bits): &(Integer, Integer, Integer, usize)| unsigned_words(Integer::from(a * b) + c, *bits),
);
paired_bench!(carrying_mul_add, NARROW,
    mp: |bits| vec![(bounded_mp_uint(bits, 42), bounded_mp_uint(bits, 1_337), bounded_mp_uint(bits, 111), bounded_mp_uint(bits, 222), bits)] =>
        |(a, b, c, d, _): &(MpUint, MpUint, MpUint, MpUint, usize)| a.carrying_mul_add(b, c, d),
    rug: |bits| vec![(rug_uint(bits, 42), rug_uint(bits, 1_337), rug_uint(bits, 111), rug_uint(bits, 222), bits)] =>
        |(a, b, c, d, bits): &(Integer, Integer, Integer, Integer, usize)| unsigned_words(Integer::from(a * b) + c + d, *bits),
);
