//! Signed floor, ceiling, and Euclidean rounding on negative dividends.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{
    Integer,
    ops::{DivRounding, RemRounding},
};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{mp_int_pairs, paired_bench},
};

paired_bench!(div_floor, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.div_floor(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.div_floor(b)),
);
paired_bench!(mod_floor, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.mod_floor(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.rem_floor(b)),
);
paired_bench!(div_ceil, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.div_ceil(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.div_ceil(b)),
);
paired_bench!(div_euclid, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.div_euclid(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.div_euc(b)),
);
paired_bench!(rem_euclid, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.rem_euclid(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.rem_euc(b)),
);
