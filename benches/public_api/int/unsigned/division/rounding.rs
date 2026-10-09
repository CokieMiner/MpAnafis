//! Unsigned division policies on nonzero divisors. Ceiling rounds upward;
//! floor, truncation, and Euclidean division agree on this domain.

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::{Div, Rem};

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{
    Integer,
    ops::{DivRounding, RemRounding},
};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{mp_uint_pairs, paired_bench},
};

paired_bench!(div_euclid, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.div_euclid(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.div_euc(b)),
);
paired_bench!(rem_euclid, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.rem_euclid(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.rem_euc(b)),
);
paired_bench!(div_floor, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.div_floor(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.div_floor(b)),
);
paired_bench!(mod_floor, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.mod_floor(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.rem_floor(b)),
);
paired_bench!(div_ceil, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.div_ceil(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a.div_ceil(b)),
);
paired_bench!(div_rem_floor, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.div_rem_floor(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(<(Integer, Integer)>::from(a.div_rem_floor_ref(b))),
);
paired_bench!(div_rem_euclid, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.div_rem_euclid(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(<(Integer, Integer)>::from(a.div_rem_euc_ref(b))),
);
paired_bench!(checked_div_floor, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_div_floor(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.div_floor(b))),
);
paired_bench!(checked_div_ceil, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_div_ceil(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.div_ceil(b))),
);
paired_bench!(checked_div_euclid, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_div_euclid(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.div_euc(b))),
);
paired_bench!(checked_div_trunc, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_div_trunc(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.div(b))),
);
paired_bench!(checked_mod_floor, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_mod_floor(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.rem_floor(b))),
);
paired_bench!(checked_rem_euclid, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_rem_euclid(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.rem_euc(b))),
);
paired_bench!(checked_rem_trunc, DIVISION, samples = (4, 30),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.checked_rem_trunc(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(Integer::from(a.rem(b))),
);
