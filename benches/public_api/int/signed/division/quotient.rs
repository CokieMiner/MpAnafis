//! Signed truncating quotient and remainder with a positive nonzero divisor.

use core::ops::{Div, Rem};

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{mp_int_pairs, paired_bench, verify_mp_int_division_pairs},
};

paired_bench!(div_trunc, DIVISION, samples = (32, 50),
    mp: |bits| {
        let inputs = mp_int_pairs(bits, true, false);
        verify_mp_int_division_pairs(&inputs);
        inputs
    } => |(a, b): &(MpInt, MpInt)| a.div(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.div(b)),
);
paired_bench!(rem_trunc, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.rem(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a.rem(b)),
);
paired_bench!(div_rem, DIVISION, samples = (32, 50),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.div_rem(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Some(<(Integer, Integer)>::from(a.div_rem_ref(b))),
);
