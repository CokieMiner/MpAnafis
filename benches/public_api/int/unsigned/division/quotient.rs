//! Truncating division on equal-width operands.
//!
//! Exact equal bit widths give a quotient of zero or one. The separate shape
//! cases use wider dividends to exercise longer quotient computations.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{
        SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, mp_uint_pairs, paired_bench,
        verify_mp_uint_division_pairs,
    },
};

paired_bench!(div_trunc, DIVISION, samples = (SAMPLE_SIZE_WIDE, SAMPLE_COUNT_WIDE),
    mp: |bits| {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        inputs
    } => |(a, b): &(MpUint, MpUint)| a.div_trunc(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a / b),
);
paired_bench!(rem_trunc, DIVISION, samples = (SAMPLE_SIZE_WIDE, SAMPLE_COUNT_WIDE),
    mp: |bits| {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        inputs
    } => |(a, b): &(MpUint, MpUint)| a.rem_trunc(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a % b),
);
paired_bench!(div_rem, DIVISION, samples = (SAMPLE_SIZE_WIDE, SAMPLE_COUNT_WIDE),
    mp: |bits| {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        inputs
    } => |(a, b): &(MpUint, MpUint)| a.div_rem(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Some(<(Integer, Integer)>::from(a.div_rem_ref(b))),
);
