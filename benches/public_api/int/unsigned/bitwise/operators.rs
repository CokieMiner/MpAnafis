//! `BitAnd`, `BitOr`, `BitXor` and the width-bounded complements
//! `not_with_width` and `try_not`.

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::Sub;
use core::ops::{BitAnd, BitOr, BitXor};

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint, rug_uint_pairs};
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, bounded_mp_uint, mp_uint, mp_uint_pairs, paired_bench},
};

paired_bench!(bitand, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_uint_pairs => |(left, right): &(MpUint, MpUint)| BitAnd::bitand(left, right),
    rug: rug_uint_pairs => |(left, right): &(Integer, Integer)| Integer::from(BitAnd::bitand(left, right)),
);
paired_bench!(bitor, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_uint_pairs => |(left, right): &(MpUint, MpUint)| BitOr::bitor(left, right),
    rug: rug_uint_pairs => |(left, right): &(Integer, Integer)| Integer::from(BitOr::bitor(left, right)),
);
paired_bench!(bitxor, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_uint_pairs => |(left, right): &(MpUint, MpUint)| BitXor::bitxor(left, right),
    rug: rug_uint_pairs => |(left, right): &(Integer, Integer)| Integer::from(BitXor::bitxor(left, right)),
);

// GMP's infinite complement is reduced to `(2^width - 1) - value`.
// Both masks are prepared and verified outside timing.
paired_bench!(not_with_width, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| {
        let mask = MpUint::zero().not_with_width(bits).expect("valid benchmark width");
        vec![(mp_uint(bits, 42), bits, mask)]
    } => |(value, width, _): &(MpUint, usize, MpUint)| value.not_with_width(*width),
    rug: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(-1).keep_bits(width);
        vec![(rug_uint(bits, 42), bits, mask)]
    } => |(value, _, mask): &(Integer, usize, Integer)| Some(Integer::from(Sub::sub(mask, value))),
);

// Bounded precision supplies the complement width; unlimited precision would
// return `WidthRequired`. The composed reference preserves the result wrapper.
paired_bench!(try_not, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| {
        let mask = MpUint::zero().not_with_width(bits).expect("valid benchmark width");
        vec![(bounded_mp_uint(bits, 42), mask)]
    } => |(value, _): &(MpUint, MpUint)| value.try_not(),
    rug: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(-1).keep_bits(width);
        vec![(rug_uint(bits, 42), mask)]
    } => |(value, mask): &(Integer, Integer)| Ok::<_, mp_anafis::MpError>(Integer::from(Sub::sub(mask, value))),
);
