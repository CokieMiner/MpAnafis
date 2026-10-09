//! Addressed reads and writes of one bit, plus the `bit_range` slice.
//!
//! Mp writers return a new value. The Rug reference clones the input before
//! applying its in-place write, so both cases include result construction.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint, paired_bench},
};

/// Bit position used by each addressed access.
const TARGET_BIT: usize = 17;

/// Inclusive lower bit index of the extracted range.
const RANGE_START: usize = 8;
/// Exclusive upper bit index of the extracted range.
const RANGE_END: usize = 200;

paired_bench!(get_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.get_bit(*index),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| value.get_bit(*index),
);
paired_bench!(test_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.test_bit(*index),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| value.get_bit(*index),
);
paired_bench!(set_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.set_bit(*index),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| {
        let mut updated = value.clone();
        let _written = updated.set_bit(*index, true);
        updated
    },
);
paired_bench!(clear_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.clear_bit(*index),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| {
        let mut updated = value.clone();
        let _written = updated.set_bit(*index, false);
        updated
    },
);
paired_bench!(toggle_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.toggle_bit(*index),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| {
        let mut updated = value.clone();
        let _written = updated.toggle_bit(*index);
        updated
    },
);
paired_bench!(set_bit_to, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_bit_input => |(value, index): &(MpUint, usize)| value.set_bit_to(*index, true),
    rug: rug_bit_input => |(value, index): &(Integer, u32)| {
        let mut updated = value.clone();
        let _written = updated.set_bit(*index, true);
        updated
    },
);

// Extract [RANGE_START, RANGE_END) with a composed shift-and-mask reference.
// The masks and range indices are prepared and verified outside timing.
paired_bench!(bit_range, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| {
        let span = RANGE_END.checked_sub(RANGE_START).expect("ordered bit range");
        let mask = MpUint::zero().not_with_width(span).expect("valid range width");
        vec![(mp_uint(bits, 42), RANGE_START, RANGE_END, mask)]
    } => |(value, start, end, _): &(MpUint, usize, usize, MpUint)| value.bit_range(*start, *end),
    rug: |bits| {
        let start = u32::try_from(RANGE_START).expect("the range start fits in u32");
        let end = u32::try_from(RANGE_END).expect("the range end fits in u32");
        let span = end.checked_sub(start).expect("ordered bit range");
        let mask = Integer::from(-1).keep_bits(span);
        vec![(rug_uint(bits, 42), start, end, mask)]
    } => |(value, start, _, mask): &(Integer, u32, u32, Integer)| {
        let shifted = Integer::from(value >> *start);
        shifted & mask
    },
);

fn mp_bit_input(bits: usize) -> Vec<(MpUint, usize)> {
    vec![(mp_uint(bits, 42), TARGET_BIT)]
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_bit_input(bits: usize) -> Vec<(Integer, u32)> {
    let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
    vec![(rug_uint(bits, 42), index)]
}
