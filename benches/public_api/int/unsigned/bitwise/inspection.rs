//! Population counts, run lengths, width and bit scans.

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::Sub;

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, bounded_mp_uint, mp_uint, paired_bench},
};

/// First position inspected by the next-bit scans.
const SCAN_ORIGIN: usize = 15;

paired_bench!(count_ones, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.count_ones(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| {
        value.count_ones().expect("nonnegative operands have a finite population count")
    },
);

// Zero bits within the operand's bounded precision.
//
// GMP counts zeros only against an infinite sign extension, which is
// unbounded for a positive number; the counterpart is the composed
// `width - count_ones`. Both sides return `Some` for these bounded,
// nonnegative operands, and the shared template verifies every result untimed.
paired_bench!(count_zeros, NARROW, samples = (SAMPLE_SIZE_FAST, 50),
    mp: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        vec![(bounded_mp_uint(bits, 42), width)]
    } => |(value, _): &(MpUint, u32)| value.count_zeros(),
    rug: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        vec![(rug_uint(bits, 42), width)]
    } => |(value, width): &(Integer, u32)| {
        value.count_ones().map(|ones| width.saturating_sub(ones))
    },
);

// The number of bits needed to represent the value.
paired_bench!(significant_bits, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.significant_bits(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.significant_bits(),
);

// Leading zeros within the operand's bounded precision.
//
// Unbounded in GMP for the same reason as count_zeros, so the counterpart
// is `width - significant_bits`. Exact-width operands produce `Some(0)`.
paired_bench!(leading_zeros, NARROW, samples = (SAMPLE_SIZE_FAST, 50),
    mp: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        vec![(bounded_mp_uint(bits, 42), width)]
    } => |(value, _): &(MpUint, u32)| value.leading_zeros(),
    rug: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        vec![(rug_uint(bits, 42), width)]
    } => |(value, width): &(Integer, u32)| {
        Some(width.saturating_sub(value.significant_bits()))
    },
);

// The run of one bits at the top of the operand's bounded precision.
//
// The Rug reference counts leading zeros in `(2^width - 1) - value`.
// Masks are prepared outside timing and verified equally on both sides.
paired_bench!(leading_ones, NARROW, samples = (SAMPLE_SIZE_FAST, 50),
    mp: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = MpUint::zero().not_with_width(bits)
            .expect("benchmark widths are valid bounded precision");
        vec![(bounded_mp_uint(bits, 42), width, mask)]
    } => |(value, _, _): &(MpUint, u32, MpUint)| value.leading_ones(),
    rug: |bits| {
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(-1).keep_bits(width);
        vec![(rug_uint(bits, 42), width, mask)]
    } => |(value, width, mask): &(Integer, u32, Integer)| {
        let complement = Integer::from(Sub::sub(mask, value));
        Some(width.saturating_sub(complement.significant_bits()))
    },
);

// The run of zero bits at the bottom, which is `mpz_scan1(0)` for nonzero values.
// Mp defines the count for zero as zero.
paired_bench!(trailing_zeros, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.trailing_zeros(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.find_one(0).unwrap_or(0),
);

// The run of one bits at the bottom, which is `mpz_scan0(0)`.
paired_bench!(trailing_ones, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.trailing_ones(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| {
        value.find_zero(0).expect("nonnegative operands have a zero bit")
    },
);

paired_bench!(find_first_set_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.find_first_set_bit(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.find_one(0),
);

paired_bench!(find_next_set_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![(mp_uint(bits, 42), SCAN_ORIGIN)]
        => |(value, origin): &(MpUint, usize)| value.find_next_set_bit(*origin),
    rug: |bits| {
        let origin = u32::try_from(SCAN_ORIGIN).expect("the scan origin fits in u32");
        vec![(rug_uint(bits, 42), origin)]
    } => |(value, origin): &(Integer, u32)| value.find_one(*origin),
);

paired_bench!(find_first_zero_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.find_first_zero_bit(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| {
        value.find_zero(0).expect("nonnegative operands have a zero bit")
    },
);

paired_bench!(find_next_zero_bit, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![(mp_uint(bits, 42), SCAN_ORIGIN)]
        => |(value, origin): &(MpUint, usize)| value.find_next_zero_bit(*origin),
    rug: |bits| {
        let origin = u32::try_from(SCAN_ORIGIN).expect("the scan origin fits in u32");
        vec![(rug_uint(bits, 42), origin)]
    } => |(value, origin): &(Integer, u32)| {
        value.find_zero(*origin).expect("nonnegative operands have a zero bit")
    },
);
