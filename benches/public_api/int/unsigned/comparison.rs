//! Ordering, equality, hashing, borrowed selection, and value predicates.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint, rug_uint_pairs, verify_hashes, verify_selection};
use crate::int::{
    ladders::NARROW,
    support::{
        SAMPLE_COUNT_FAST, SAMPLE_SIZE_FAST, clamped, hash_value, maximum, minimum, mp_uint,
        mp_uint_pairs, paired_bench,
    },
};

paired_bench!(cmp, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => |(left, right): &(MpUint, MpUint)| left.cmp(right),
    rug: rug_uint_pairs => |(left, right): &(Integer, Integer)| left.cmp(right),
);
paired_bench!(eq, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => |(left, right): &(MpUint, MpUint)| left == right,
    rug: rug_uint_pairs => |(left, right): &(Integer, Integer)| left == right,
);

// Hash representations differ between libraries. Verification checks operand
// identity and equal-value hashes within each implementation.
paired_bench!(hash, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => hash_value::<MpUint>,
    rug: |bits| vec![rug_uint(bits, 42)] => hash_value::<Integer>,
    verify = verify_hashes,
);

// Selection returns a borrowed operand. Encoding is restricted to verification.
paired_bench!(min, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => minimum::<MpUint>,
    rug: rug_uint_pairs => minimum::<Integer>,
    verify = verify_selection,
);
paired_bench!(max, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => maximum::<MpUint>,
    rug: rug_uint_pairs => maximum::<Integer>,
    verify = verify_selection,
);
paired_bench!(clamp, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| {
        let value = mp_uint(bits, 42);
        let first_bound = mp_uint(bits, 1);
        let second_bound = mp_uint(bits, 9_999);
        let (lower, upper) = if first_bound <= second_bound {
            (first_bound, second_bound)
        } else {
            (second_bound, first_bound)
        };
        vec![(value, lower, upper)]
    } => clamped::<MpUint>,
    rug: |bits| {
        let value = rug_uint(bits, 42);
        let first_bound = rug_uint(bits, 1);
        let second_bound = rug_uint(bits, 9_999);
        let (lower, upper) = if first_bound <= second_bound {
            (first_bound, second_bound)
        } else {
            (second_bound, first_bound)
        };
        vec![(value, lower, upper)]
    } => clamped::<Integer>,
    verify = verify_selection,
);
paired_bench!(is_even, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.is_even(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.is_even(),
);
paired_bench!(is_odd, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.is_odd(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.is_odd(),
);
paired_bench!(is_power_of_two, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.is_power_of_two(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.count_ones() == Some(1),
);
