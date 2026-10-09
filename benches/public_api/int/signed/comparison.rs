//! Ordering, equality, hashing, and borrowed selection on negative operands.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, rug_int_pairs, verify_hashes, verify_selection};
use crate::int::{
    ladders::NARROW,
    support::{
        SAMPLE_COUNT_FAST, SAMPLE_SIZE_FAST, hash_value, maximum, minimum, mp_int, mp_int_pairs,
        paired_bench,
    },
};

// Ordering of two negative values reverses their magnitude order.
paired_bench!(cmp, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| mp_int_pairs(bits, true, true) => |(left, right): &(MpInt, MpInt)| left.cmp(right),
    rug: |bits| rug_int_pairs(bits, true, true) => |(left, right): &(Integer, Integer)| left.cmp(right),
);
paired_bench!(eq, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| mp_int_pairs(bits, true, true) => |(left, right): &(MpInt, MpInt)| left == right,
    rug: |bits| rug_int_pairs(bits, true, true) => |(left, right): &(Integer, Integer)| left == right,
);

// Hash encodings are library-specific. Verify numeric operand identity and the
// equal-value hash contract within each library, without equating their hashes.
paired_bench!(hash, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_int(bits, 42, true)] => hash_value::<MpInt>,
    rug: |bits| vec![rug_int(bits, 42, true)] => hash_value::<Integer>,
    verify = verify_hashes,
);

// Selection returns a borrowed operand, so timing includes no clone or export.
paired_bench!(min, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| mp_int_pairs(bits, true, true) => minimum::<MpInt>,
    rug: |bits| rug_int_pairs(bits, true, true) => minimum::<Integer>,
    verify = verify_selection,
);
paired_bench!(max, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| mp_int_pairs(bits, true, true) => maximum::<MpInt>,
    rug: |bits| rug_int_pairs(bits, true, true) => maximum::<Integer>,
    verify = verify_selection,
);
