//! Bitwise operators on negative and mixed-sign operands, including inline boundaries.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int_pairs;
use crate::int::support::{mp_int_pairs, paired_bench};

// Four native limbs occupy 256 bits on the comparison host.
const BITWISE_WIDTHS: &[usize] = &[
    64, 128, 192, 256, 320, 512, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576,
];

paired_bench!(bitand, BITWISE_WIDTHS,
    mp: |bits| mp_int_pairs(bits, true, true) => |(a, b): &(MpInt, MpInt)| a & b,
    rug: |bits| rug_int_pairs(bits, true, true) => |(a, b): &(Integer, Integer)| Integer::from(a & b),
);
paired_bench!(bitor, BITWISE_WIDTHS,
    mp: |bits| mp_int_pairs(bits, true, true) => |(a, b): &(MpInt, MpInt)| a | b,
    rug: |bits| rug_int_pairs(bits, true, true) => |(a, b): &(Integer, Integer)| Integer::from(a | b),
);
paired_bench!(bitxor, BITWISE_WIDTHS,
    mp: |bits| mp_int_pairs(bits, true, true) => |(a, b): &(MpInt, MpInt)| a ^ b,
    rug: |bits| rug_int_pairs(bits, true, true) => |(a, b): &(Integer, Integer)| Integer::from(a ^ b),
);
paired_bench!(bitor_mixed_signs, BITWISE_WIDTHS,
    mp: |bits| mp_int_pairs(bits, false, true) => |(a, b): &(MpInt, MpInt)| a | b,
    rug: |bits| rug_int_pairs(bits, false, true) => |(a, b): &(Integer, Integer)| Integer::from(a | b),
);
