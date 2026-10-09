//! Radix formatting and parsing.
//!
//! Non-power-of-two bases cover limb-sized digit blocks and recursive
//! division. Power-of-two bases cover byte lookup tables and bit extraction.
//! Display and binary, octal, and hexadecimal traits use the same operands.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint, paired_bench},
};

paired_bench!(to_string_radix_10, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(10),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(10),
);
paired_bench!(to_string_radix_3, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(3),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(3),
);
paired_bench!(to_string_radix_9, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(9),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(9),
);
paired_bench!(to_string_radix_19, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(19),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(19),
);
paired_bench!(to_string_radix_36, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(36),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(36),
);
paired_bench!(from_string_radix_10, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42).to_string_radix(10)] => |text: &String| MpUint::from_str_radix(text, 10),
    rug: |bits| vec![rug_uint(bits, 42).to_string_radix(10)] => |text: &String| Integer::from_str_radix(text, 10),
);
paired_bench!(to_string_radix_16, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(16),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(16),
);
paired_bench!(from_string_radix_16, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42).to_string_radix(16)] => |text: &String| MpUint::from_str_radix(text, 16),
    rug: |bits| vec![rug_uint(bits, 42).to_string_radix(16)] => |text: &String| Integer::from_str_radix(text, 16),
);
paired_bench!(to_string_radix_32, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(32),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(32),
);
paired_bench!(to_string_radix_8, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(8),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(8),
);
paired_bench!(to_string_radix_2, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(2),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(2),
);
paired_bench!(to_string_radix_4, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string_radix(4),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string_radix(4),
);
paired_bench!(from_string_radix_2, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42).to_string_radix(2)] => |text: &String| MpUint::from_str_radix(text, 2),
    rug: |bits| vec![rug_uint(bits, 42).to_string_radix(2)] => |text: &String| Integer::from_str_radix(text, 2),
);

// Decimal formatting through `Display`.
paired_bench!(display, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.to_string(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.to_string(),
);
paired_bench!(lower_hex, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| format!("{value:x}"),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| format!("{value:x}"),
);
paired_bench!(binary, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| format!("{value:b}"),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| format!("{value:b}"),
);
paired_bench!(octal, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| format!("{value:o}"),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| format!("{value:o}"),
);
