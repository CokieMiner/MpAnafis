//! Integer roots and the perfect-square test.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_square_plus_one, rug_true_squares, rug_uint};
use crate::int::{
    ladders::{NARROW, ROOTS},
    support::{
        SAMPLE_COUNT_FAST, SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_WIDE,
        mp_square_plus_one, mp_true_squares, mp_uint, paired_bench,
    },
};

/// Registers higher degrees below the cube-root comparison.
macro_rules! higher_degree_roots {
    ($widths:expr, $size:expr, $_count:expr, $_mp_op:expr, $_rug_op:expr, $verify:expr) => {
        paired_bench!(degree_five, $widths, samples = ($size, SAMPLE_COUNT_WIDE),
            mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.nth_root(5),
            rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(Integer::from(value.root_ref(5))),
            verify = $verify,
        );
        paired_bench!(degree_seven, $widths, samples = ($size, SAMPLE_COUNT_WIDE),
            mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.nth_root(7),
            rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(Integer::from(value.root_ref(7))),
            verify = $verify,
        );
        paired_bench!(degree_seventeen, $widths, samples = ($size, SAMPLE_COUNT_WIDE),
            mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.nth_root(17),
            rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(Integer::from(value.root_ref(17))),
            verify = $verify,
        );
    };
}

paired_bench!(isqrt, ROOTS, samples = (SAMPLE_SIZE_WIDE, SAMPLE_COUNT_WIDE),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.isqrt(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(Integer::from(value.sqrt_ref())),
);
paired_bench!(sqrt_rem, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.sqrt_rem(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(value.clone().sqrt_rem(Integer::new())),
);
paired_bench!(nth_root, ROOTS, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.nth_root(3),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| Some(Integer::from(value.root_ref(3))),
    scenarios = higher_degree_roots,
);

// Random exact-width magnitudes.
paired_bench!(is_perfect_square_random, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: |bits| vec![mp_uint(bits, 42)] => |value: &MpUint| value.is_perfect_square(),
    rug: |bits| vec![rug_uint(bits, 42)] => |value: &Integer| value.is_perfect_square(),
);
// Exact squares requiring the integer square root.
paired_bench!(is_perfect_square_true, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_true_squares => |value: &MpUint| value.is_perfect_square(),
    rug: rug_true_squares => |value: &Integer| value.is_perfect_square(),
);
// Non-squares k^2+1 may be rejected before the integer square root.
paired_bench!(is_perfect_square_plus_one, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_square_plus_one => |value: &MpUint| value.is_perfect_square(),
    rug: rug_square_plus_one => |value: &Integer| value.is_perfect_square(),
);
