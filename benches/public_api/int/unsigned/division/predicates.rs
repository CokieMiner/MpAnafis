//! Divisibility tests, which may answer without producing a quotient.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_divisibility_pairs, rug_uint_pairs};
use crate::int::{
    ladders::NARROW,
    support::{
        DivisibilityShape, SAMPLE_COUNT_FAST, SAMPLE_SIZE_FAST, mp_divisibility_pairs,
        mp_uint_pairs, paired_bench,
    },
};

/// Registers exact and nonmultiple inputs for each cancellation shape.
macro_rules! cancellation_scenarios {
    ($widths:expr, $_size:expr, $_count:expr, $_mp:expr, $_rug:expr, $verify:expr) => {
        cancellation_scenarios!(@cases $widths, $verify;
            odd_exact = (Odd, true), odd_nonmultiple = (Odd, false),
            shifted_exact = (Shifted, true), shifted_nonmultiple = (Shifted, false),
            shifted_scalar_exact = (ShiftedScalar, true),
            shifted_scalar_nonmultiple = (ShiftedScalar, false));
    };
    (@cases $widths:expr, $verify:expr;
        $($name:ident = ($shape:ident, $exact:literal)),+ $(,)?) => { $(
        paired_bench!($name, $widths, samples = (1, 20),
            mp: |bits| mp_divisibility_pairs(bits, DivisibilityShape::$shape, $exact)
                => |(a, b): &(MpUint, MpUint)| a.is_divisible_by(b),
            rug: |bits| rug_divisibility_pairs(bits, DivisibilityShape::$shape, $exact)
                => |(a, b): &(Integer, Integer)| a.is_divisible(b),
            verify = $verify,
        );
    )+ };
}

paired_bench!(is_divisible_by, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a.is_divisible_by(b),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| a.is_divisible(b),
    scenarios = cancellation_scenarios,
);
paired_bench!(is_divisor_of, NARROW, samples = (SAMPLE_SIZE_FAST, SAMPLE_COUNT_FAST),
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| b.is_divisor_of(a),
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| a.is_divisible(b),
);
