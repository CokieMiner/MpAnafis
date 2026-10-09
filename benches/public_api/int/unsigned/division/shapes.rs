//! Independent quotient, remainder, and combined calls on recursive and short
//! quotient shapes. Arguments are divisor bits; exact dividend widths depend
//! on the product `Q*D+R` and can be one bit below the nominal ratio.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_division_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{DivisionResidue, DivisionShape, mp_division_pairs, paired_bench},
};

/// Each shape uses the same fixtures, validation, and sampling for all outputs.
macro_rules! division_cases {
    ($quotient:ident, $remainder:ident, $combined:ident, $shape:ident, $residue:ident) => {
        paired_bench!($quotient, DIVISION, samples = (1, 20),
            mp: |bits| mp_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(MpUint, MpUint)| a.div_trunc(b),
            rug: |bits| rug_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(Integer, Integer)| Integer::from(a / b),
        );
        paired_bench!($remainder, DIVISION, samples = (1, 20),
            mp: |bits| mp_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(MpUint, MpUint)| a.rem_trunc(b),
            rug: |bits| rug_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(Integer, Integer)| Integer::from(a % b),
        );
        paired_bench!($combined, DIVISION, samples = (1, 20),
            mp: |bits| mp_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(MpUint, MpUint)| a.div_rem(b),
            rug: |bits| rug_division_pairs(bits, DivisionShape::$shape, DivisionResidue::$residue)
                => |(a, b): &(Integer, Integer)| Some(<(Integer, Integer)>::from(a.div_rem_ref(b))),
        );
    };
}

division_cases!(
    div_trunc_2n_by_n,
    rem_trunc_2n_by_n,
    div_rem_2n_by_n,
    Balanced,
    Random
);
division_cases!(
    div_trunc_2n_by_n_exact,
    rem_trunc_2n_by_n_exact,
    div_rem_2n_by_n_exact,
    Balanced,
    Zero
);
division_cases!(
    div_trunc_2n_by_n_near_multiple,
    rem_trunc_2n_by_n_near_multiple,
    div_rem_2n_by_n_near_multiple,
    Balanced,
    Maximal
);
division_cases!(
    div_trunc_3n2_by_n,
    rem_trunc_3n2_by_n,
    div_rem_3n2_by_n,
    ThreeByTwo,
    Random
);
division_cases!(
    div_trunc_3n2_by_n_exact,
    rem_trunc_3n2_by_n_exact,
    div_rem_3n2_by_n_exact,
    ThreeByTwo,
    Zero
);
division_cases!(
    div_trunc_3n2_by_n_near_multiple,
    rem_trunc_3n2_by_n_near_multiple,
    div_rem_3n2_by_n_near_multiple,
    ThreeByTwo,
    Maximal
);
division_cases!(
    div_trunc_4n_by_n,
    rem_trunc_4n_by_n,
    div_rem_4n_by_n,
    Long,
    Random
);
division_cases!(
    div_trunc_4n_by_n_exact,
    rem_trunc_4n_by_n_exact,
    div_rem_4n_by_n_exact,
    Long,
    Zero
);
division_cases!(
    div_trunc_4n_by_n_near_multiple,
    rem_trunc_4n_by_n_near_multiple,
    div_rem_4n_by_n_near_multiple,
    Long,
    Maximal
);
division_cases!(
    div_trunc_small_quotient,
    rem_trunc_small_quotient,
    div_rem_small_quotient,
    SmallQuotient,
    Random
);
division_cases!(
    div_trunc_small_quotient_exact,
    rem_trunc_small_quotient_exact,
    div_rem_small_quotient_exact,
    SmallQuotient,
    Zero
);
division_cases!(
    div_trunc_small_quotient_near_multiple,
    rem_trunc_small_quotient_near_multiple,
    div_rem_small_quotient_near_multiple,
    SmallQuotient,
    Maximal
);
division_cases!(
    div_trunc_power_of_two,
    rem_trunc_power_of_two,
    div_rem_power_of_two,
    PowerOfTwo,
    Random
);
division_cases!(
    div_trunc_power_of_two_exact,
    rem_trunc_power_of_two_exact,
    div_rem_power_of_two_exact,
    PowerOfTwo,
    Zero
);
division_cases!(
    div_trunc_power_of_two_near_multiple,
    rem_trunc_power_of_two_near_multiple,
    div_rem_power_of_two_near_multiple,
    PowerOfTwo,
    Maximal
);
