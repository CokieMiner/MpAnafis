//! Signed operators and fused assignments on identical prepared operand batches.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Assign, Integer};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int_pairs;
use crate::int::{
    ladders::{ADDITIVE, MULTIPLICATIVE},
    support::{mp_int_pairs, paired_assign, paired_bench},
};

paired_bench!(add_like_signs, ADDITIVE,
    mp: |bits| mp_int_pairs(bits, false, false) => |(a, b): &(MpInt, MpInt)| a + b,
    rug: |bits| rug_int_pairs(bits, false, false) => |(a, b): &(Integer, Integer)| Integer::from(a + b),
);
paired_bench!(add_unlike_signs, ADDITIVE,
    mp: |bits| mp_int_pairs(bits, false, true) => |(a, b): &(MpInt, MpInt)| a + b,
    rug: |bits| rug_int_pairs(bits, false, true) => |(a, b): &(Integer, Integer)| Integer::from(a + b),
);
paired_bench!(sub_unlike_signs, ADDITIVE,
    mp: |bits| mp_int_pairs(bits, false, true) => |(a, b): &(MpInt, MpInt)| a - b,
    rug: |bits| rug_int_pairs(bits, false, true) => |(a, b): &(Integer, Integer)| Integer::from(a - b),
);
paired_bench!(mul, MULTIPLICATIVE,
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a * b,
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Integer::from(a * b),
);
paired_assign!(assign_add, ADDITIVE,
    mp: |bits| mp_int_pairs(bits, true, false), mp_destination => |out: &mut MpInt, (a, b): &(MpInt, MpInt)| out.assign_add(a, b),
    rug: |bits| rug_int_pairs(bits, true, false), rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a + b),
);
paired_assign!(assign_sub, ADDITIVE,
    mp: |bits| mp_int_pairs(bits, true, false), mp_destination => |out: &mut MpInt, (a, b): &(MpInt, MpInt)| out.assign_sub(a, b),
    rug: |bits| rug_int_pairs(bits, true, false), rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a - b),
);
paired_assign!(assign_mul, MULTIPLICATIVE,
    mp: |bits| mp_int_pairs(bits, true, false), mp_destination => |out: &mut MpInt, (a, b): &(MpInt, MpInt)| out.assign_mul(a, b),
    rug: |bits| rug_int_pairs(bits, true, false), rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a * b),
);
paired_assign!(assign_square, MULTIPLICATIVE,
    mp: |bits| mp_int_pairs(bits, true, false).into_iter().map(|(a, _)| a).collect::<Vec<_>>(), mp_destination => |out: &mut MpInt, a: &MpInt| out.assign_square(a),
    rug: |bits| rug_int_pairs(bits, true, false).into_iter().map(|(a, _)| a).collect::<Vec<_>>(), rug_destination => |out: &mut Integer, a: &Integer| out.assign(a.square_ref()),
);

fn mp_destination(bits: usize) -> MpInt {
    let output_bits = bits
        .checked_mul(2)
        .expect("benchmark output width fits usize");
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    MpInt::with_capacity(output_bits.div_ceil(limb_bits))
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_destination(bits: usize) -> Integer {
    Integer::with_capacity(
        bits.checked_mul(2)
            .expect("benchmark output width fits usize"),
    )
}
