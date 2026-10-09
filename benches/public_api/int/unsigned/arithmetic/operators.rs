//! Unsigned operators and fused assignments on identical prepared operand batches.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Assign, Integer};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint_lopsided_pairs, rug_uint_pairs};
use crate::int::{
    ladders::{ADDITIVE, MULTIPLICATIVE},
    support::{mp_uint_lopsided_pairs, mp_uint_pairs, paired_assign, paired_bench},
};

paired_bench!(add, ADDITIVE,
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a + b,
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a + b),
);
paired_bench!(sub, ADDITIVE,
    mp: |bits| ordered(mp_uint_pairs(bits)) => |(a, b): &(MpUint, MpUint)| a - b,
    rug: |bits| ordered(rug_uint_pairs(bits)) => |(a, b): &(Integer, Integer)| Integer::from(a - b),
);
paired_bench!(mul, MULTIPLICATIVE,
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a * b,
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a * b),
);
paired_bench!(mul_3n2_by_n, MULTIPLICATIVE,
    mp: |bits: usize| mp_uint_lopsided_pairs(bits.checked_add(bits >> 1).expect("unbalanced width fits"), bits)
        => |(a, b): &(MpUint, MpUint)| a * b,
    rug: |bits: usize| rug_uint_lopsided_pairs(bits.checked_add(bits >> 1).expect("unbalanced width fits"), bits)
        => |(a, b): &(Integer, Integer)| Integer::from(a * b),
);
paired_bench!(mul_2n_by_n, MULTIPLICATIVE,
    mp: |bits: usize| mp_uint_lopsided_pairs(bits.checked_mul(2).expect("unbalanced width fits"), bits)
        => |(a, b): &(MpUint, MpUint)| a * b,
    rug: |bits: usize| rug_uint_lopsided_pairs(bits.checked_mul(2).expect("unbalanced width fits"), bits)
        => |(a, b): &(Integer, Integer)| Integer::from(a * b),
);
paired_bench!(mul_4n_by_n, MULTIPLICATIVE,
    mp: |bits: usize| mp_uint_lopsided_pairs(bits.checked_mul(4).expect("unbalanced width fits"), bits)
        => |(a, b): &(MpUint, MpUint)| a * b,
    rug: |bits: usize| rug_uint_lopsided_pairs(bits.checked_mul(4).expect("unbalanced width fits"), bits)
        => |(a, b): &(Integer, Integer)| Integer::from(a * b),
);
paired_assign!(assign_add, ADDITIVE,
    mp: mp_uint_pairs, mp_destination => |out: &mut MpUint, (a, b): &(MpUint, MpUint)| out.assign_add(a, b),
    rug: rug_uint_pairs, rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a + b),
);
paired_assign!(assign_sub, ADDITIVE,
    mp: |bits| ordered(mp_uint_pairs(bits)), mp_destination => |out: &mut MpUint, (a, b): &(MpUint, MpUint)| { let _underflow = out.assign_sub(a, b); },
    rug: |bits| ordered(rug_uint_pairs(bits)), rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a - b),
);
paired_assign!(assign_mul, MULTIPLICATIVE,
    mp: mp_uint_pairs, mp_destination => |out: &mut MpUint, (a, b): &(MpUint, MpUint)| out.assign_mul(a, b),
    rug: rug_uint_pairs, rug_destination => |out: &mut Integer, (a, b): &(Integer, Integer)| out.assign(a * b),
);
paired_assign!(assign_square, MULTIPLICATIVE,
    mp: |bits| mp_uint_pairs(bits).into_iter().map(|(a, _)| a).collect::<Vec<_>>(), mp_destination => |out: &mut MpUint, a: &MpUint| out.assign_square(a),
    rug: |bits| rug_uint_pairs(bits).into_iter().map(|(a, _)| a).collect::<Vec<_>>(), rug_destination => |out: &mut Integer, a: &Integer| out.assign(a.square_ref()),
);

fn ordered<T: Ord>(pairs: Vec<(T, T)>) -> Vec<(T, T)> {
    pairs
        .into_iter()
        .map(|(a, b)| if a >= b { (a, b) } else { (b, a) })
        .collect()
}

fn mp_destination(bits: usize) -> MpUint {
    let output_bits = bits
        .checked_mul(2)
        .expect("benchmark output width fits usize");
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    MpUint::with_capacity(output_bits.div_ceil(limb_bits))
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_destination(bits: usize) -> Integer {
    Integer::with_capacity(
        bits.checked_mul(2)
            .expect("benchmark output width fits usize"),
    )
}
