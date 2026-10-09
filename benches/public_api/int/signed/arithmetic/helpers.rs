//! Signed arithmetic helpers with exact composed Rug equivalents.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::Pow};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, signed_words};
use crate::int::{
    ladders::{BALANCED, NARROW},
    support::{bounded_mp_int, mp_int, paired_bench},
};

paired_bench!(mul_add, NARROW,
    mp: |bits| vec![(mp_int(bits, 42, true), mp_int(bits, 1_337, false), mp_int(bits, 9_999, false))] =>
        |(a, b, c): &(MpInt, MpInt, MpInt)| a.mul_add(b, c),
    rug: |bits| vec![(rug_int(bits, 42, true), rug_int(bits, 1_337, false), rug_int(bits, 9_999, false))] =>
        |(a, b, c): &(Integer, Integer, Integer)| Integer::from(a * b) + c,
);
paired_bench!(square, BALANCED,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.square(),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a.square_ref()),
);
paired_bench!(pow, [256, 1_024],
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.pow(17),
    rug: |bits| vec![rug_int(bits, 42, true)] => |a: &Integer| Integer::from(a.pow(17)),
);
paired_bench!(midpoint, NARROW,
    mp: |bits| vec![(mp_int(bits, 42, true), mp_int(bits, 1_337, true))] => |(a, b): &(MpInt, MpInt)| a.midpoint(b),
    rug: |bits| vec![(rug_int(bits, 42, true), rug_int(bits, 1_337, true))] => |(a, b): &(Integer, Integer)| Integer::from(a + b) / 2_u32,
);
paired_bench!(widening_mul, NARROW,
    mp: |bits| vec![(bounded_mp_int(bits, 42, false), bounded_mp_int(bits, 1_337, true), bits)] =>
        |(a, b, _): &(MpInt, MpInt, usize)| a.widening_mul(b),
    rug: |bits| vec![(bounded_rug(bits, 42, false), bounded_rug(bits, 1_337, true), bits)] =>
        |(a, b, bits): &(Integer, Integer, usize)| signed_words(Integer::from(a * b), *bits),
);
paired_bench!(carrying_mul, NARROW,
    mp: |bits| vec![(bounded_mp_int(bits, 42, false), bounded_mp_int(bits, 1_337, true), bounded_mp_int(bits, 9_999, false), bits)] =>
        |(a, b, c, _): &(MpInt, MpInt, MpInt, usize)| a.carrying_mul(b, c),
    rug: |bits| vec![(bounded_rug(bits, 42, false), bounded_rug(bits, 1_337, true), bounded_rug(bits, 9_999, false), bits)] =>
        |(a, b, c, bits): &(Integer, Integer, Integer, usize)| signed_words(Integer::from(a * b) + c, *bits),
);
paired_bench!(carrying_mul_add, NARROW,
    mp: |bits| vec![(bounded_mp_int(bits, 42, false), bounded_mp_int(bits, 1_337, true), bounded_mp_int(bits, 111, false), bounded_mp_int(bits, 222, true), bits)] =>
        |(a, b, c, d, _): &(MpInt, MpInt, MpInt, MpInt, usize)| a.carrying_mul_add(b, c, d),
    rug: |bits| vec![(bounded_rug(bits, 42, false), bounded_rug(bits, 1_337, true), bounded_rug(bits, 111, false), bounded_rug(bits, 222, true), bits)] =>
        |(a, b, c, d, bits): &(Integer, Integer, Integer, Integer, usize)| signed_words(Integer::from(a * b) + c + d, *bits),
);

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn bounded_rug(bits: usize, seed: u32, negative: bool) -> Integer {
    rug_int(bits, seed, negative)
        .keep_signed_bits(u32::try_from(bits).expect("benchmark width fits Rug"))
}
