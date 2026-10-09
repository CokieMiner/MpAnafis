//! Signed GCD families on mixed and negative operands.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int_pairs;
use crate::int::{
    ladders::{EXTENDED_GCD, GCD},
    support::{mp_int_pairs, paired_bench},
};

paired_bench!(gcd, GCD, samples = (1, 15),
    mp: |bits| mp_int_pairs(bits, true, true) => |(a, b): &(MpInt, MpInt)| a.gcd(b),
    rug: |bits| rug_int_pairs(bits, true, true) => |(a, b): &(Integer, Integer)| Integer::from(a.gcd_ref(b)),
);
paired_bench!(extended_gcd, EXTENDED_GCD, samples = (1, 15),
    mp: |bits| mp_int_pairs(bits, true, false) => |(a, b): &(MpInt, MpInt)| a.extended_gcd(b),
    rug: |bits| rug_int_pairs(bits, true, false) => |(a, b): &(Integer, Integer)| Some(a.clone().extended_gcd(b.clone(), Integer::new())),
);
