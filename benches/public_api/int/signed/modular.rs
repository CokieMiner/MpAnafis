//! Signed modular exponentiation on magnitudes, matching `MpInt`'s contract.
//! The Rug reference includes absolute-value conversion for a negative base.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::MODULAR_EXP,
    support::{mp_int, odd_hex, paired_bench},
};

paired_bench!(pow_mod, MODULAR_EXP, samples = (4, 20),
    mp: |bits| vec![(mp_int(bits, 42, true), mp_int(bits, 1_337, false), MpInt::from_str_radix(&odd_hex(bits, 9_999), 16).expect("odd modulus parses"))]
        => |(a, b, m): &(MpInt, MpInt, MpInt)| a.pow_mod(b, m),
    rug: |bits| vec![(rug_int(bits, 42, true), rug_int(bits, 1_337, false), Integer::from_str_radix(&odd_hex(bits, 9_999), 16).expect("odd modulus parses"))]
        => |(a, b, m): &(Integer, Integer, Integer)| Integer::from(a.abs_ref()).pow_mod_ref(b, m).map(Integer::from),
);
