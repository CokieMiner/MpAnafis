//! Identical minimal signed two's-complement byte encodings.

use mp_anafis::MpInt;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{from_signed_be_bytes, rug_int, signed_be_bytes};
use crate::int::{
    ladders::NARROW,
    support::{mp_int, paired_bench},
};

paired_bench!(to_be_bytes, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true)] => |a: &MpInt| a.to_be_bytes(),
    rug: |bits| vec![rug_int(bits, 42, true)] => signed_be_bytes,
);
paired_bench!(from_be_bytes, NARROW,
    mp: |bits| vec![mp_int(bits, 42, true).to_be_bytes()] => |a: &Vec<u8>| MpInt::from_be_bytes(a),
    rug: |bits| vec![signed_be_bytes(&rug_int(bits, 42, true))] => |a: &Vec<u8>| from_signed_be_bytes(a),
);
