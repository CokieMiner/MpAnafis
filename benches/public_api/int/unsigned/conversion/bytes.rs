//! Endian byte vector serialisation.
//!
//! Both implementations allocate an owned byte vector inside timing. Rug
//! prepares the export length outside timing and writes the GMP digits into
//! that vector.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::Order};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint, paired_bench},
};

paired_bench!(to_be_bytes, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_byte_output => |(value, _): &(MpUint, usize)| value.to_be_bytes(),
    rug: rug_byte_output => |(value, len): &(Integer, usize)| {
        let mut buffer = vec![0_u8; *len];
        value.write_digits(&mut buffer, Order::MsfBe);
        buffer
    },
);
paired_bench!(from_be_bytes, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42).to_be_bytes()] => |bytes: &Vec<u8>| MpUint::from_be_bytes(bytes),
    rug: |bits| vec![rug_uint(bits, 42).to_digits::<u8>(Order::MsfBe)]
        => |bytes: &Vec<u8>| Integer::from_digits(bytes, Order::MsfBe),
);
paired_bench!(to_le_bytes, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: mp_byte_output => |(value, _): &(MpUint, usize)| value.to_le_bytes(),
    rug: rug_byte_output => |(value, len): &(Integer, usize)| {
        let mut buffer = vec![0_u8; *len];
        value.write_digits(&mut buffer, Order::LsfLe);
        buffer
    },
);
paired_bench!(from_le_bytes, NARROW, samples = (SAMPLE_SIZE_FAST, 100),
    mp: |bits| vec![mp_uint(bits, 42).to_le_bytes()] => |bytes: &Vec<u8>| MpUint::from_le_bytes(bytes),
    rug: |bits| vec![rug_uint(bits, 42).to_digits::<u8>(Order::LsfLe)]
        => |bytes: &Vec<u8>| Integer::from_digits(bytes, Order::LsfLe),
);

// Export lengths are prepared outside the timed allocation and byte writes.
fn mp_byte_output(bits: usize) -> Vec<(MpUint, usize)> {
    let value = mp_uint(bits, 42);
    let len = value.significant_bits().div_ceil(8);
    vec![(value, len)]
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_byte_output(bits: usize) -> Vec<(Integer, usize)> {
    let value = rug_uint(bits, 42);
    let len = value.significant_digits::<u8>();
    vec![(value, len)]
}
