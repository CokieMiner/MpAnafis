//! Width-bounded bit rearrangement, with composed Rug references.
//! Reversal exports bytes, reverses each byte's bits, and imports the opposite
//! byte order. Rotations include shift, mask, and combine in the timed operation.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::Order};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{mp_uint, paired_bench},
};

paired_bench!(reverse_bits, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), bits)] => |(a, bits): &(MpUint, usize)| a.reverse_bits(*bits),
    rug: |bits| vec![(rug_uint(bits, 42), bits)] => |(a, bits): &(Integer, usize)| {
        let mut bytes = vec![0_u8; bits.div_ceil(8)];
        a.write_digits(&mut bytes, Order::MsfBe);
        for byte in &mut bytes { *byte = byte.reverse_bits(); }
        Some(Integer::from_digits(&bytes, Order::LsfLe))
    },
);
paired_bench!(rotate_left, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), bits)] => |(a, bits): &(MpUint, usize)| a.rotate_left(17, *bits),
    rug: |bits| vec![(rug_uint(bits, 42), bits)] => |(a, bits): &(Integer, usize)| {
        let width = u32::try_from(*bits).expect("width fits Rug");
        let counter = width.checked_sub(17).expect("width exceeds rotation");
        Some((Integer::from(a << 17_u32) | Integer::from(a >> counter)).keep_bits(width))
    },
);
paired_bench!(rotate_right, NARROW,
    mp: |bits| vec![(mp_uint(bits, 42), bits)] => |(a, bits): &(MpUint, usize)| a.rotate_right(17, *bits),
    rug: |bits| vec![(rug_uint(bits, 42), bits)] => |(a, bits): &(Integer, usize)| {
        let width = u32::try_from(*bits).expect("width fits Rug");
        let counter = width.checked_sub(17).expect("width exceeds rotation");
        Some((Integer::from(a >> 17_u32) | Integer::from(a << counter)).keep_bits(width))
    },
);
paired_bench!(swap_bytes, NARROW,
    mp: |bits| vec![mp_uint(bits, 42)] => |a: &MpUint| a.swap_bytes(),
    rug: |bits| vec![rug_uint(bits, 42)] => |a: &Integer| Integer::from_digits(&a.to_digits::<u8>(Order::MsfBe), Order::LsfLe),
);
