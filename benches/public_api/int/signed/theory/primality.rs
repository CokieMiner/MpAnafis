//! Probable-prime classification; Mp and GMP use different witness policies.

use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::IsPrime};

use crate::int::support::{odd_hex, paired_bench};

paired_bench!(is_probably_prime, [256, 1_024], samples = (4, 20),
    mp: |bits| vec![MpInt::from_str_radix(&odd_hex(bits, 42), 16).expect("odd hexadecimal parses")]
        => |a: &MpInt| a.is_probably_prime(24),
    rug: |bits| vec![Integer::from_str_radix(&odd_hex(bits, 42), 16).expect("odd hexadecimal parses")]
        => |a: &Integer| a.is_probably_prime(24) != IsPrime::No,
);
