//! Primality and strict prime searches on identical deterministic inputs.
//!
//! Mp's fixed-base policy and GMP's probable-prime policy differ. The comparison
//! checks classifications for these fixtures; equal requested counts do not imply
//! equal witness sequences, work, or probabilistic guarantees.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::IsPrime};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_known_primes, rug_semiprimes_no_small_factors};
use crate::int::{
    ladders::PRIMALITY,
    support::{mp_known_primes, mp_semiprimes_no_small_factors, odd_hex, paired_bench},
};

paired_bench!(is_probably_prime_random, PRIMALITY, samples = (4, 20),
    mp: |bits| vec![mp_odd(bits)] => |a: &MpUint| a.is_probably_prime(24),
    rug: |bits| vec![rug_odd(bits)] => |a: &Integer| a.is_probably_prime(24) != IsPrime::No,
);
paired_bench!(is_probably_prime_known_prime, PRIMALITY, samples = (4, 20),
    mp: mp_known_primes => |a: &MpUint| a.is_probably_prime(24),
    rug: rug_known_primes => |a: &Integer| a.is_probably_prime(24) != IsPrime::No,
);
paired_bench!(is_probably_prime_semiprime, PRIMALITY, samples = (4, 20),
    mp: mp_semiprimes_no_small_factors => |a: &MpUint| a.is_probably_prime(24),
    rug: rug_semiprimes_no_small_factors => |a: &Integer| a.is_probably_prime(24) != IsPrime::No,
);
paired_bench!(is_prime, PRIMALITY, samples = (4, 20),
    mp: mp_known_primes => |a: &MpUint| a.is_prime(),
    rug: rug_known_primes => |a: &Integer| a.is_probably_prime(24) != IsPrime::No,
);
paired_bench!(next_prime, PRIMALITY, samples = (4, 20),
    mp: |bits| vec![mp_odd(bits)] => |a: &MpUint| a.next_prime(),
    rug: |bits| vec![rug_odd(bits)] => |a: &Integer| Some(a.clone().next_prime()),
);
paired_bench!(prev_prime, PRIMALITY, samples = (4, 20),
    mp: |bits| vec![mp_odd(bits)] => |a: &MpUint| a.prev_prime(),
    rug: |bits| vec![rug_odd(bits)] => |a: &Integer| Some(a.clone().prev_prime()),
);

fn mp_odd(bits: usize) -> MpUint {
    MpUint::from_str_radix(&odd_hex(bits, 42), 16).expect("generated hexadecimal parses")
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_odd(bits: usize) -> Integer {
    Integer::from_str_radix(&odd_hex(bits, 42), 16).expect("generated hexadecimal parses")
}
