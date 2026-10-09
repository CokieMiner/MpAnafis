//! Exact native primality and wide prime/search fixtures.

use mp_anafis::{MpInt, MpUint};
use rug::{
    Integer,
    integer::{IsPrime, Order},
};

use crate::TheoryReference;

#[test]
#[cfg_attr(
    miri,
    ignore = "Primality comparisons use GMP native FFI unavailable to Miri"
)]
fn native_primality_agrees_with_gmp_at_exponent_boundaries() {
    for bit in 6..usize::BITS {
        for offset in 0..=64 {
            for value in [(1_usize << bit) - offset, (1_usize << bit) + offset] {
                assert_eq!(
                    MpUint::from(value).is_prime(),
                    Integer::from(value).is_probably_prime(30) != IsPrime::No,
                    "n={value}"
                );
            }
        }
    }
    for offset in 0..=128 {
        let value = usize::MAX - offset;
        assert_eq!(
            MpUint::from(value).is_prime(),
            Integer::from(value).is_probably_prime(30) != IsPrime::No,
            "n={value}"
        );
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Wide primality comparisons use GMP native FFI unavailable to Miri"
)]
fn wide_primality_checks_qualification_search_order_and_square_rejection() {
    // Published Mersenne primes provide exact positive cases above u64::MAX.
    // https://www.mersenne.org/primes/
    for exponent in [127_u32, 521] {
        let oracle = (Integer::from(1) << exponent) - 1_u32;
        let prime = MpUint::from_le_bytes(&oracle.to_digits::<u8>(Order::Lsf));
        let signed = MpInt::from(prime.clone());
        assert!(prime.is_prime(), "Mersenne exponent {exponent}");
        assert!(signed.is_prime(), "Mersenne exponent {exponent}");
        for rounds in [0, 1, 64, 65] {
            assert!(prime.is_probably_prime(rounds));
            assert!(signed.is_probably_prime(rounds));
        }
    }
    for bits in [65_u32, 127, 255, 256, 257, 511, 1024, 2048] {
        let mut oracle = Integer::from(1) << bits;
        oracle.next_prime_mut();
        let prime = MpUint::from_le_bytes(&oracle.to_digits::<u8>(Order::Lsf));
        let classified = prime.is_prime();
        if oracle.is_probably_prime(30) == IsPrime::Yes {
            assert!(classified);
        }
        if classified {
            assert!(TheoryReference::probably_prime(&oracle, 1));
        }
        let preceding = prime.checked_sub(&MpUint::from(2_u8)).unwrap();
        let next = preceding.next_prime().unwrap();
        assert!(next > preceding);
        assert!(next.is_prime());
        let signed_next = MpInt::from(preceding.clone()).next_prime().unwrap();
        assert!(signed_next > preceding);
        assert!(signed_next.is_prime());
        let previous = prime.prev_prime().unwrap();
        assert!(previous < prime);
        assert!(previous.is_prime());
        assert!(!prime.square().is_prime(), "prime square at {bits} bits");
        for offset in [2_u8, 4, 6, 8, 10, 12, 30, 100] {
            let candidate = &prime + MpUint::from(offset);
            let reference = Integer::from_digits(&candidate.to_le_bytes(), Order::Lsf);
            let classified = candidate.is_prime();
            if reference.is_probably_prime(30) == IsPrime::Yes {
                assert!(classified);
            }
            if classified {
                assert!(TheoryReference::probably_prime(&reference, 1));
            }
        }
        for rounds in [0, 1, 12, 64, 65] {
            assert_eq!(
                prime.is_probably_prime(rounds),
                TheoryReference::probably_prime(&oracle, rounds)
            );
        }
    }
    // This composite passes the first twelve prime Miller–Rabin bases.
    let candidate = MpUint::from_str_radix("318665857834031151167461", 10).unwrap();
    assert!(candidate.is_probably_prime(12));
    assert!(!candidate.is_prime());
}
