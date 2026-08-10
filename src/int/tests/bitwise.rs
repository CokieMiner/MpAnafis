//! Unsigned and signed bitwise algebra properties.

use super::*;

proptest! {
    #[test]
    fn prop_bitop_and_or_absorption(a in strategies::uint(8), b in strategies::uint(8)) {
        let and = &a & &b;
        let or = &a | &b;
        prop_assert!((&and | &or) == or, "and | or != or");
        prop_assert!((&and & &or) == and, "and & or != and");
    }
}

proptest! {
    #[test]
    fn prop_bitop_xor_cancel(a in strategies::uint(8), b in strategies::uint(8)) {
        prop_assert_eq!(&(&a ^ &b) ^ &b, a, "(a xor b) xor b != a");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_idempotency(a in strategies::int(16)) {
        prop_assert!((&a & &a) == a, "bitwise AND idempotency");
        prop_assert_eq!(&a | &a, a, "bitwise OR idempotency");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_commutativity(a in strategies::int(16), b in strategies::int(16)) {
        prop_assert_eq!(&a & &b, &b & &a, "bitwise AND commutativity");
        prop_assert_eq!(&a | &b, &b | &a, "bitwise OR commutativity");
        prop_assert_eq!(&a ^ &b, &b ^ &a, "bitwise XOR commutativity");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_associativity(a in strategies::int(8), b in strategies::int(8), c in strategies::int(8)) {
        prop_assert_eq!(&(&a ^ &b) ^ &c, &a ^ &(&b ^ &c), "bitwise XOR associativity");
        prop_assert_eq!(&(&a & &b) & &c, &a & &(&b & &c), "bitwise AND associativity");
        prop_assert_eq!(&(&a | &b) | &c, &a | &(&b | &c), "bitwise OR associativity");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_cancellation(a in strategies::int(16)) {
        prop_assert_eq!(&a ^ &a, ArbiInt::zero(), "bitwise XOR cancellation");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_not_identity(a in strategies::int(16)) {
        prop_assert_eq!(!&a, -&a - ArbiInt::one(), "bitwise NOT identity (~a == -a - 1)");
    }
}

proptest! {
    #[test]
    fn prop_signed_bitwise_counts_and_swap_bytes(
        bits in 8_usize..=128,
        input_a in strategies::bounded_int_wrapped(128),
    ) {
        let bounded_a = ArbiInt {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };

        let ones = bounded_a
            .count_ones()
            .expect("bounded precision has count_ones");
        let zeros = bounded_a
            .count_zeros()
            .expect("bounded precision has count_zeros");
        prop_assert_eq!(
            ones + zeros,
            bits,
            "ones + zeros should equal precision for bounded ArbiInt"
        );

        let trailing_zeros = bounded_a.trailing_zeros();
        let trailing_ones = bounded_a
            .trailing_ones()
            .expect("bounded precision has trailing_ones");
        if trailing_zeros > 0 {
            prop_assert_eq!(trailing_ones, 0, "if trailing zeros > 0, trailing ones must be 0");
        } else if !bounded_a.is_zero() {
            prop_assert!(
                trailing_ones > 0,
                "if trailing zeros == 0 and non-zero, trailing ones must be > 0"
            );
        }

        let swapped = bounded_a
            .swap_bytes()
            .expect("bounded precision has swap_bytes");
        let swapped_back = swapped.swap_bytes().expect("swapped has swap_bytes");
        if bits.is_multiple_of(8) {
            prop_assert_eq!(
                &bounded_a,
                &swapped_back,
                "swap_bytes should be involutive when bit width is a multiple of 8"
            );
        }

        let mut unlimited_a = bounded_a;
        unlimited_a.precision = Precision::Unlimited;
        prop_assert!(
            unlimited_a.swap_bytes().is_none(),
            "unlimited precision swap_bytes should return None"
        );
    }
}

proptest! {
    #[test]
    fn prop_bit_scanning(a in strategies::uint(8)) {
        let tz = a.trailing_zeros();
        let ffs = a.find_first_set_bit();
        if !a.is_zero() {
            prop_assert_eq!(tz, ffs.expect("test expects non-zero value for ffs"));
            let first_set = ffs.expect("ffs");
            prop_assert!(a.get_bit(first_set));
            let b = a.set_bit_to(first_set, false);
            let next_set = a.find_next_set_bit(first_set + 1);
            let b_ffs = b.find_first_set_bit();
            prop_assert_eq!(next_set, b_ffs);
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_swap_bytes_old_compat(bit_width in 8_usize..=256) {
        let mut value = ArbiUint::one() << bit_width.saturating_sub(1);
        let byte_mask = ArbiUint::from(255_u64);
        if (&value & &byte_mask).value.is_zero() {
            value |= ArbiUint::one();
        }
        let swapped = value.swap_bytes();
        let swapped_back = swapped.swap_bytes();
        prop_assert_eq!(value, swapped_back, "swap_bytes should be involutive");
    }
}
