//! Division identities, retained storage, and scalar modular certificates.

use super::{Division, Gcd, InternalMpUint, Limb};

#[test]
fn single_limb_assignment_reuses_storage_with_exact_overlap() {
    for width in [0, 1, 2, 4, 5, 33, 64] {
        let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for shift in 0..Limb::BITS {
            for divisor in [1_usize << shift, Limb::MAX >> shift] {
                let denominator = InternalMpUint::from_limb(divisor);
                let mut expected = InternalMpUint::zero();
                let remainder =
                    Division::div_rem_1::<true>(numerator.limbs(), divisor, &mut expected);
                let mut destination = InternalMpUint::with_capacity(128);
                destination.clone_from(&numerator);
                let capacity = destination.capacity();
                let pointer = destination.limbs().as_ptr();
                destination.div_assign(&denominator);
                assert_eq!(destination, expected);
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), pointer);
                destination.clone_from(&numerator);
                destination.rem_assign(&denominator);
                assert_eq!(destination, InternalMpUint::from_limb(remainder));
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), pointer);
            }
        }
    }
}

#[test]
fn value_and_assignment_identities_preserve_retained_storage() {
    let one = InternalMpUint::one();
    let zero = InternalMpUint::zero();
    let larger = InternalMpUint::from_limbs((1..=80).collect());
    for width in [0, 1, 2, 4, 5, 64] {
        let value = InternalMpUint::from_limbs((1..=width).collect());
        for (divisor, quotient, remainder) in [
            (&one, &value, &zero),
            (&larger, &zero, &value),
            (&value, &one, &zero),
        ] {
            if divisor.is_zero() {
                continue;
            }
            assert_eq!(&value.div(divisor), quotient);
            assert_eq!(&value.rem(divisor), remainder);
            let mut destination = InternalMpUint::with_capacity(128);
            destination.clone_from(&value);
            let capacity = destination.capacity();
            let pointer = destination.limbs().as_ptr();
            destination.div_assign(divisor);
            assert_eq!(&destination, quotient);
            assert_eq!(destination.capacity(), capacity);
            assert_eq!(destination.limbs().as_ptr(), pointer);
            destination.clone_from(&value);
            destination.rem_assign(divisor);
            assert_eq!(&destination, remainder);
            assert_eq!(destination.capacity(), capacity);
            assert_eq!(destination.limbs().as_ptr(), pointer);
        }
    }
}

#[test]
fn modular_inverse_limb_contract() {
    // All supported limb widths are even, so (B - 1)/3 is 0x55...55.
    let alternating = Limb::MAX.div_euclid(3);
    for odd in [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        17,
        31,
        127,
        255,
        257,
        65535,
        #[cfg(not(target_pointer_width = "16"))]
        65_537,
        Limb::MAX,
        Limb::MAX - 2, // odd
        (Limb::MAX >> 1) | 1,
        alternating,
        (alternating << 1) | 1,
    ] {
        let inv = Division::modular_inverse_limb(odd);
        assert_eq!(
            odd.wrapping_mul(inv),
            1,
            "modular inverse of {odd:#x} failed: inv = {inv:#x}"
        );
    }
    for low in (1_usize..=255).step_by(2) {
        for high in [0, 256, 1 << Limb::BITS.wrapping_sub(1), Limb::MAX ^ 255] {
            let odd = high | low;
            assert_eq!(odd.wrapping_mul(Division::modular_inverse_limb(odd)), 1);
        }
    }
}

#[test]
fn modexact_1_odd_divisibility_and_gcd_identities() {
    for d in [
        1,
        3,
        5,
        7,
        17,
        257,
        #[cfg(not(target_pointer_width = "16"))]
        65_537,
        Limb::MAX,
    ] {
        for width in [0, 1, 2, 3, 4, 8, 17] {
            // Exact multiples give zero in the radix-inverse residue.
            let mut num = InternalMpUint::from_limb(d);
            let factor = InternalMpUint::from_limbs((1..=width).collect());
            num.mul_assign(&factor);
            let r = Division::modexact_1_odd(num.limbs(), d);
            assert_eq!(r, 0, "exact multiple should produce r == 0");

            // Nonzero residues preserve gcd(d, d*k+r) = gcd(d, r).
            if d > 1 {
                for add in 1..d.min(5) {
                    let mut num_plus = num.clone();
                    num_plus.add_assign(&InternalMpUint::from_limb(add));
                    let r_plus = Division::modexact_1_odd(num_plus.limbs(), d);
                    assert_ne!(r_plus, 0, "non-multiple should produce r != 0");
                    let gcd_from_r = Gcd::gcd_1(d, r_plus);
                    let gcd_from_div = Gcd::gcd_1(d, add);
                    assert_eq!(gcd_from_r, gcd_from_div, "gcd(d, r) must match gcd(d, add)");
                }
            }
        }
    }
}

#[test]
fn modexact_1_odd_matches_hardware_division_remainder() {
    // Deterministic generator staying in native Limb width on every
    // pointer-width target, so no narrowing casts are required.
    let mut state: Limb = 0x9E37;
    let mut next_limb = || {
        state = state.wrapping_mul(33).wrapping_add(0x85EB);
        state ^ state.wrapping_shr(Limb::BITS.wrapping_div(2))
    };

    for d in [3, 5, 17, 257, 0x7FFF, Limb::MAX.wrapping_sub(2), Limb::MAX] {
        for width in [1, 2, 3, 5, 8] {
            for _ in 0..32_u32 {
                let mut limbs = alloc::vec::Vec::with_capacity(width);
                for _ in 0..width {
                    limbs.push(next_limb());
                }
                let num = InternalMpUint::from_limbs(limbs);
                let r = Division::modexact_1_odd(num.limbs(), d);
                let mut dummy = InternalMpUint::zero();
                let rem = Division::div_rem_1::<false>(num.limbs(), d, &mut dummy);
                assert_eq!(
                    r == 0,
                    rem == 0,
                    "modexact divisibility must agree with div_rem_1 for d = {d:#x}"
                );
                assert_eq!(
                    Gcd::gcd_1(d, r),
                    Gcd::gcd_1(d, rem),
                    "gcd(d, r) must agree with gcd(d, rem) for d = {d:#x}"
                );
            }
        }
    }
}
