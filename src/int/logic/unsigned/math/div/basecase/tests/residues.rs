//! Scalar modular inverses and radix-inverse residue certificates.

use super::{Division, DoubleLimb, Gcd, InternalMpUint, Limb};

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
                assert!(r < d, "the modular residue must be canonical");
                let exponent = if num.limbs().len() <= 1 {
                    0
                } else if num.limbs().last().is_some_and(|&top| top <= d) {
                    num.limbs().len().checked_sub(1).expect("multiple limbs")
                } else {
                    num.limbs().len()
                };
                let wide_divisor = DoubleLimb::try_from(d).expect("a limb fits its double width");
                let mut scaled = DoubleLimb::try_from(r).expect("a limb fits its double width");
                for _ in 0..exponent {
                    // scaled < d < B bounds scaled*B below the double-limb radix.
                    scaled = scaled
                        .checked_shl(Limb::BITS)
                        .expect("sub-double shift")
                        .checked_rem(wide_divisor)
                        .expect("positive divisor");
                }
                let negative_remainder = if rem == 0 {
                    0
                } else {
                    wide_divisor
                        .checked_sub(DoubleLimb::try_from(rem).expect("native remainder"))
                        .expect("the remainder is below the divisor")
                };
                assert_eq!(scaled, negative_remainder, "r*B^k == -A (mod d)");
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
