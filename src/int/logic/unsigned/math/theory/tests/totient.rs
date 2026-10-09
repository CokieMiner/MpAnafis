//! Euler totient from residue counts and independent prime factorizations.

use proptest::prelude::*;

use super::super::InternalMpUint;

#[test]
#[cfg_attr(
    miri,
    ignore = "Factor searches on 89-bit prime powers require native execution; independently factored small values run under Miri."
)]
fn totient_factor_search_and_square_reduction_cover_wide_cofactors() {
    let prime = InternalMpUint::one().shl(89).sub(&InternalMpUint::one());
    assert!(prime.is_prime(), "2^89-1 is a prime fixture");
    let prime_minus_one = prime.sub(&InternalMpUint::one());
    for exponent in [2_u32, 4] {
        assert_eq!(
            prime.pow(exponent).euler_phi(),
            Some(prime.pow(exponent.wrapping_sub(1)).mul(&prime_minus_one))
        );
    }
    for small in [57_719, 65_537, 104_729] {
        let factor = InternalMpUint::from_u64(small);
        let factor_minus_one = factor.sub(&InternalMpUint::one());
        for exponent in [1_u32, 2, 3] {
            let value = prime.mul(&factor.pow(exponent));
            let expected = prime_minus_one
                .mul(&factor.pow(exponent.wrapping_sub(1)))
                .mul(&factor_minus_one);
            assert_eq!(value.euler_phi(), Some(expected));
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Wide repeated-factor and Pollard splitting matrices require native execution; fused trial carry properties run under Miri."
)]
fn totient_handles_repeated_factors_and_prime_cofactors() {
    for (prime, degree) in [(2, 500), (3, 100), (5, 80), (7, 60), (251, 25)] {
        let p = InternalMpUint::from_limb(prime);
        let n = p.pow(degree);
        let expected = p.pow(degree - 1).mul(&InternalMpUint::from_limb(prime - 1));
        assert_eq!(n.euler_phi(), Some(expected));
    }
    let two_power = InternalMpUint::from_limb(2).pow(80);
    let three_power = InternalMpUint::from_limb(3).pow(40);
    let value = two_power
        .mul(&three_power)
        .mul(&InternalMpUint::from_limb(101));
    let expected = two_power
        .mul(&InternalMpUint::from_limb(3).pow(39))
        .mul(&InternalMpUint::from_limb(100));
    assert_eq!(value.euler_phi().as_ref(), Some(&expected));

    // After the trial budget, square reduction or Pollard splitting resolves
    // these cofactors. Equal factors must contribute (p-1) only once.
    for (first, second) in [
        (65_537_u64, 65_537_u64),
        (104_729, 1_299_709),
        (57_719, 60_013),
        (5_771_401, 459_040_441),
        // Products immediately around B/9 and B/4 on 64-bit targets
        // exercise redundant products, deferred additions and full reduction.
        (50_021, 40_975_554_875_179),
        (50_021, 40_975_554_875_201),
        (50_021, 92_194_998_469_169),
        (50_021, 92_194_998_469_219),
        (4_294_967_291, 4_294_967_291),
        (4_294_967_291, 4_294_967_279),
    ] {
        let a = InternalMpUint::from_u64(first);
        let b = InternalMpUint::from_u64(second);
        assert!(
            a.is_prime() && b.is_prime(),
            "the fixture factors are prime"
        );
        let expected_phi = if first == second {
            a.mul(&b.sub(&InternalMpUint::one()))
        } else {
            a.sub(&InternalMpUint::one())
                .mul(&b.sub(&InternalMpUint::one()))
        };
        // The wide trial path removes 2, 3 and 101 completely before the
        // same recursive tail. Its accumulator must not apply them twice.
        let combined = a.mul(&b).mul(&value);
        let combined_phi = expected_phi.mul(&expected);
        assert_eq!(a.mul(&b).euler_phi(), Some(expected_phi));
        assert_eq!(combined.euler_phi(), Some(combined_phi));
    }
}

#[test]
fn totient_recomputes_trial_bounds_after_factor_removal() {
    let first_factors = if cfg!(miri) {
        &[50_021_u64][..]
    } else {
        &[313_u64, 317, 331, 49_999, 50_021][..]
    };
    let second_factors = if cfg!(miri) {
        &[50_023_u64][..]
    } else {
        &[331_u64, 49_999, 50_021, 50_023][..]
    };
    for &first in first_factors {
        let a = InternalMpUint::from_u64(first);
        assert!(a.is_prime(), "trial-bound fixtures are prime");
        for &second in second_factors {
            let b = InternalMpUint::from_u64(second);
            assert!(b.is_prime(), "trial-bound fixtures are prime");
            for exponent in 1_u32..=if cfg!(miri) { 1 } else { 3 } {
                let power = a.pow(exponent);
                let value = power.mul(&b);
                let expected = if first == second {
                    power.mul(&a.sub(&InternalMpUint::one()))
                } else {
                    a.pow(exponent - 1)
                        .mul(&a.sub(&InternalMpUint::one()))
                        .mul(&b.sub(&InternalMpUint::one()))
                };
                assert_eq!(value.euler_phi(), Some(expected));
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 48 }))]

    #[test]
    #[expect(clippy::arithmetic_side_effects, reason = "Each totient quotient is at most the accumulator; trial candidates increase only through sqrt(u32::MAX)+1.")]
    fn totient_trial_inverses_match_primitive_factorization(value in 1_u32..=if cfg!(miri) { u32::from(u16::MAX) } else { u32::MAX }) {
        let mut remaining = value;
        let mut expected = value;
        let mut candidate = 2_u32;
        while candidate <= remaining.div_euclid(candidate) {
            if remaining.rem_euclid(candidate) == 0 {
                expected -= expected.div_euclid(candidate);
                while remaining.rem_euclid(candidate) == 0 {
                    remaining = remaining.div_euclid(candidate);
                }
            }
            candidate += 1;
        }
        if remaining > 1 {
            expected -= expected.div_euclid(remaining);
        }
        prop_assert_eq!(
            InternalMpUint::from_u64(u64::from(value)).euler_phi(),
            Some(InternalMpUint::from_u64(u64::from(expected)))
        );
    }

    #[test]
    #[expect(clippy::arithmetic_side_effects, reason = "Prime factors and generated degrees are positive; the power-of-two exponent is reduced only when nonzero.")]
    fn totient_fused_trial_updates_preserve_wide_carries(
        first in proptest::sample::select(if cfg!(miri) { alloc::vec![3_usize, 7, 251] } else { alloc::vec![3_usize, 7, 251, 32_749, 49_999] }),
        degree in 1_u32..=if cfg!(miri) { 3 } else { 12 },
        second_degree in 1_u32..=if cfg!(miri) { 3 } else { 12 },
        twos in 0_usize..=if cfg!(miri) { 80 } else { 300 },
    ) {
        let p = InternalMpUint::from_limb(first);
        let q = InternalMpUint::from_limb(101);
        let value = p.pow(degree).mul(&q.pow(second_degree)).shl(twos);
        let mut expected = p.pow(degree - 1)
            .mul(&InternalMpUint::from_limb(first - 1))
            .mul(&q.pow(second_degree - 1))
            .mul(&InternalMpUint::from_limb(100));
        if twos != 0 {
            expected.shl_assign(twos - 1);
        }
        prop_assert_eq!(value.euler_phi(), Some(expected));
    }

    /// Counts the residues coprime to n, with phi(0) undefined and phi(1) = 1.
    #[test]
    #[expect(clippy::arithmetic_side_effects, reason = "The residue count is at most value-1 <= 1999 and therefore fits u16.")]
    fn euler_phi_prop(value in 0_u16..=2_000) {
        let n = InternalMpUint::from_u64(u64::from(value));
        let Some(phi) = n.euler_phi() else {
            prop_assert!(n.is_zero(), "euler_phi is only undefined at zero");
            return Ok(());
        };
        prop_assert!(!n.is_zero());

        if n.is_one() {
            prop_assert!(phi.is_one());
            return Ok(());
        }
        // For every n > 1, `0 < phi(n) < n`, and phi(n) is by definition the
        // number of residues in `1..n` coprime to n.
        prop_assert!(!phi.is_zero());
        prop_assert!(phi < n);

        let mut coprime_count = 0_u16;
        for candidate in 1..value {
            let (mut a, mut b) = (candidate, value);
            while b != 0 {
                (a, b) = (b, a.rem_euclid(b));
            }
            if a == 1 {
                coprime_count += 1;
            }
        }
        prop_assert_eq!(phi, InternalMpUint::from_u64(u64::from(coprime_count)));
    }
}
