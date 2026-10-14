use crate::{LIMB_BITS, MAX_PRIMES, NttPlan, PRIME_CATALOG, PlannedProfile, profile::NttProfile};

use super::planning::forced;

#[test]
fn native_plans_cover_profiles_shapes_and_reject_unaddressable_geometry() {
    for count in 1..=MAX_PRIMES {
        let chunk_bits = u32::try_from(
            count
                .checked_mul(49)
                .expect("small catalog")
                .checked_sub(16)
                .expect("positive capacity")
                .div_euclid(2),
        )
        .expect("small coefficient width");
        let profile = NttProfile {
            np: count,
            chunk_bits,
            max_transform_len: usize::MAX,
        };
        for (left, right) in [(1, 1), (3, 7), (4, 4), (16, 65), (257, 256)] {
            let plan = forced(left, right, profile).expect("conservative forced profile");
            assert!(plan.required_mul_scratch_limbs() >= plan.required_sqr_scratch_limbs());
        }
    }
    for (left, right, bits) in [
        (0, 1, LIMB_BITS),
        (1, 0, LIMB_BITS),
        (1, 1, 0),
        (usize::MAX, 1, LIMB_BITS),
        (usize::MAX.div_euclid(64), usize::MAX.div_euclid(64), 64),
    ] {
        assert!(PlannedProfile::select(left, right, bits).is_none());
        if bits == LIMB_BITS {
            assert!(NttPlan::for_multiplication(left, right).is_none());
        }
    }
    for profile in [
        NttProfile {
            np: 0,
            chunk_bits: 16,
            max_transform_len: 4,
        },
        NttProfile {
            np: MAX_PRIMES.checked_add(1).expect("catalog count"),
            chunk_bits: 16,
            max_transform_len: usize::MAX,
        },
        NttProfile {
            np: 1,
            chunk_bits: 50,
            max_transform_len: usize::MAX,
        },
        NttProfile {
            np: 1,
            chunk_bits: 0,
            max_transform_len: usize::MAX,
        },
    ] {
        assert!(forced(2, 2, profile).is_none());
    }
    for prime in PRIME_CATALOG {
        assert_eq!(
            prime.two_valuation(),
            prime
                .modulus()
                .checked_sub(1)
                .expect("positive prime")
                .trailing_zeros()
        );
        for log in [1, 2, 8, prime.two_valuation()] {
            let root = prime.primitive_root_of_unity(log);
            assert_eq!(
                prime.pow_mod(root, 1_u64.checked_shl(log).expect("catalog order")),
                1
            );
            assert_eq!(
                prime.pow_mod(
                    root,
                    1_u64
                        .checked_shl(log.checked_sub(1).expect("positive log"))
                        .expect("catalog half-order")
                ),
                prime.modulus().checked_sub(1).expect("positive prime")
            );
        }
    }
}
