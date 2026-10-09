//! Tests and verification for Möller's HGCD2 double-limb quotient simulation.

use proptest::prelude::*;

use super::{Gcd, InternalMpUint, Limb};

#[test]
fn hgcd2_accepts_fibonacci_batches() {
    let (mut u, mut v) = consecutive_fibonacci(8);
    let expected = u.gcd(&v);
    let mut next_u = InternalMpUint::zero();
    let mut next_v = InternalMpUint::zero();

    while u.limbs().len() >= 2 && v.limbs().len() >= 2 {
        let Some((u0, v0, u1, v1)) = Gcd::hgcd2(u.limbs(), v.limbs()) else {
            break;
        };
        assert_eq!(
            u0.wrapping_mul(v1).wrapping_sub(v0.wrapping_mul(u1)),
            1,
            "HGCD2 determinant must be exactly one"
        );
        if !Gcd::lehmer_update_dispatched(
            &mut u,
            &mut v,
            &mut next_u,
            &mut next_v,
            u0,
            v0,
            u1,
            v1,
            true,
            None,
        ) {
            break;
        }
    }

    assert_eq!(u.gcd(&v), expected);
}

#[test]
fn hgcd2_accepts_arbitrary_quotients() {
    // Construct operands with varying quotients (testing q > 1 and last == true retention).
    let quotients = [2_usize, 3, 5, 1, 7, 2, 4];
    let mut v = InternalMpUint::from_limb(0xDEAD_BEEF);
    v.shl_assign(128);
    v.add_assign(&InternalMpUint::from_limb(0xCAFE_BABE));

    let mut u = v.clone();
    for &q in &quotients {
        let next = u.mul(&InternalMpUint::from_limb(q)).add(&v);
        v = u;
        u = next;
    }

    let expected = u.gcd(&v);
    let mut next_u = InternalMpUint::zero();
    let mut next_v = InternalMpUint::zero();

    let mut steps = 0;
    while u.limbs().len() >= 2 && v.limbs().len() >= 2 {
        let Some((u0, v0, u1, v1)) = Gcd::hgcd2(u.limbs(), v.limbs()) else {
            break;
        };
        assert_eq!(
            u0.wrapping_mul(v1).wrapping_sub(v0.wrapping_mul(u1)),
            1,
            "HGCD2 determinant must be exactly one"
        );
        let u_orig = u.clone();
        let v_orig = v.clone();
        let ok = Gcd::lehmer_update_dispatched(
            &mut u,
            &mut v,
            &mut next_u,
            &mut next_v,
            u0,
            v0,
            u1,
            v1,
            true,
            None,
        );
        assert!(ok, "HGCD2 update must succeed on valid batch");

        // Verify exact matrix action: (u_orig, v_orig) = M * (u, v)
        // M = [[v1, v0], [u1, u0]]
        let rec_u = u
            .mul(&InternalMpUint::from_limb(v1))
            .add(&v.mul(&InternalMpUint::from_limb(v0)));
        let rec_v = u
            .mul(&InternalMpUint::from_limb(u1))
            .add(&v.mul(&InternalMpUint::from_limb(u0)));
        assert_eq!(rec_u, u_orig, "Reconstructed u must match original");
        assert_eq!(rec_v, v_orig, "Reconstructed v must match original");

        steps += 1;
    }

    assert!(steps > 0, "HGCD2 should have executed at least one step");
    assert_eq!(u.gcd(&v), expected);
}

proptest! {
    #[test]
    fn hgcd2_matrix_invariants(
        u_limbs in prop::collection::vec(any::<Limb>(), 2..=6),
        v_limbs in prop::collection::vec(any::<Limb>(), 2..=6),
    ) {
        let mut u = InternalMpUint::from_limbs_slice(&u_limbs);
        let mut v = InternalMpUint::from_limbs_slice(&v_limbs);
        if u < v {
            core::mem::swap(&mut u, &mut v);
        }
        if v.is_zero() || u.limbs().len().wrapping_sub(v.limbs().len()) > 1 {
            return Ok(());
        }

        if let Some((u0, v0, u1, v1)) = Gcd::hgcd2(u.limbs(), v.limbs()) {
            let det = u0.wrapping_mul(v1).wrapping_sub(v0.wrapping_mul(u1));
            prop_assert_eq!(det, 1, "HGCD2 determinant must be 1");

            let u_orig = u.clone();
            let v_orig = v.clone();
            let mut next_u = InternalMpUint::zero();
            let mut next_v = InternalMpUint::zero();

            let ok = Gcd::lehmer_update_dispatched(
                &mut u, &mut v, &mut next_u, &mut next_v, u0, v0, u1, v1, true, None,
            );
            prop_assert!(ok, "Lehmer update must succeed for accepted HGCD2 batch");
            prop_assert!(!u.is_zero(), "Reduced u must be non-zero");
            prop_assert!(!v.is_zero(), "Reduced v must be non-zero");

            // Verify exact reconstruction: (u_orig, v_orig) = M * (u, v)
            let rec_u = u.mul(&InternalMpUint::from_limb(v1)).add(&v.mul(&InternalMpUint::from_limb(v0)));
            let rec_v = u.mul(&InternalMpUint::from_limb(u1)).add(&v.mul(&InternalMpUint::from_limb(u0)));
            prop_assert_eq!(rec_u, u_orig);
            prop_assert_eq!(rec_v, v_orig);
        }
    }
}

fn consecutive_fibonacci(target_limbs: usize) -> (InternalMpUint, InternalMpUint) {
    let mut previous = InternalMpUint::zero();
    let mut current = InternalMpUint::one();
    while current.limbs().len() < target_limbs {
        let next = previous.add(&current);
        previous = current;
        current = next;
    }
    (current, previous)
}
