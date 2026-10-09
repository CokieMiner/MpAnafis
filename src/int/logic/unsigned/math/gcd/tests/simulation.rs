//! Quotient simulation, truncation admission, and exact Euclidean prefixes.

use alloc::vec::Vec;

use proptest::prelude::*;

use crate::int::logic::unsigned::math::gcd::lehmer_simulation::lehmer_simulate_wide;

use super::{DoubleLimb, Gcd, HgcdWorkspace, InternalMpUint, LIMB_BITS, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 4_096 }))]
    #[test]
    fn narrow_simulation_policies_follow_exact_quotients(
        first in any::<Limb>(), second in any::<Limb>(),
    ) {
        let head = first.max(second);
        let tail = first.min(second);
        let mut exact_head = head;
        let mut exact_tail = tail;
        let mut accepted = Vec::new();
        let masked = Gcd::lehmer_simulate::<true>(head, tail, |q| {
            assert_eq!(q, exact_head.checked_div(exact_tail).expect("accepted divisor nonzero"));
            let remainder = exact_head.checked_rem(exact_tail).expect("accepted divisor nonzero");
            exact_head = exact_tail;
            exact_tail = remainder;
            accepted.push(q);
        });
        let mut branched_quotients = Vec::new();
        let branched = Gcd::lehmer_simulate::<false>(head, tail, |q| branched_quotients.push(q));
        prop_assert_eq!(masked, branched);
        prop_assert_eq!(accepted, branched_quotients);
    }

    #[test]
    fn wide_simulation_policies_produce_the_same_transition(
        u_hat in any::<DoubleLimb>(), v_hat in any::<DoubleLimb>(),
    ) {
        prop_assert_eq!(lehmer_simulate_wide::<true>(u_hat, v_hat, |_| {}),
            lehmer_simulate_wide::<false>(u_hat, v_hat, |_| {}));
    }
}

#[test]
fn simulated_batch_rejects_an_adverse_truncation_error() {
    // U_hat=3*V_hat+1 with V_hat=B/4 has simulated remainder one.
    // The adverse coefficient is three, so this batch cannot certify
    // the full quotient two for U=U_hat*B^4, V=V_hat*B^4+B^4-1.
    let v_hat = (Limb::MAX >> 2).checked_add(1).expect("quarter radix");
    let u_hat = v_hat
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .expect("three-quarter radix");
    assert_eq!(
        Gcd::lehmer_simulate::<false>(u_hat, v_hat, |_| {}),
        (1, 0, 0, 1, true)
    );
    let wide_half = DoubleLimb::try_from(v_hat).expect("limb fits double width");
    let wide_v = (wide_half << LIMB_BITS) | wide_half;
    let wide_u = wide_v
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .expect("bounded double-limb window");
    assert_eq!(
        lehmer_simulate_wide::<false>(wide_u, wide_v, |_| {}),
        (1, 0, 0, 1, true)
    );
    let u = InternalMpUint::from_limbs(alloc::vec![0, 0, 0, 0, u_hat]);
    let v = InternalMpUint::from_limbs(alloc::vec![
        Limb::MAX,
        Limb::MAX,
        Limb::MAX,
        Limb::MAX,
        v_hat
    ]);
    let mut workspace = HgcdWorkspace::default();
    assert_eq!(u.gcd(&v), Gcd::compute_lehmer(&u, &v));
    assert_eq!(
        Gcd::compute_half_gcd(&u, &v, &mut workspace),
        Gcd::compute_lehmer(&u, &v)
    );
}
