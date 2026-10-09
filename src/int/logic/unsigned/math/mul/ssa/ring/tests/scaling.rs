//! One-bit scales against independent integer halving and modular shifts.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded initialized guarded coefficients provide disjoint exact scaling windows and sentinel limbs"
)]

use alloc::vec;

use proptest::prelude::*;

use super::{LIMB_BITS, Limb, SsaRing, oracle_half, oracle_negation};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn one_bit_scales_match_ring_shifts_and_integer_halving(data in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 129 }), guard in 0_usize..=1) {
        check_scaling(&data, guard);
    }
}

#[test]
fn scaling_covers_guards_carry_chains_and_backend_widths() {
    for ml in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 257] {
        if cfg!(miri) && ml > 8 {
            continue;
        }
        for guard in [0, 1] {
            for low in [0, 1, 2, 3, Limb::MAX - 1, Limb::MAX] {
                let mut data = vec![0; ml];
                data[0] = low;
                check_scaling(&data, guard);
                data.fill(Limb::MAX);
                data[0] = low;
                check_scaling(&data, guard);
            }
            let mut data = vec![Limb::MAX; ml];
            for stop in 0..ml {
                data[stop] = 0;
                check_scaling(&data, guard);
                data[stop] = Limb::MAX;
            }
        }
    }
}

fn check_scaling(data: &[Limb], guard: Limb) {
    let ml = data.len();
    let bits = ml * LIMB_BITS;
    let mut source = data.to_vec();
    source.push(guard);
    let negated = oracle_negation(data, guard);
    let canonical = oracle_negation(&negated[..ml], negated[ml]);
    for divide in [false, true] {
        let mut actual = vec![37; ml + 3];
        actual[1..=ml + 1].copy_from_slice(&source);
        let mut expected = vec![Limb::MAX; ml + 1];
        // SAFETY: complete disjoint initialized coefficients have guards<=1.
        // Both exponents are reduced in the positive limb-aligned ring; the
        // actual window excludes both sentinels from every kernel call.
        unsafe {
            if divide {
                SsaRing::halve_in_place(&mut actual[1..=ml + 1], bits);
            } else {
                SsaRing::double_in_place(&mut actual[1..=ml + 1], bits);
            }
            assert!(actual[ml + 1] <= 1);
            SsaRing::shift_from(
                &mut expected,
                &source,
                if divide { 2 * bits - 1 } else { 1 },
                bits,
            );
            let _ = SsaRing::normalize(&mut actual[1..=ml + 1], bits);
            let _ = SsaRing::normalize(&mut expected, bits);
        }
        assert_eq!(
            &actual[1..=ml + 1],
            expected,
            "ml={ml}, guard={guard}, divide={divide}"
        );
        if divide {
            assert_eq!(expected, oracle_half(&canonical));
        }
        // SAFETY: the canonical complete result permits the opposite scale;
        // the same exclusive window retains its initialized guard.
        unsafe {
            if divide {
                SsaRing::double_in_place(&mut actual[1..=ml + 1], bits);
            } else {
                SsaRing::halve_in_place(&mut actual[1..=ml + 1], bits);
            }
            let _ = SsaRing::normalize(&mut actual[1..=ml + 1], bits);
        }
        assert_eq!(&actual[1..=ml + 1], canonical);
        assert_eq!((actual[0], actual[ml + 2]), (37, 37));
    }
}
