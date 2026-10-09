//! Scalar predicates and exact unsigned precision bounds.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn scalar_properties_and_binary_bounds_match_explicit_bits() {
    let check = |limbs: &[Limb]| {
        let value = InternalMpUint::from_limbs_slice(limbs);
        assert_eq!(value.is_zero(), limbs.iter().all(|limb| *limb == 0));
        assert_eq!(
            value.is_one(),
            limbs.first() == Some(&1) && limbs.iter().skip(1).all(|limb| *limb == 0)
        );
        assert_eq!(
            value.is_odd(),
            limbs.first().is_some_and(|limb| limb & 1 != 0)
        );
        assert_eq!(value.is_even(), !value.is_odd());
        assert_eq!(
            value.is_power_of_two(),
            limbs.iter().map(|limb| limb.count_ones()).sum::<u32>() == 1
        );
        let width = value.significant_bits();
        for slack in [0, 1, LIMB_BITS] {
            let bits = width.checked_add(slack).expect("bounded precision");
            for shift in [
                0,
                slack,
                slack.checked_add(1).expect("bounded shift"),
                usize::MAX,
            ] {
                assert_eq!(
                    value.bounded_shl_overflows(bits, shift),
                    width != 0 && shift > slack
                );
            }
        }
    };
    for width in [0, 1, 3, 4, 5, 8] {
        check(&vec![0; width]);
        check(&vec![Limb::MAX; width]);
    }
    for bits in 0..=LIMB_BITS
        .checked_mul(5)
        .and_then(|width| width.checked_add(1))
        .expect("bounded test width")
    {
        let power = InternalMpUint::power_of_two(bits);
        let maximum = InternalMpUint::max_for_bits(bits);
        assert_eq!(maximum.add(&InternalMpUint::one()), power);
        assert_eq!(maximum.significant_bits(), bits);
        assert_eq!(
            power.significant_bits(),
            bits.checked_add(1).expect("bounded width")
        );
        assert!(power.is_power_of_two(), "binary powers have one set bit");
        assert!(
            power.get_bit(bits),
            "the constructed binary power sets its designated bit"
        );
        if bits < LIMB_BITS.checked_mul(INLINE_LIMBS).expect("inline width") {
            assert_eq!(power.capacity(), INLINE_LIMBS);
        }
        check(power.limbs());
        check(maximum.limbs());
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&collection::vec(any::<Limb>(), 0..=16), |limbs| {
            check(&limbs);
            Ok(())
        })
        .expect("unsigned scalar property");
}
