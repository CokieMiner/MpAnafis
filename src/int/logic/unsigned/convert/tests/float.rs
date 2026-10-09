//! Native rounding oracles, discarded-bit ties, exponent carries, and overflow.

use proptest::{
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Native integer casts supply the independent IEEE-754 rounding oracle"
)]
fn float_conversions_match_native_casts_in_both_storage_classes() {
    let check = |value: u128| {
        let integer = InternalMpUint::from_u128(value);
        let mut heap = InternalMpUint::with_capacity(16);
        heap.clone_from(&integer);
        let expected32 = value as f32;
        let expected64 = value as f64;
        for source in [&integer, &heap] {
            assert_eq!(
                source.to_f32(),
                expected32.is_finite().then_some(expected32)
            );
            assert_eq!(source.to_f64(), Some(expected64));
        }
        if let Ok(limb) = Limb::try_from(value) {
            assert_eq!(InternalMpUint::from_limb(limb).to_f32(), Some(limb as f32));
            assert_eq!(InternalMpUint::from_limb(limb).to_f64(), Some(limb as f64));
        }
    };
    for value in [
        0,
        1,
        u128::try_from(Limb::MAX).expect("limb fits u128"),
        u128::from(u64::MAX),
        u128::MAX,
    ] {
        check(value);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&any::<u128>(), |value| {
            check(value);
            Ok(())
        })
        .expect("native floating conversion property");
}

#[test]
fn float32_rounding_preserves_ties_and_reports_exponent_overflow() {
    let check = |significand: u32, shift: usize| {
        let base = InternalMpUint::from_u64(u64::from(significand)).shl(shift);
        let halfway = InternalMpUint::power_of_two(shift.checked_sub(1).expect("positive shift"));
        let bits = u32::try_from(shift.checked_add(150).expect("bounded exponent"))
            .expect("biased exponent fits u32")
            << 23
            | (significand & 0x007f_ffff);
        let one = InternalMpUint::one();
        for (tail, round_up) in [
            (halfway.sub(&one), false),
            (halfway.clone(), significand & 1 != 0),
            (halfway.add(&one), true),
        ] {
            let expected = f32::from_bits(
                bits.checked_add(u32::from(round_up))
                    .expect("float bits fit u32"),
            );
            assert_eq!(
                base.add(&tail).to_f32(),
                expected.is_finite().then_some(expected)
            );
        }
    };
    for significand in [
        1_u32 << 23,
        (1_u32 << 23).checked_add(1).expect("bounded significand"),
        (1_u32 << 24).checked_sub(1).expect("positive mask"),
    ] {
        for shift in [1, 30, LIMB_BITS, 104] {
            check(significand, shift);
        }
    }
    let maximum = InternalMpUint::max_for_bits(24).shl(104);
    assert_eq!(maximum.to_f32(), Some(f32::MAX));
    assert_eq!(maximum.add(&InternalMpUint::one()).to_f32(), Some(f32::MAX));
    assert_eq!(InternalMpUint::power_of_two(128).to_f32(), None);
    let strategy = ((1_u32 << 23)..(1_u32 << 24), 1_usize..=104);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(significand, shift)| {
            check(significand, shift);
            Ok(())
        })
        .expect("f32 rounding property");
}

#[test]
fn float64_rounding_preserves_ties_and_reports_exponent_overflow() {
    let check = |significand: u64, shift: usize| {
        let base = InternalMpUint::from_u64(significand).shl(shift);
        let halfway = InternalMpUint::power_of_two(shift.checked_sub(1).expect("positive shift"));
        let bits = u64::try_from(shift.checked_add(1075).expect("bounded exponent"))
            .expect("biased exponent fits u64")
            << 52
            | (significand & 0x000f_ffff_ffff_ffff);
        let one = InternalMpUint::one();
        for (tail, round_up) in [
            (halfway.sub(&one), false),
            (halfway.clone(), significand & 1 != 0),
            (halfway.add(&one), true),
        ] {
            let expected = f64::from_bits(
                bits.checked_add(u64::from(round_up))
                    .expect("float bits fit u64"),
            );
            assert_eq!(
                base.add(&tail).to_f64(),
                expected.is_finite().then_some(expected)
            );
        }
    };
    for significand in [
        1_u64 << 52,
        (1_u64 << 52).checked_add(1).expect("bounded significand"),
        (1_u64 << 53).checked_sub(1).expect("positive mask"),
    ] {
        for shift in [1, LIMB_BITS, 971] {
            check(significand, shift);
        }
    }
    let maximum = InternalMpUint::max_for_bits(53).shl(971);
    assert_eq!(maximum.to_f64(), Some(f64::MAX));
    assert_eq!(maximum.add(&InternalMpUint::one()).to_f64(), Some(f64::MAX));
    assert_eq!(InternalMpUint::power_of_two(1024).to_f64(), None);
    let strategy = ((1_u64 << 52)..(1_u64 << 53), 1_usize..=971);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(significand, shift)| {
            check(significand, shift);
            Ok(())
        })
        .expect("f64 rounding property");
}
