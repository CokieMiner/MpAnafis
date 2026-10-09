//! Isolated global precision and scoped restoration contracts.

#[cfg(any(target_has_atomic = "ptr", feature = "std"))]
use std::panic::catch_unwind;
#[cfg(target_has_atomic = "ptr")]
use std::{env, process::Command};

use mp_anafis::{AmbientPrecision, BoundedPrecision, MpInt, MpUint, Precision, PrecisionContext};
#[cfg(target_has_atomic = "ptr")]
use mp_anafis::{ParseMpIntErrorKind, ParseMpUintErrorKind};

#[test]
#[cfg(target_has_atomic = "ptr")]
#[cfg_attr(
    miri,
    ignore = "Global precision is tested in an isolated native subprocess"
)]
fn global_context_preserves_construction_and_scope_contracts() {
    const CHILD: &str = "MP_ANAFIS_FUZZ_PRECISION_CHILD";
    if env::var_os(CHILD).is_none() {
        let status = Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::precision::global_context_preserves_construction_and_scope_contracts",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let initial = PrecisionContext::set_global(AmbientPrecision::Unset);
    assert_eq!(PrecisionContext::active(), AmbientPrecision::Unset);
    assert_eq!(MpUint::from(255_u16).precision(), Precision::Unlimited);
    let width = BoundedPrecision::new(8).unwrap();
    let precision = Precision::Bounded(width);
    assert_eq!(
        PrecisionContext::set_global(AmbientPrecision::Bounded(width)),
        AmbientPrecision::Unset
    );
    assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(width));
    assert_eq!(MpUint::from(255_u16).precision(), precision);
    assert_eq!(
        MpUint::from(256_u16).precision(),
        Precision::new_bounded(9).unwrap()
    );
    assert_eq!(
        MpInt::from(128_u16).precision(),
        Precision::new_bounded(9).unwrap()
    );
    assert_eq!(MpInt::from(-128_i16).precision(), precision);
    assert_eq!(MpUint::from_be_bytes(&[]).precision(), precision);
    assert_eq!(MpInt::from_be_bytes(&[]).precision(), Precision::Unlimited);
    assert_eq!(MpInt::from_le_bytes(&[]).precision(), Precision::Unlimited);
    assert_eq!(
        MpUint::from_be_bytes(&[1, 0]).precision(),
        Precision::new_bounded(9).unwrap()
    );
    assert_eq!(
        MpInt::from_be_bytes(&[0, 128]).precision(),
        Precision::new_bounded(9).unwrap()
    );
    assert_eq!(
        MpUint::from_str_radix("256", 10).unwrap_err().kind(),
        &ParseMpUintErrorKind::TooLarge
    );
    assert_eq!(
        MpInt::from_str_radix("128", 10).unwrap_err().kind(),
        &ParseMpIntErrorKind::TooLarge
    );
    assert_eq!(
        MpUint::from_str_radix("255", 10).unwrap().precision(),
        precision
    );
    assert_eq!(
        MpInt::from_str_radix("-128", 10).unwrap().precision(),
        precision
    );
    assert!(catch_unwind(|| MpUint::from(255_u16) + MpUint::from(1_u8)).is_err());
    assert!(catch_unwind(|| MpInt::from(-128_i16) / MpInt::from(-1_i8)).is_err());
    for value in [
        MpUint::default(),
        MpUint::zero(),
        MpUint::one(),
        core::iter::empty::<MpUint>().sum(),
        core::iter::empty::<MpUint>().product(),
    ] {
        assert_eq!(value.precision(), Precision::Unlimited);
    }
    for value in [
        MpInt::default(),
        MpInt::zero(),
        MpInt::one(),
        MpInt::minus_one(),
    ] {
        assert_eq!(value.precision(), Precision::Unlimited);
    }
    assert_eq!(
        Precision::from(AmbientPrecision::Unset),
        Precision::Unlimited
    );
    assert_eq!(
        Precision::from(AmbientPrecision::Unlimited),
        Precision::Unlimited
    );
    assert_eq!(Precision::from(AmbientPrecision::Bounded(width)), precision);
    assert_eq!(
        PrecisionContext::set_global(AmbientPrecision::Unlimited),
        AmbientPrecision::Bounded(width)
    );
    assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
    assert_eq!(
        PrecisionContext::set_global(initial),
        AmbientPrecision::Unlimited
    );
}

#[test]
#[cfg(feature = "std")]
fn scoped_context_restores_after_nested_closures_and_panics() {
    let before = PrecisionContext::active();
    PrecisionContext::with_bounded(1, || {
        let width = BoundedPrecision::new(1).unwrap();
        assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(width));
        assert_eq!(MpUint::from(1_u8).precision(), Precision::Bounded(width));
        assert_eq!(
            MpInt::from(1_u8).precision(),
            Precision::new_bounded(2).unwrap()
        );
        assert_eq!(MpInt::from(-1_i8).precision(), Precision::Bounded(width));
        let child = std::thread::spawn(PrecisionContext::active).join().unwrap();
        assert_eq!(child, before);
        assert!(
            catch_unwind(|| PrecisionContext::with_unlimited(|| panic!("scope restoration")))
                .is_err()
        );
        assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(width));
    });
    assert_eq!(PrecisionContext::active(), before);
    assert!(
        catch_unwind(|| PrecisionContext::with_bounded(7, || panic!("scope restoration"))).is_err()
    );
    assert_eq!(PrecisionContext::active(), before);
    for invalid in [0, usize::MAX] {
        assert!(catch_unwind(|| PrecisionContext::with_bounded(invalid, || ())).is_err());
    }
    assert_eq!(PrecisionContext::active(), before);
}
