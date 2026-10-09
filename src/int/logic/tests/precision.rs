//! Global precision precedence in a separate test executable.

#![cfg(target_has_atomic = "ptr")]

use mp_anafis::{AmbientPrecision, PrecisionContext};
use proptest::prelude::{Just, ProptestConfig, Strategy, prop_assert_eq, prop_oneof, proptest};

struct GlobalPrecisionRestore(AmbientPrecision);

impl Drop for GlobalPrecisionRestore {
    fn drop(&mut self) {
        let _previous = PrecisionContext::set_global(self.0);
    }
}

fn ambient_precision() -> impl Strategy<Value = AmbientPrecision> {
    prop_oneof![
        Just(AmbientPrecision::Unset),
        Just(AmbientPrecision::Unlimited),
        (1_usize..usize::MAX).prop_map(|bits| {
            AmbientPrecision::new_bounded(bits).expect("generated bounded widths exclude sentinels")
        }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn policies_round_trip_and_scopes_override_the_global_default(
        bits in 1_usize..=256,
        precision in ambient_precision(),
    ) {
        let original = PrecisionContext::set_global(precision);
        let _restore = GlobalPrecisionRestore(original);
        prop_assert_eq!(PrecisionContext::active(), precision);
        let minimum = AmbientPrecision::new_bounded(1).expect("one is a valid bounded width");
        let maximum = AmbientPrecision::new_bounded(usize::MAX.checked_sub(1).expect("sentinel is nonzero")).expect("width precedes the sentinel");
        let bounded = AmbientPrecision::new_bounded(bits).expect("generated width is nonzero");
        let mut previous = precision;
        for policy in [AmbientPrecision::Unset, minimum, maximum, bounded, AmbientPrecision::Unlimited] {
            prop_assert_eq!(PrecisionContext::set_global(policy), previous);
            prop_assert_eq!(PrecisionContext::active(), policy);
            previous = policy;
        }

        #[cfg(feature = "std")]
        {
            let result: Result<(), proptest::test_runner::TestCaseError> =
                PrecisionContext::with_bounded(bits, || {
                    prop_assert_eq!(PrecisionContext::active(), bounded);
                    let _previous = PrecisionContext::set_global(precision);
                    prop_assert_eq!(PrecisionContext::active(), bounded);
                    let nested: Result<(), proptest::test_runner::TestCaseError> =
                        PrecisionContext::with_unlimited(|| {
                            prop_assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                            Ok(())
                        });
                    nested?;
                    prop_assert_eq!(PrecisionContext::active(), bounded);
                    Ok(())
                });
            result?;
            prop_assert_eq!(PrecisionContext::active(), precision);
        }
    }
}
