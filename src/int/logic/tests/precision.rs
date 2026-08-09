use proptest::prelude::*;

use super::*;

struct GlobalPrecisionRestore(AmbientPrecision);

impl Drop for GlobalPrecisionRestore {
    fn drop(&mut self) {
        let _previous = InternalPrecisionContext::set_global(self.0);
    }
}

fn ambient_precision() -> impl Strategy<Value = AmbientPrecision> {
    prop_oneof![
        Just(AmbientPrecision::Unset),
        Just(AmbientPrecision::Unlimited),
        (1_usize..usize::MAX).prop_map(|bits| {
            AmbientPrecision::new_bounded(bits)
                .expect("the strategy only generates valid bounded widths")
        }),
    ]
}

proptest! {
    #[test]
    fn ambient_encoding_round_trips(precision in ambient_precision()) {
        prop_assert_eq!(decode_precision(encode_precision(precision)), precision);
    }

    #[test]
    fn ambient_resolution(bits in 1_usize..=256) {
        let initial_precision = InternalPrecisionContext::set_global(AmbientPrecision::Unset);
        let _restore = GlobalPrecisionRestore(initial_precision);
        prop_assert_eq!(InternalPrecisionContext::active(), AmbientPrecision::Unset);

        let bounded_precision = AmbientPrecision::new_bounded(bits)
            .expect("the strategy only generates valid bounded widths");
        let before_bounded = InternalPrecisionContext::set_global(bounded_precision);
        prop_assert_eq!(before_bounded, AmbientPrecision::Unset);
        prop_assert_eq!(InternalPrecisionContext::active(), bounded_precision);

        let before_unlimited = InternalPrecisionContext::set_global(AmbientPrecision::Unlimited);
        prop_assert_eq!(before_unlimited, bounded_precision);
        prop_assert_eq!(InternalPrecisionContext::active(), AmbientPrecision::Unlimited);

        #[cfg(feature = "std")]
        {
            let bounded_result: Result<(), TestCaseError> =
                InternalPrecisionContext::with_bounded(bits, || {
                    prop_assert_eq!(InternalPrecisionContext::active(), bounded_precision);

                    let unlimited_result: Result<(), TestCaseError> =
                        InternalPrecisionContext::with_unlimited(|| {
                            prop_assert_eq!(InternalPrecisionContext::active(), AmbientPrecision::Unlimited);
                            Ok(())
                        });
                    unlimited_result?;

                    prop_assert_eq!(InternalPrecisionContext::active(), bounded_precision);
                    Ok(())
                });
            bounded_result?;

            prop_assert_eq!(InternalPrecisionContext::active(), AmbientPrecision::Unlimited);
        }

        #[cfg(not(feature = "std"))]
        let _ = bits;

        let before_restore = InternalPrecisionContext::set_global(initial_precision);
        prop_assert_eq!(before_restore, AmbientPrecision::Unlimited);
    }
}
