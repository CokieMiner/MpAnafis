//! Cloning and swapping preserve numeric values, precision, and reusable capacity.

use alloc::vec;

use proptest::prelude::{any, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, Precision};

use super::support::{nz, uint_from_words};

proptest! {
    #[test]
    fn clone_and_swap_preserve_values_precision_and_capacity(
        generated in proptest::collection::vec(any::<usize>(), 0..=64),
    ) {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<MpInt>();
        require_send_sync::<MpUint>();

        let mut unsigned = MpUint::with_capacity(96);
        let mut signed = MpInt::with_capacity(96);
        let magnitudes = [64, 5, 4, 1, 0].into_iter()
            .map(|width| uint_from_words(&vec![1; width]))
            .chain(core::iter::once(uint_from_words(&generated)));
        for magnitude in magnitudes {
            for bounded in [false, true] {
                let source = if bounded {
                    MpUint::with_precision_checked(magnitude.clone(), nz(8192)).expect("magnitude fits")
                } else {
                    magnitude.clone()
                };
                let precision = source.precision();
                let mut copy = source.clone();
                prop_assert_eq!(&copy, &source);
                prop_assert_eq!(copy.precision(), precision);
                copy += MpUint::one();
                prop_assert_eq!(&(&copy - MpUint::one()), &source);
                unsigned.clone_from(&source);
                prop_assert_eq!(&unsigned, &source);
                prop_assert_eq!(unsigned.precision(), precision);
                prop_assert_eq!(unsigned.capacity(), 96);
                let mut other = MpUint::zero();
                unsigned.swap(&mut other);
                prop_assert_eq!(&other, &source);
                prop_assert_eq!(other.precision(), precision);
                prop_assert_eq!(unsigned.precision(), Precision::Unlimited);
                unsigned.swap(&mut other);

                for positive in [false, true] {
                    let signed_magnitude = MpInt::from(source.clone());
                    let signed_source = if positive { signed_magnitude } else { -signed_magnitude };
                    let mut signed_copy = signed_source.clone();
                    prop_assert_eq!(&signed_copy, &signed_source);
                    prop_assert_eq!(signed_copy.precision(), precision);
                    signed_copy += MpInt::one();
                    prop_assert_eq!(&(&signed_copy - MpInt::one()), &signed_source);
                    signed.clone_from(&signed_source);
                    prop_assert_eq!(&signed, &signed_source);
                    prop_assert_eq!(signed.precision(), precision);
                    prop_assert_eq!(signed.capacity(), 96);
                    prop_assert_eq!(signed.is_negative(), !positive && !magnitude.is_zero());
                    let mut signed_other = MpInt::zero();
                    signed.swap(&mut signed_other);
                    prop_assert_eq!(&signed_other, &signed_source);
                    prop_assert_eq!(signed_other.precision(), precision);
                    prop_assert_eq!(signed.precision(), Precision::Unlimited);
                    signed.swap(&mut signed_other);
                }
            }
        }
    }
}

#[test]
fn signed_endpoints_preserve_value_domain_and_precision() {
    for bits in [1_usize, 2, 8, 64, 65, 256] {
        let unsigned = MpUint::max_for_precision(bits);
        let signed = MpInt::from(unsigned.clone());
        assert_eq!(signed, unsigned);
        assert_eq!(signed.precision().significant_bits(), bits.checked_add(1));
        let minimum = MpInt::min_for_precision(bits);
        assert!(minimum.nth_root(1).is_none());
        let negative_one = MpInt::with_precision_checked(-1_i8, nz(bits)).expect("minus one fits");
        let extracted = negative_one.bit_range(0, bits);
        assert_eq!(extracted, unsigned);
        assert_eq!(
            extracted.precision().significant_bits(),
            bits.checked_add(1)
        );
        assert_eq!(negative_one.checked_pow(2).is_none(), bits == 1);
        assert_eq!(minimum.sqrt_rem().is_none(), bits == 1);
    }
}
