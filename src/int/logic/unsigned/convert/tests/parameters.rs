//! Native radix powers and digit-estimate bounds.

use super::{Convert, InternalMpUint, LIMB_BITS, RadixParameters};

#[test]
fn radix_parameters_and_digit_estimates_bound_every_supported_radix() {
    for radix in 2_u32..=36 {
        let base = usize::try_from(radix).expect("small radix");
        let parameters = RadixParameters::for_limb(radix);
        let mut power = 1_usize;
        let mut digits = 0_usize;
        while let Some(next) = power.checked_mul(base) {
            power = next;
            digits = digits.checked_add(1).expect("native digit count");
        }
        assert_eq!(parameters.max_power, power);
        assert_eq!(parameters.max_digits, digits);
        for bits in [
            1,
            LIMB_BITS,
            LIMB_BITS.checked_mul(4).expect("inline bits"),
            257,
        ] {
            for value in [
                InternalMpUint::power_of_two(bits.checked_sub(1).expect("positive width")),
                InternalMpUint::max_for_bits(bits),
            ] {
                let length = value.to_string_radix(radix).len();
                let estimate = Convert::estimated_digits(bits, radix);
                assert!(estimate >= length, "radix={radix}, bits={bits}");
                assert_eq!(
                    estimate,
                    bits.div_ceil(usize::try_from(radix.ilog2()).expect("small radix logarithm"))
                );
            }
        }
        for bits in [
            0,
            usize::MAX.checked_sub(4).expect("maximum width"),
            usize::MAX,
        ] {
            let estimate = Convert::estimated_digits(bits, radix);
            assert!(estimate <= bits.max(1), "radix={radix}, bits={bits}");
            assert!(
                bits == 0 || estimate != 0,
                "a positive width has positive output length"
            );
        }
    }
}
