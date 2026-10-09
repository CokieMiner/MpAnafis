//! Shape predicates against exact widened arithmetic at native-width limits.

use proptest::prelude::*;

use crate::int::logic::unsigned::math::mul::Widths;
#[cfg(not(target_pointer_width = "16"))]
use crate::int::logic::unsigned::math::mul::{
    TRANSFORM_MAX_OPERAND_RATIO, TRANSFORM_MIN_SMALLER_LIMBS,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 64 }))]

    #[test]
    fn ratio_preferences_match_exact_products(
        left in any::<usize>(), right in any::<usize>(),
    ) {
        for widths in [Widths::new(left, right), Widths::new(0, 0), Widths::new(1, usize::MAX),
            Widths::new(usize::MAX, usize::MAX), Widths::new(usize::MAX.div_euclid(18), usize::MAX.div_euclid(8)),
            Widths::new(usize::MAX.div_euclid(21), usize::MAX)] {
            let smaller = u128::try_from(widths.smaller).expect("native width fits u128");
            let larger = u128::try_from(widths.larger).expect("native width fits u128");
            prop_assert_eq!(widths.prefers_blocked_product(), smaller != 0 && larger.checked_mul(3).expect("widened product fits") >= smaller.checked_mul(4).expect("widened product fits"));
            prop_assert_eq!(widths.toom4_balanced(), smaller.checked_mul(4).expect("widened product fits") >= larger.checked_mul(3).expect("widened product fits"));
            prop_assert_eq!(widths.toom6_balanced(), smaller.checked_mul(18).expect("widened product fits") >= larger.checked_mul(17).expect("widened product fits"));
            prop_assert_eq!(widths.degenerate_child_split(), smaller != 0 && larger >= smaller.checked_mul(8).expect("widened product fits"));
            let split = larger.div_ceil(8);
            let balanced_eight = smaller >= 8
                && smaller.checked_mul(21).expect("widened product fits") >= larger.checked_mul(20).expect("widened product fits")
                && smaller > split.checked_mul(7).expect("widened product fits");
            prop_assert_eq!(widths.toom8_balanced(), balanced_eight);
            let half_split = larger.div_ceil(9).max(smaller.div_ceil(8));
            let half_eight = smaller >= 8
                && smaller.checked_mul(5).expect("widened product fits") >= larger.checked_mul(4).expect("widened product fits")
                && larger > half_split.checked_mul(8).expect("widened product fits")
                && smaller > half_split.checked_mul(7).expect("widened product fits");
            prop_assert_eq!(widths.toom8_half_suitable(), half_eight);
            #[cfg(not(target_pointer_width = "16"))]
            prop_assert_eq!(widths.transform_padding_is_affordable(), widths.smaller >= TRANSFORM_MIN_SMALLER_LIMBS && larger <= smaller.checked_mul(u128::try_from(TRANSFORM_MAX_OPERAND_RATIO).expect("ratio fits")).expect("widened product fits"));
        }
    }
}
