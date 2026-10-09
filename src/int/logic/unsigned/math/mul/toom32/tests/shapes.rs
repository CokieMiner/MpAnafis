//! Shape admission, ceiling-rounding bounds, and selector reachability.

use crate::int::logic::unsigned::math::{TOOM_COOK_THRESHOLD, mul::MulPlan};

use super::super::{Multiplication, TierCeiling, Widths};

#[test]
fn admitted_shapes_have_three_by_two_parts_and_reach_the_selector() {
    let end = if cfg!(miri) {
        TOOM_COOK_THRESHOLD.checked_add(64)
    } else {
        TOOM_COOK_THRESHOLD
            .checked_add(300)
            .map(|value| value.max(600))
    }
    .expect("test range fits");
    let mut named = false;
    for larger in 2_usize..end {
        for smaller in 1..=larger {
            if !Widths::new(larger, smaller).toom32_suitable() {
                continue;
            }
            let split = larger.div_ceil(3);
            let twice = split.checked_mul(2).expect("test split fits");
            assert!(larger > twice && smaller > split && smaller <= twice);
            assert!(larger < smaller.checked_mul(3).expect("test ratio fits"));
            assert!(
                smaller.checked_mul(3).expect("test ratio fits")
                    <= larger
                        .checked_mul(2)
                        .and_then(|value| value.checked_add(4))
                        .expect("rounded bound fits")
            );
            named |=
                Multiplication::select_plan(larger, smaller, TierCeiling::Toom6) == MulPlan::Toom32;
        }
    }
    assert!(
        named,
        "the selector must reach an admitted three-by-two shape"
    );
}
