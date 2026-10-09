//! Exact part counts and selector precedence for the four-by-three band.

use crate::int::logic::unsigned::math::{TOOM_COOK_THRESHOLD, mul::MulPlan};

use super::super::{Multiplication, TierCeiling, Widths};

#[test]
fn admitted_shapes_split_four_by_three_and_reach_the_fractional_band() {
    let end = if cfg!(miri) {
        TOOM_COOK_THRESHOLD.checked_add(64)
    } else {
        TOOM_COOK_THRESHOLD
            .checked_add(400)
            .map(|value| value.max(600))
    }
    .expect("test range fits");
    let mut named = false;
    let mut below_three_by_two = false;
    for larger in 2_usize..end {
        for smaller in 1..=larger {
            let widths = Widths::new(larger, smaller);
            if widths.toom43_suitable() {
                let split = larger.div_ceil(4);
                assert!(larger > split.checked_mul(3).expect("test split fits"));
                assert!(
                    smaller > split.checked_mul(2).expect("test split fits")
                        && smaller <= split.checked_mul(3).expect("test split fits")
                );
            }
            if Multiplication::select_plan(larger, smaller, TierCeiling::Toom6) == MulPlan::Toom43 {
                named = true;
                assert!(widths.toom43_suitable() && !widths.toom32_suitable());
                below_three_by_two |= larger.checked_mul(2).expect("test ratio fits")
                    < smaller.checked_mul(3).expect("test ratio fits");
            }
        }
    }
    assert!(
        named && below_three_by_two,
        "the selector must serve the four-by-three band"
    );
}
