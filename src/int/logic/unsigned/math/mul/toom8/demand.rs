//! Exact scratch demand of the fixed-width Toom-8 evaluation children.
//!
//! Evaluation products retain their complete low block and discard only zero
//! guard limbs. Their widths therefore lie in `split_len..=eval_len`, independent
//! of cancellation inside the polynomial. No smaller-width scan or monotonicity
//! assumption is needed. Endpoint products are sized separately by the layout.

use core::cmp::max;

use super::{
    EVALUATION_GUARD_BITS, LIMB_BITS, Multiplication, TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS,
    TierCeiling, Toom8,
};

/// Namespace for the child workspace of the Toom-8 tiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChildDemand;

impl ChildDemand {
    /// Covers every evaluation-product shape and the split-width zero endpoint.
    pub fn recursive_mul_scratch(split_len: usize, eval_len: usize) -> usize {
        debug_assert_eq!(
            eval_len,
            Toom8::evaluation_len(split_len),
            "child sizing requires the complete Toom-8 evaluation width"
        );
        let guarded = split_len < TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS
            && EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS) == 1;
        let largest_child = if guarded { split_len } else { eval_len };
        let mut inner = 0;
        for len_a in split_len..=largest_child {
            for len_b in split_len..=largest_child {
                let plan = Multiplication::select_plan(len_a, len_b, TierCeiling::Toom6);
                inner = max(inner, Multiplication::scratch_len(plan, len_a, len_b));
            }
        }
        inner
    }

    /// Covers every evaluation square and the split-width zero endpoint.
    pub fn recursive_sqr_scratch(split_len: usize, eval_len: usize) -> usize {
        debug_assert_eq!(
            eval_len,
            Toom8::evaluation_len(split_len),
            "child sizing requires the complete Toom-8 evaluation width"
        );
        let largest_child = if EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS) == 1 {
            split_len
        } else {
            eval_len
        };
        let mut inner = 0;
        for len in split_len..=largest_child {
            let plan = Multiplication::select_square_plan(len, TierCeiling::Toom6);
            inner = max(inner, Multiplication::square_scratch_len(plan, len));
        }
        inner
    }
}
