//! Selected and dispatched products against the conventional multiplication tower.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::{
    int::logic::unsigned::math::mul::{Limb, MulScratch, Multiplication, TierCeiling},
    parallel::SequentialExecutor,
};

use super::shapes;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[cfg_attr(miri, ignore = "the full crossover matrix executes multi-thousand-limb products and squares")]
    #[test]
    fn selected_and_dispatched_results_match_the_conventional_tower(seed in any::<Limb>()) {
        let executor = SequentialExecutor;
        for (larger, smaller) in shapes::products() {
            if smaller == 0 { continue; }
            let left = operand(larger, seed);
            let right = operand(smaller, !seed);
            let width = larger.checked_add(smaller).expect("test product width fits");
            let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
            let mut scratch = vec![Limb::MAX; Multiplication::scratch_len_for_parallelism(plan, larger, smaller, 1)];
            let mut planned = vec![Limb::MAX; width];
            Multiplication::execute_plan_with_executor(plan, &mut planned, &left, &right, &mut scratch, &executor);
            let reference_plan = Multiplication::select_plan(larger, smaller, TierCeiling::Toom6);
            let mut reference_scratch = vec![Limb::MAX; Multiplication::scratch_len_for_parallelism(reference_plan, larger, smaller, 1)];
            let mut expected = vec![Limb::MAX; width];
            Multiplication::execute_plan_with_executor(reference_plan, &mut expected, &left, &right, &mut reference_scratch, &executor);
            prop_assert_eq!(&planned, &expected, "{:?} at {}x{}", plan, larger, smaller);
            let mut dispatched = vec![Limb::MAX; width];
            scratch.resize(Multiplication::required_scratch(larger, smaller), Limb::MAX);
            Multiplication::mul_limbs_with_slice_scratch(&left, &right, &mut dispatched, &mut scratch);
            prop_assert_eq!(dispatched, expected, "dispatched {:?} at {}x{}", plan, larger, smaller);
        }
        let mut pool = MulScratch::default();
        for len in shapes::widths() {
            if len == 0 { continue; }
            let operand = operand(len, seed);
            let plan = Multiplication::select_square_plan(len, TierCeiling::Full);
            let width = len.checked_mul(2).expect("test square width fits");
            let mut scratch = vec![Limb::MAX; Multiplication::square_scratch_len_for_parallelism(plan, len, 1)];
            let mut planned = vec![Limb::MAX; width];
            Multiplication::execute_square_plan_with_executor(plan, &mut planned, &operand, &mut scratch, &executor);
            let reference_plan = Multiplication::select_square_plan(len, TierCeiling::Toom6);
            let mut reference_scratch = vec![Limb::MAX; Multiplication::square_scratch_len_for_parallelism(reference_plan, len, 1)];
            let mut expected = vec![Limb::MAX; width];
            Multiplication::execute_square_plan_with_executor(reference_plan, &mut expected, &operand, &mut reference_scratch, &executor);
            prop_assert_eq!(&planned, &expected, "{:?} at {}^2", plan, len);
            let mut dispatched = vec![Limb::MAX; width];
            pool.buf.fill(Limb::MAX);
            Multiplication::sqr_limbs_with_scratch(&operand, &mut dispatched, &mut pool);
            prop_assert_eq!(dispatched, expected, "dispatched {:?} at {}^2", plan, len);
        }
    }
}

fn operand(len: usize, seed: Limb) -> Vec<Limb> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // The recurrence generates dense words modulo the native limb radix.
            state = state.wrapping_mul(25_173).wrapping_add(13_849);
            state | 1
        })
        .collect()
}
