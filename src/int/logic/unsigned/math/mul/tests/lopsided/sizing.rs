//! Empty dispatches, selected block widths, and virtual workspace overflow.

use core::num::NonZeroUsize;

use crate::{
    int::logic::unsigned::math::mul::{Limb, Lopsided, MulPlan, Multiplication},
    parallel::SequentialExecutor,
};

#[test]
fn empty_operands_stop_before_blocked_workspace_arithmetic() {
    for (a, b) in [(0, 0), (0, usize::MAX), (usize::MAX, 0)] {
        assert_eq!(
            Multiplication::scratch_len_for_parallelism(MulPlan::Lopsided, a, b, 4),
            0
        );
        assert_eq!(Lopsided::mul_scratch_len(a, b, NonZeroUsize::MAX, 4), 0);
    }
    let operand = [Limb::MAX, 7];
    for (left, right) in [
        (&[][..], &[][..]),
        (&operand[..], &[][..]),
        (&[][..], &operand[..]),
    ] {
        let mut output = [Limb::MAX; 3];
        Multiplication::execute_plan_with_executor(
            MulPlan::Lopsided,
            &mut output,
            left,
            right,
            &mut [],
            &SequentialExecutor,
        );
        assert_eq!(output, [0; 3]);
    }
}

#[test]
fn selected_blocks_fit_the_ordered_operands() {
    let widths: &[usize] = if cfg!(miri) {
        &[1, 2, 3, 32]
    } else {
        &[1, 2, 3, 31, 32, 63, 64, 127, 257, 509, 1024, 4096]
    };
    for &smaller in widths {
        let smaller_width = NonZeroUsize::new(smaller).expect("nonempty operand");
        for ratio in [2, 3, 4, 9, 16] {
            for tail in [0, smaller.div_euclid(2), smaller - 1] {
                let Some(larger) = smaller
                    .checked_mul(ratio)
                    .and_then(|width| width.checked_add(tail))
                else {
                    continue;
                };
                let block = Lopsided::block_len(
                    NonZeroUsize::new(larger).expect("nonempty operand"),
                    smaller_width,
                );
                assert!(block.get() <= larger, "{larger} by {smaller} limbs");
            }
        }
    }
    // A positive virtual width can exceed the 9:8 extension's range while
    // remaining a valid fallback block. These cases allocate no operands.
    for smaller in [
        usize::MAX,
        usize::MAX
            .checked_sub(1)
            .expect("maximum has a predecessor"),
        usize::MAX
            .checked_sub(usize::MAX.div_euclid(9))
            .expect("ninth does not exceed the maximum"),
    ] {
        assert_eq!(smaller.checked_add(smaller.div_ceil(8)), None);
        let smaller_width = NonZeroUsize::new(smaller).expect("positive virtual width");
        assert_eq!(
            Lopsided::block_len(NonZeroUsize::MAX, smaller_width),
            smaller_width
        );
    }
}

#[test]
#[should_panic(expected = "lopsided worker region overflows usize")]
fn workspace_overflow_is_rejected() {
    let _ = Lopsided::mul_scratch_len(usize::MAX, 1, NonZeroUsize::MAX, 1);
}
