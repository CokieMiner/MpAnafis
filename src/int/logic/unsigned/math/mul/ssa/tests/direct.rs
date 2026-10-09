//! Exact destinations and reused arenas for forced direct Fermat squares.

#![expect(
    unsafe_code,
    reason = "Prepared plans bind exact square destinations and complete reusable arenas; surrounding sentinel limbs lie outside the output"
)]

use super::*;

#[cfg(feature = "_internal-tune")]
#[cfg_attr(
    miri,
    ignore = "direct and CRT squares span multi-thousand-limb native workloads"
)]
#[test]
fn direct_squares_match_crt_with_exact_dirty_scratch_and_output_guards() {
    for len in [1_usize, 3, 31, 127, 257, 2_048, 3_073] {
        for sparse in [false, true] {
            let mut input = vec![Limb::MAX; len];
            if sparse {
                input.fill(0);
                *input.first_mut().expect("nonempty input") = 3;
            }
            let plan = SsaSquaringPlan::try_new(&input, TransformChoice::FORCED_DIRECT_FERMAT, 1)
                .expect("direct square fits");
            let crt = SsaSquaringPlan::try_new(&input, TransformChoice::FORCED_CRT, 1)
                .expect("CRT square fits");
            let mut scratch = vec![Limb::MAX; plan.scratch_len];
            let mut crt_scratch = vec![Limb::MAX; crt.scratch_len];
            let width = len.checked_mul(2).expect("square fits");
            let mut expected = vec![0; width];
            assert!(Ssa::try_sqr_with_executor(
                &mut expected,
                &input,
                TransformChoice::FORCED_CRT,
                &mut crt_scratch,
                &SequentialExecutor
            ));
            let mut output = vec![Limb::MAX; width + 2];
            for _ in 0..2 {
                scratch.fill(Limb::MAX);
                // SAFETY: the window is exactly 2*len limbs, and the
                // disjoint scratch is sized by this plan for one worker.
                unsafe {
                    plan.run_with_scratch(
                        output.get_mut(1..width + 1).expect("output fits"),
                        &mut scratch,
                        &SequentialExecutor,
                    );
                }
                assert_eq!(output.get(1..width + 1).expect("output fits"), expected);
                assert_eq!(
                    (output.first(), output.last()),
                    (Some(&Limb::MAX), Some(&Limb::MAX))
                );
            }
        }
    }
}
