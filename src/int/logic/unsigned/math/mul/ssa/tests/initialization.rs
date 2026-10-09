//! Prepared SSA products into uninitialized outputs with dirty exact arenas.

#![expect(
    unsafe_code,
    reason = "Operand-bound plans initialize exact disjoint MaybeUninit output spans before tests expose them as limbs"
)]

use core::mem::MaybeUninit;

use super::*;

#[test]
fn prepared_planned_and_direct_products_initialize_every_output_limb() {
    for choice in [
        TransformChoice::PLANNED,
        TransformChoice::FORCED_DIRECT_FERMAT,
    ] {
        let a = vec![Limb::MAX; 32];
        let b = vec![Limb::MAX; 17];
        let plan = SsaMultiplicationPlan::try_new(&a, &b, choice, NonZeroUsize::MIN)
            .expect("small SSA geometry fits");
        let mut expected = vec![0; plan.result_len];
        Schoolbook::mul_nonempty_distinct(&mut expected, &a, &b);
        let mut output = vec![MaybeUninit::uninit(); plan.result_len];
        let mut scratch = vec![Limb::MAX; plan.scratch_len];
        // SAFETY: the immutable operand-bound plan has its exact destination
        // and arena and executes with its construction-time worker budget.
        unsafe {
            plan.run_with_scratch(&mut output, &mut scratch, &SequentialExecutor);
        }
        // SAFETY: successful plan execution initialized every output limb.
        let initialized = unsafe { LimbOutput::assume_init(&output) };
        assert_eq!(initialized, expected, "SSA {choice:?}");
    }
}
