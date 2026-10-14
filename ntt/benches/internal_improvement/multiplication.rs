//! Balanced products with retained tables or complete raw-scratch execution.

use divan::{Bencher, black_box};
use mp_anafis::tune_api::MultiplicationBenchState;
use mp_ntt::{Ntt, NttPlan, NttWorkspace};

use super::{Executor, Inputs, SIZES};

#[divan::bench(args = SIZES)]
fn ntt_workspace(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let b = Inputs::operand(len, 2);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark product width")];
    let mut reference = dst.clone();
    MultiplicationBenchState::default()
        .prepare(&mut reference, &a, &b)
        .run();
    let mut workspace = NttWorkspace::new();
    let executor = &Executor::default();
    assert!(
        workspace.mul_with_executor(&mut dst, &a, &b, executor),
        "benchmark shape is admitted"
    );
    assert_eq!(dst, reference, "NTT product matches the production tower");
    bencher.bench_local(|| {
        let _completed = black_box(workspace.mul_with_executor(
            black_box(&mut dst),
            black_box(&a),
            black_box(&b),
            executor,
        ));
    });
}

#[divan::bench(args = SIZES)]
fn ntt_raw_scratch(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let b = Inputs::operand(len, 2);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark product width")];
    let mut reference = dst.clone();
    MultiplicationBenchState::default()
        .prepare(&mut reference, &a, &b)
        .run();
    let plan = NttPlan::for_multiplication(len, len).expect("benchmark plan is admitted");
    let mut scratch = vec![0; plan.required_mul_scratch_limbs()];
    let executor = &Executor::default();
    assert!(
        Ntt::try_mul_with_scratch_and_executor(&mut dst, &a, &b, &mut scratch, executor),
        "raw benchmark shape is admitted"
    );
    assert_eq!(
        dst, reference,
        "raw NTT product matches the production tower"
    );
    bencher.bench_local(|| {
        let _completed = black_box(Ntt::try_mul_with_scratch_and_executor(
            black_box(&mut dst),
            black_box(&a),
            black_box(&b),
            black_box(&mut scratch),
            executor,
        ));
    });
}

#[divan::bench(args = SIZES)]
fn mp_production(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let b = Inputs::operand(len, 2);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark product width")];
    let mut reference = dst.clone();
    assert!(
        NttWorkspace::new().mul(&mut reference, &a, &b),
        "reference shape is admitted"
    );
    let mut state = MultiplicationBenchState::default();
    state.prepare(&mut dst, &a, &b).run();
    assert_eq!(dst, reference, "production product matches NTT");
    let mut prepared = state.prepare(&mut dst, &a, &b);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}
