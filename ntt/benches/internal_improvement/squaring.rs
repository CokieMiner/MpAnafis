//! Squares with retained tables or complete raw-scratch execution.

use divan::{Bencher, black_box};
use mp_anafis::tune_api::SquaringBenchState;
use mp_ntt::{Ntt, NttPlan, NttWorkspace};

use super::{Executor, Inputs, SIZES};

#[divan::bench(args = SIZES)]
fn ntt_workspace(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark square width")];
    let mut reference = dst.clone();
    SquaringBenchState::default()
        .prepare(&mut reference, &a)
        .run();
    let mut workspace = NttWorkspace::new();
    let executor = &Executor::default();
    assert!(
        workspace.sqr_with_executor(&mut dst, &a, executor),
        "benchmark shape is admitted"
    );
    assert_eq!(dst, reference, "NTT square matches the production tower");
    bencher.bench_local(|| {
        let _completed =
            black_box(workspace.sqr_with_executor(black_box(&mut dst), black_box(&a), executor));
    });
}

#[divan::bench(args = SIZES)]
fn ntt_raw_scratch(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark square width")];
    let mut reference = dst.clone();
    SquaringBenchState::default()
        .prepare(&mut reference, &a)
        .run();
    let plan = NttPlan::for_multiplication(len, len).expect("benchmark plan is admitted");
    let mut scratch = vec![0; plan.required_sqr_scratch_limbs()];
    let executor = &Executor::default();
    assert!(
        Ntt::try_sqr_with_scratch_and_executor(&mut dst, &a, &mut scratch, executor),
        "raw benchmark shape is admitted"
    );
    assert_eq!(
        dst, reference,
        "raw NTT square matches the production tower"
    );
    bencher.bench_local(|| {
        let _completed = black_box(Ntt::try_sqr_with_scratch_and_executor(
            black_box(&mut dst),
            black_box(&a),
            black_box(&mut scratch),
            executor,
        ));
    });
}

#[divan::bench(args = SIZES)]
fn mp_production(bencher: Bencher, len: usize) {
    let a = Inputs::operand(len, 1);
    let mut dst = vec![0; len.checked_mul(2).expect("benchmark square width")];
    let mut reference = dst.clone();
    assert!(
        NttWorkspace::new().sqr(&mut reference, &a),
        "reference shape is admitted"
    );
    let mut state = SquaringBenchState::default();
    state.prepare(&mut dst, &a).run();
    assert_eq!(dst, reference, "production square matches NTT");
    let mut prepared = state.prepare(&mut dst, &a);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}
