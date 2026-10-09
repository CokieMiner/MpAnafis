//! Timed execution of prepared operands and reusable destinations.

use divan::{Bencher, black_box};

/// Runs a prepared batch without timing setup, conversion, or verification.
pub fn measure<I, O>(bencher: Bencher, inputs: &[I], operation: impl Fn(&I) -> O) {
    bencher.bench_local(|| {
        for input in inputs {
            let _output = black_box(operation(black_box(input)));
        }
    });
}

/// Reuses initialized output storage across every measured operation.
pub fn measure_assignment<I, S>(
    bencher: Bencher,
    inputs: &[I],
    mut state: S,
    operation: impl Fn(&mut S, &I),
) {
    bencher.bench_local(|| {
        for input in inputs {
            operation(black_box(&mut state), black_box(input));
            let _output = black_box(&state);
        }
    });
}

/// Clones the initial receiver outside timing and times only its mutation.
pub fn measure_mutation<I: Clone>(bencher: Bencher, input: I, operation: impl Fn(&mut I)) {
    bencher
        .with_inputs(|| input.clone())
        .bench_local_refs(|state| {
            operation(black_box(&mut *state));
            let _output = black_box(state);
        });
}
