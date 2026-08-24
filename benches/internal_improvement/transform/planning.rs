//! Production SSA planning cost on the geometry cache's hot path.
//!
//! The prepared-SSA rows elsewhere time the kernel with a plan already bound.
//! These rows time the other half of a first product at a new width: the
//! geometry-cache hit and the operand-bound plan build, for both executors.
//! `Allocating` means runner construction allocates no execution scratch.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{
    Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{SSA_PLANNING_SIZES, operands};

#[divan::bench(args = SSA_PLANNING_SIZES)]
fn ssa_cached_planning_sequential(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa_cached_planning(bencher, len, TransformExecutor::Sequential);
}

#[divan::bench(args = SSA_PLANNING_SIZES)]
fn ssa_cached_planning_parallel(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa_cached_planning(bencher, len, TransformExecutor::Default);
}

fn bench_ssa_cached_planning(
    bencher: divan::Bencher<'_, '_>,
    len: usize,
    executor: TransformExecutor,
) {
    let (left, right, _destination) = operands(len);
    let _warm = Tuner::bench_ssa_multiplication(
        SsaGeometryPolicy::Production,
        executor,
        SsaScratchPolicy::Allocating,
        &left,
        &right,
    );
    bencher.bench_local(|| {
        black_box(Tuner::bench_ssa_multiplication(
            SsaGeometryPolicy::Production,
            executor,
            SsaScratchPolicy::Allocating,
            black_box(&left),
            black_box(&right),
        ))
    });
}
