//! Controlled Mp SSA execution-policy matrix.
//!
//! Every argument label names independent dimensions: geometry policy,
//! executor and its reported parallelism, and workspace ownership.
//! Includes like-for-like allocating and reusable-scratch rows for
//! every available executor.

use core::{fmt, hint::black_box};

extern crate alloc;

use alloc::{format, string::String, vec, vec::Vec};

use mp_anafis::tune_api::tier::{
    Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{SCALING_SIZES, gmp_equal_reference, operands, validate_and_warm_product};

#[derive(Clone, Copy, Debug)]
struct TransformCase {
    len: usize,
    geometry: SsaGeometryPolicy,
    executor: TransformExecutor,
    scratch: SsaScratchPolicy,
}

impl fmt::Display for TransformCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ssa/{}/{}/{}/{}-limbs",
            geometry_label(self.geometry),
            executor_label(self.executor),
            scratch_label(self.scratch),
            self.len
        )
    }
}

#[divan::bench(args = transform_cases())]
fn mp_transform_matrix(bencher: divan::Bencher<'_, '_>, case: TransformCase) {
    let (left, right, mut destination) = operands(case.len);
    let expected = gmp_equal_reference(&left, &right);
    let mut runner =
        Tuner::bench_ssa_multiplication(case.geometry, case.executor, case.scratch, &left, &right)
            .expect("matrix SSA geometry is valid for this width");
    validate_and_warm_product(&expected, "prepared SSA product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

fn transform_cases() -> Vec<TransformCase> {
    let executors = transform_executors();
    let mut cases = Vec::with_capacity(
        SCALING_SIZES
            .len()
            .saturating_mul(executors.len())
            .saturating_mul(4),
    );
    for len in SCALING_SIZES {
        for &executor in &executors {
            for geometry in [SsaGeometryPolicy::Forced, SsaGeometryPolicy::Production] {
                for scratch in [SsaScratchPolicy::Allocating, SsaScratchPolicy::Reusable] {
                    cases.push(TransformCase {
                        len,
                        geometry,
                        executor,
                        scratch,
                    });
                }
            }
        }
    }
    cases
}

fn transform_executors() -> Vec<TransformExecutor> {
    let mut executors = vec![TransformExecutor::Sequential];
    let default = TransformExecutor::Default;
    if cfg!(feature = "rayon") && default.parallelism().get() > 1 {
        executors.push(default);
    }
    executors
}

fn executor_label(executor: TransformExecutor) -> String {
    let name = match executor {
        TransformExecutor::Sequential => "sequential",
        TransformExecutor::Default if cfg!(feature = "rayon") => "default-rayon",
        TransformExecutor::Default => "default-sequential",
        _ => "unknown-executor",
    };
    format!("{name}-p{}", executor.parallelism())
}

const fn geometry_label(geometry: SsaGeometryPolicy) -> &'static str {
    match geometry {
        SsaGeometryPolicy::Forced => "forced",
        SsaGeometryPolicy::Production => "production",
        SsaGeometryPolicy::ForcedExponent(_) => "forced-exponent",
        _ => "unknown-geometry",
    }
}

const fn scratch_label(scratch: SsaScratchPolicy) -> &'static str {
    match scratch {
        SsaScratchPolicy::Allocating => "allocating",
        SsaScratchPolicy::Reusable => "reusable-scratch",
        _ => "unknown-scratch",
    }
}
