//! Controlled Mp NTT/SSA execution-policy matrix.
//!
//! Every argument label names four independent dimensions: engine, geometry
//! policy, executor and its reported parallelism, and workspace ownership.
//! Both engines include like-for-like allocating and reusable-scratch rows for
//! every available executor.

use core::{fmt, hint::black_box};

use mp_anafis::tune_api::tier::{
    Tuner,
    transform::{
        NttPlanGeometry, NttPlanPolicy, NttScratchPolicy, SsaGeometryPolicy, SsaScratchPolicy,
        TransformExecutor,
    },
};

use crate::shared::{SCALING_SIZES, gmp_equal_reference, operands, validate_and_warm_product};

#[derive(Clone, Copy, Debug)]
enum TransformCaseKind {
    Ntt {
        policy: NttPlanPolicy,
        geometry: NttPlanGeometry,
        executor: TransformExecutor,
        scratch: NttScratchPolicy,
    },
    Ssa {
        geometry: SsaGeometryPolicy,
        executor: TransformExecutor,
        scratch: SsaScratchPolicy,
    },
}

#[derive(Clone, Copy, Debug)]
struct TransformCase {
    len: usize,
    kind: TransformCaseKind,
}

impl fmt::Display for TransformCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TransformCaseKind::Ntt {
                policy,
                geometry,
                executor,
                scratch,
            } => write!(
                formatter,
                "ntt/{}/{}/{}/{}-limbs",
                ntt_policy_label(policy, geometry),
                executor_label(executor),
                ntt_scratch_label(scratch),
                self.len
            ),
            TransformCaseKind::Ssa {
                geometry,
                executor,
                scratch,
            } => write!(
                formatter,
                "ssa/{}/{}/{}/{}-limbs",
                geometry_label(geometry),
                executor_label(executor),
                scratch_label(scratch),
                self.len
            ),
        }
    }
}

#[divan::bench(args = transform_cases())]
fn mp_transform_matrix(bencher: divan::Bencher<'_, '_>, case: TransformCase) {
    let (left, right, mut destination) = operands(case.len);
    let expected = gmp_equal_reference(&left, &right);
    match case.kind {
        TransformCaseKind::Ntt {
            policy,
            executor,
            scratch,
            ..
        } => {
            let mut runner =
                Tuner::bench_ntt_multiplication(policy, executor, scratch, &left, &right)
                    .expect("matrix NTT plan is valid for this width");
            validate_and_warm_product(&expected, "prepared NTT product", |probe| {
                runner.prepare(probe).run();
            });
            let mut prepared = runner.prepare(&mut destination);
            bencher.bench_local(|| black_box(&mut prepared).run());
        }
        TransformCaseKind::Ssa {
            geometry,
            executor,
            scratch,
        } => {
            let mut runner =
                Tuner::bench_ssa_multiplication(geometry, executor, scratch, &left, &right)
                    .expect("matrix SSA geometry is valid for this width");
            validate_and_warm_product(&expected, "prepared SSA product", |probe| {
                runner.prepare(probe).run();
            });
            let mut prepared = runner.prepare(&mut destination);
            bencher.bench_local(|| black_box(&mut prepared).run());
        }
    }
}

fn transform_cases() -> Vec<TransformCase> {
    let executors = transform_executors();
    let mut cases = Vec::with_capacity(
        SCALING_SIZES
            .len()
            .saturating_mul(executors.len())
            .saturating_mul(8),
    );
    for len in SCALING_SIZES {
        for &executor in &executors {
            for policy in [
                NttPlanPolicy::Production,
                NttPlanPolicy::Forced {
                    digit_bits: 31,
                    modulus_count: 3,
                },
            ] {
                let geometry = policy
                    .resolve(len, len)
                    .expect("matrix NTT policy resolves for this width");
                for scratch in [NttScratchPolicy::Allocating, NttScratchPolicy::Reusable] {
                    cases.push(TransformCase {
                        len,
                        kind: TransformCaseKind::Ntt {
                            policy,
                            geometry,
                            executor,
                            scratch,
                        },
                    });
                }
            }
            for geometry in [SsaGeometryPolicy::Forced, SsaGeometryPolicy::Production] {
                for scratch in [SsaScratchPolicy::Allocating, SsaScratchPolicy::Reusable] {
                    cases.push(TransformCase {
                        len,
                        kind: TransformCaseKind::Ssa {
                            geometry,
                            executor,
                            scratch,
                        },
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

fn ntt_policy_label(policy: NttPlanPolicy, geometry: NttPlanGeometry) -> String {
    let selection = match policy {
        NttPlanPolicy::Production => "production",
        NttPlanPolicy::Forced { .. } => "forced",
        _ => "unknown",
    };
    let field = if geometry.modulus_count == 1 {
        "goldilocks"
    } else {
        "monty-u32"
    };
    format!(
        "{selection}-{}bit-{}prime-{field}",
        geometry.digit_bits, geometry.modulus_count
    )
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

const fn ntt_scratch_label(scratch: NttScratchPolicy) -> &'static str {
    match scratch {
        NttScratchPolicy::Allocating => "allocating",
        NttScratchPolicy::Reusable => "reusable-scratch",
        _ => "unknown-scratch",
    }
}
