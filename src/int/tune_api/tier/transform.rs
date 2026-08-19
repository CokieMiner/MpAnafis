//! Prepared NTT and SSA benchmark state plus transform inspection entry points.

use core::{num::NonZeroUsize, ptr::fn_addr_eq};

use alloc::{vec, vec::Vec};

#[cfg(feature = "rayon")]
pub use crate::parallel::RayonExecutor;
pub use crate::parallel::{DefaultExecutor, ParallelExecutor, SequentialExecutor};

use super::{
    ArchKernels, BenchValidation, Limb, Ntt, NttMultiplicationPlan, Ssa, SsaMultiplicationPlan,
    TransformBench, TransformChoice, TransformPlan, Tuner,
};

const NTT_KERNEL_BENCH_PRIME: u32 = 2_147_483_647;
// p = 2^31 - 1 satisfies p^2 = 1 (mod 2^32), so -p^-1 is 2^32 - p.
const NTT_KERNEL_BENCH_NEG_INVERSE: u32 = 2_147_483_649;
type NttRadix4Kernel = unsafe fn(*mut u32, *const u32, usize, u32, u32);

/// Executor selected for one transform benchmark row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransformExecutor {
    /// Never schedules work on another thread.
    Sequential,
    /// Uses the crate's feature-selected executor.
    Default,
}

/// NTT plan policy selected before timing begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NttPlanPolicy {
    /// Use the production plan selected for the operand widths.
    Production,
    /// Use an exact digit width and modulus count.
    Forced {
        /// Bits stored in each transform digit.
        digit_bits: u32,
        /// Number of prime moduli used for CRT reconstruction.
        modulus_count: usize,
    },
}

/// Resolved NTT geometry recorded in benchmark labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct NttPlanGeometry {
    /// Bits stored in each transform digit.
    pub digit_bits: u32,
    /// Number of prime moduli used by the transform.
    pub modulus_count: usize,
}

/// Workspace ownership policy selected for one NTT row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NttScratchPolicy {
    /// Allocate exact transform workspace inside every measured call.
    Allocating,
    /// Allocate exact 32-bit and Goldilocks workspaces once and reuse them.
    Reusable,
}

/// Architecture backend selected for a fused radix-4 kernel row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NttKernelBackend {
    /// Use the backend selected by the production runtime dispatcher.
    RuntimeSelected,
    /// Force the portable scalar reference kernel.
    ScalarReference,
}

/// Direction selected for a fused radix-4 kernel row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NttKernelDirection {
    /// Decimation in frequency.
    Dif,
    /// Decimation in time.
    Dit,
}

/// Validated, allocation-free fused radix-4 benchmark state.
#[derive(Debug)]
#[non_exhaustive]
pub struct NttRadix4Runner {
    kernel: NttRadix4Kernel,
    values: Vec<u32>,
    twiddles: Vec<u32>,
    quarter_len: usize,
}

/// SSA geometry policy selected before timing begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SsaGeometryPolicy {
    /// Force the transform while retaining the planner's geometry.
    Forced,
    /// Use the production transform/basecase and geometry decision.
    Production,
    /// Force one exact top-level transform exponent.
    ForcedExponent(u32),
}

/// Workspace ownership policy selected for one SSA row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SsaScratchPolicy {
    /// Let SSA allocate its transform workspace inside every measured call.
    Allocating,
    /// Allocate exact workspace during benchmark construction and reuse it.
    Reusable,
}

/// Operand-bound NTT state with explicit plan, executor, and scratch policy.
#[derive(Debug)]
#[non_exhaustive]
pub struct NttMultiplicationRunner<'operands> {
    plan: NttMultiplicationPlan<'operands>,
    executor: TransformExecutor,
    scratch: Option<NttScratch>,
}

#[derive(Debug)]
struct NttScratch {
    words: Vec<u32>,
    goldilocks: Vec<u64>,
}

/// Borrowed NTT call whose plan and buffer shape were validated before timing.
#[derive(Debug)]
#[non_exhaustive]
pub struct PreparedNttMultiplication<'runner, 'operands, 'buffers> {
    runner: &'runner mut NttMultiplicationRunner<'operands>,
    dst: &'buffers mut [Limb],
}

/// Shape-validated SSA state with explicit geometry, executor, and scratch policy.
#[derive(Debug)]
#[non_exhaustive]
pub struct SsaMultiplicationRunner<'operands> {
    plan: SsaMultiplicationPlan<'operands>,
    executor: TransformExecutor,
    scratch: Option<Vec<Limb>>,
}

/// Borrowed SSA call whose configuration and buffer shape were validated.
#[derive(Debug)]
#[non_exhaustive]
pub struct PreparedSsaMultiplication<'runner, 'operands, 'buffers> {
    runner: &'runner mut SsaMultiplicationRunner<'operands>,
    dst: &'buffers mut [Limb],
}

impl TransformExecutor {
    /// Returns the scheduling capacity recorded with the benchmark row.
    #[must_use]
    pub fn parallelism(self) -> NonZeroUsize {
        match self {
            Self::Sequential => SequentialExecutor.parallelism(),
            Self::Default => DefaultExecutor::default().parallelism(),
        }
    }
}

impl NttPlanPolicy {
    /// Resolves production or forced geometry before benchmark timing begins.
    #[must_use]
    pub fn resolve(self, len_a: usize, len_b: usize) -> Option<NttPlanGeometry> {
        let plan = match self {
            Self::Production => Ntt::choose_transform_plan(len_a, len_b)?,
            Self::Forced {
                digit_bits,
                modulus_count,
            } => TransformPlan {
                digit_bits,
                modulus_count,
            },
        };
        plan.is_valid().then_some(NttPlanGeometry {
            digit_bits: plan.digit_bits,
            modulus_count: plan.modulus_count,
        })
    }
}

impl NttPlanGeometry {
    const fn transform(self) -> TransformPlan {
        TransformPlan {
            digit_bits: self.digit_bits,
            modulus_count: self.modulus_count,
        }
    }
}

impl NttKernelBackend {
    /// Stable label for the concrete code selected on this process.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::RuntimeSelected => match ArchKernels::ntt_radix4_selected_backend_name() {
                "avx2" => "runtime-selected-avx2",
                "neon" => "runtime-selected-neon",
                "scalar" => "runtime-selected-scalar",
                _ => "runtime-selected-unknown",
            },
            Self::ScalarReference => "scalar-reference",
        }
    }

    /// Whether this backend/direction resolves to code distinct from the
    /// scalar reference on the current process and target.
    #[must_use]
    pub fn differs_from_scalar(self, direction: NttKernelDirection) -> bool {
        if self == Self::ScalarReference {
            return false;
        }
        let selected = ArchKernels::ntt_radix4_selected_kernels();
        let scalar = ArchKernels::ntt_radix4_scalar_kernels();
        match direction {
            NttKernelDirection::Dif => !fn_addr_eq(selected.0, scalar.0),
            NttKernelDirection::Dit => !fn_addr_eq(selected.1, scalar.1),
        }
    }
}

impl NttRadix4Runner {
    /// Executes one complete fused stage over the prepared residue spans.
    pub fn run(&mut self) {
        // SAFETY: construction allocated four writable value quarters and two
        // readable twiddle quarters, populated their documented residue ranges,
        // and selected a kernel with the identical span contract.
        unsafe {
            (self.kernel)(
                self.values.as_mut_ptr(),
                self.twiddles.as_ptr(),
                self.quarter_len,
                NTT_KERNEL_BENCH_PRIME,
                NTT_KERNEL_BENCH_NEG_INVERSE,
            );
        }
    }
}

impl SsaGeometryPolicy {
    const fn choice(self) -> TransformChoice {
        match self {
            Self::Forced => TransformChoice::FORCED,
            Self::Production => TransformChoice::PLANNED,
            Self::ForcedExponent(exponent) => TransformChoice::forced_at(exponent),
        }
    }
}

impl PreparedNttMultiplication<'_, '_, '_> {
    /// Runs the NTT without plan selection, allocation, or repeated validation
    /// when reusable scratch was selected.
    pub fn run(&mut self) {
        self.runner.run_kernel(self.dst);
    }
}

impl<'operands> NttMultiplicationRunner<'operands> {
    fn new(
        policy: NttPlanPolicy,
        executor: TransformExecutor,
        scratch_policy: NttScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<Self> {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "NTT operands must be nonzero-width"
        );
        let transform = policy.resolve(a.len(), b.len())?.transform();
        let plan = NttMultiplicationPlan::try_new(a, b, transform)?;
        let scratch = match scratch_policy {
            NttScratchPolicy::Allocating => None,
            NttScratchPolicy::Reusable => Some(NttScratch {
                words: vec![u32::MIN; plan.scratch_u32_len()],
                goldilocks: vec![u64::MIN; plan.scratch_u64_len()],
            }),
        };
        Some(Self {
            plan,
            executor,
            scratch,
        })
    }

    /// Prepares an exact borrowed NTT call for repeated measurements.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
    ) -> PreparedNttMultiplication<'runner, 'operands, 'buffers> {
        assert_eq!(
            dst.len(),
            self.plan.destination_len(),
            "NTT destination width changed"
        );
        PreparedNttMultiplication { runner: self, dst }
    }

    fn run_kernel(&mut self, dst: &mut [Limb]) {
        match self.executor {
            TransformExecutor::Sequential => {
                self.run_with_executor(dst, &SequentialExecutor);
            }
            TransformExecutor::Default => {
                self.run_with_executor(dst, &DefaultExecutor::default());
            }
        }
    }

    fn run_with_executor<E: ParallelExecutor>(&mut self, dst: &mut [Limb], executor: &E) {
        if let Some(scratch) = &mut self.scratch {
            // SAFETY: construction allocated both exact workspaces and
            // `prepare` validated the immutable plan's destination width.
            unsafe {
                self.plan.run_with_scratch(
                    dst,
                    &mut scratch.words,
                    &mut scratch.goldilocks,
                    executor,
                );
            }
        } else {
            // SAFETY: `prepare` validated the exact destination width.
            unsafe { self.plan.run_allocating(dst, executor) }
        }
    }
}

impl PreparedSsaMultiplication<'_, '_, '_> {
    /// Runs SSA without repeating facade validation or allocating reusable scratch.
    pub fn run(&mut self) {
        self.runner.run_kernel(self.dst);
    }
}

impl<'operands> SsaMultiplicationRunner<'operands> {
    fn new(
        geometry: SsaGeometryPolicy,
        executor: TransformExecutor,
        scratch_policy: SsaScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<Self> {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "SSA operands must be nonzero-width"
        );
        assert!(
            Ssa::admits_mul(a.len(), b.len()),
            "SSA cannot represent the benchmark product width"
        );
        let plan = SsaMultiplicationPlan::try_new(a, b, geometry.choice(), executor.parallelism())?;
        let scratch = match scratch_policy {
            SsaScratchPolicy::Allocating => None,
            SsaScratchPolicy::Reusable => Some(vec![Limb::MIN; plan.scratch_len()]),
        };
        Some(Self {
            plan,
            executor,
            scratch,
        })
    }

    /// Prepares an exact borrowed SSA call for repeated measurements.
    pub fn prepare<'runner, 'buffers>(
        &'runner mut self,
        dst: &'buffers mut [Limb],
    ) -> PreparedSsaMultiplication<'runner, 'operands, 'buffers> {
        assert_eq!(
            dst.len(),
            self.plan.destination_len(),
            "SSA destination width changed"
        );
        PreparedSsaMultiplication { runner: self, dst }
    }

    fn run_kernel(&mut self, dst: &mut [Limb]) {
        match self.executor {
            TransformExecutor::Sequential => {
                self.run_with_executor(dst, &SequentialExecutor);
            }
            TransformExecutor::Default => {
                self.run_with_executor(dst, &DefaultExecutor::default());
            }
        }
    }

    fn run_with_executor<E: ParallelExecutor>(&mut self, dst: &mut [Limb], executor: &E) {
        if let Some(scratch) = self.scratch.as_deref_mut() {
            // SAFETY: construction allocated the exact plan scratch and
            // `prepare` validated the exact destination width. The executor is
            // the same policy whose parallelism was recorded in the plan.
            unsafe { self.plan.run_with_scratch(dst, scratch, executor) }
        } else {
            // SAFETY: `prepare` validated the destination width and this is the
            // same executor policy used during plan construction.
            unsafe { self.plan.run_allocating(dst, executor) }
        }
    }
}

impl Tuner {
    /// Creates a warmed and scalar-validated fused radix-4 kernel row.
    #[must_use]
    pub fn bench_ntt_radix4_kernel(
        backend: NttKernelBackend,
        direction: NttKernelDirection,
        quarter_len: usize,
    ) -> NttRadix4Runner {
        assert!(quarter_len != 0, "radix-4 quarter width must be nonzero");
        let value_len = quarter_len
            .checked_mul(4)
            .expect("radix-4 value width overflows usize");
        let twiddle_len = quarter_len
            .checked_mul(2)
            .expect("radix-4 twiddle width overflows usize");
        let selected = ArchKernels::ntt_radix4_selected_kernels();
        let scalar = ArchKernels::ntt_radix4_scalar_kernels();
        let kernel = match (backend, direction) {
            (NttKernelBackend::RuntimeSelected, NttKernelDirection::Dif) => selected.0,
            (NttKernelBackend::RuntimeSelected, NttKernelDirection::Dit) => selected.1,
            (NttKernelBackend::ScalarReference, NttKernelDirection::Dif) => scalar.0,
            (NttKernelBackend::ScalarReference, NttKernelDirection::Dit) => scalar.1,
        };
        let scalar_reference = match direction {
            NttKernelDirection::Dif => scalar.0,
            NttKernelDirection::Dit => scalar.1,
        };
        let two_prime = NTT_KERNEL_BENCH_PRIME
            .checked_mul(2)
            .expect("benchmark modulus is below 2^31");
        let values: Vec<u32> = (0..value_len)
            .map(|index| {
                let wide_index = u64::try_from(index).expect("slice indices fit in u64");
                let residue = wide_index
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(17)
                    .rem_euclid(u64::from(two_prime));
                u32::try_from(residue).expect("lazy residue fits in u32")
            })
            .collect();
        let twiddles: Vec<u32> = (0..twiddle_len)
            .map(|index| {
                let wide_index = u64::try_from(index).expect("slice indices fit in u64");
                let residue = wide_index
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(1)
                    .rem_euclid(u64::from(NTT_KERNEL_BENCH_PRIME));
                u32::try_from(residue).expect("Montgomery residue fits in u32")
            })
            .collect();
        let mut warmed = values.clone();
        let mut expected = values;
        // SAFETY: all three allocations have the complete disjoint spans and
        // residue bounds established immediately above.
        unsafe {
            kernel(
                warmed.as_mut_ptr(),
                twiddles.as_ptr(),
                quarter_len,
                NTT_KERNEL_BENCH_PRIME,
                NTT_KERNEL_BENCH_NEG_INVERSE,
            );
        }
        // SAFETY: the scalar reference has the identical validated contract.
        unsafe {
            scalar_reference(
                expected.as_mut_ptr(),
                twiddles.as_ptr(),
                quarter_len,
                NTT_KERNEL_BENCH_PRIME,
                NTT_KERNEL_BENCH_NEG_INVERSE,
            );
        }
        assert_eq!(
            warmed, expected,
            "selected NTT radix-4 kernel differs from the scalar reference"
        );
        NttRadix4Runner {
            kernel,
            values: warmed,
            twiddles,
            quarter_len,
        }
    }

    /// Creates reusable state for one explicitly configured NTT benchmark row.
    #[must_use]
    pub fn bench_ntt_multiplication<'operands>(
        policy: NttPlanPolicy,
        executor: TransformExecutor,
        scratch: NttScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<NttMultiplicationRunner<'operands>> {
        NttMultiplicationRunner::new(policy, executor, scratch, a, b)
    }

    /// Creates reusable state for one explicitly configured SSA benchmark row.
    ///
    /// Returns `None` only when a forced exponent is invalid for the shape.
    #[must_use]
    pub fn bench_ssa_multiplication<'operands>(
        geometry: SsaGeometryPolicy,
        executor: TransformExecutor,
        scratch: SsaScratchPolicy,
        a: &'operands [Limb],
        b: &'operands [Limb],
    ) -> Option<SsaMultiplicationRunner<'operands>> {
        SsaMultiplicationRunner::new(geometry, executor, scratch, a, b)
    }

    /// Execute the full-width Fermat-ring product used inside recursive SSA.
    ///
    /// # Panics
    ///
    /// Panics unless the operands and destination describe a supported equal-width
    /// Fermat-ring product.
    pub fn bench_ssa_fermat_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
        assert_eq!(a.len(), b.len(), "Fermat benchmark widths differ");
        let completed = TransformBench::ssa_fermat_mul(dst, a, b, None);
        assert!(completed, "benchmark dimensions exceed the SSA size bounds");
    }

    /// Execute a full-width Fermat product with a forced transform geometry.
    ///
    /// # Panics
    ///
    /// Panics unless the widths and forced geometry are valid.
    pub fn bench_ssa_fermat_mul_forced_plan(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        transform_exponent: u32,
    ) {
        assert_eq!(a.len(), b.len(), "Fermat benchmark widths differ");
        let completed = TransformBench::ssa_fermat_mul(dst, a, b, Some(transform_exponent));
        assert!(completed, "forced Fermat geometry exceeds its ring bounds");
    }

    /// Return exact scratch for a Mersenne-ring product.
    #[must_use]
    pub fn bench_ssa_mersenne_mul_scratch_len(len: usize) -> usize {
        TransformBench::ssa_mul_mod_bnm1_scratch_len(len)
    }

    /// Execute the recursive Mersenne-ring half of SSA with caller-owned scratch.
    ///
    /// # Panics
    ///
    /// Panics unless all data spans have equal width and scratch is sufficient.
    pub fn bench_ssa_mersenne_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        assert_eq!(a.len(), b.len(), "Mersenne benchmark widths differ");
        assert_eq!(dst.len(), a.len(), "Mersenne destination width differs");
        BenchValidation::scratch(
            scratch,
            TransformBench::ssa_mul_mod_bnm1_scratch_len(a.len()),
        );
        TransformBench::ssa_mul_mod_bnm1(dst, a, b, scratch);
    }
}
