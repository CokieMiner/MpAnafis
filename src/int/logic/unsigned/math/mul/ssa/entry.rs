//! SSA product and square boundaries, strategy selection, and workspace sizing.
//!
//! Capability and scratch sizing use declared widths. CRT computes residues
//! modulo `B^n + 1` and `B^n - 1` before exact reconstruction. The direct
//! strategy uses one full-width Fermat ring under its worker-budget policy.
//!
//! Tuning entry points can force either strategy and request transforms below
//! the planner's crossover through [`TransformChoice`].
//!
#![expect(
    unsafe_code,
    reason = "Validated operand and arena widths bound complete staging writes and operand-bound plan execution"
)]

use core::ptr::{copy_nonoverlapping, write_bytes};

use crate::parallel::{DefaultExecutor, ParallelExecutor};

use super::{
    CrtMulPlan, FftPlan, LIMB_BITS, Limb, LimbOutput, MUL_MOD_BNM1_THRESHOLD, MulScratch,
    Multiplication, SSA_BASE_MODULUS_BITS, SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS,
    SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD, ScratchBuffer, SsaCrt, SsaMultiplicationPlan,
    SsaOperation, SsaPlan, SsaSquaringPlan,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SsaProductStrategyChoice {
    #[default]
    Planned,
    #[cfg(feature = "_internal-tune")]
    CrtTwoModuli,
    #[cfg(any(test, feature = "_internal-tune"))]
    DirectFermat,
}

/// Namespace for recursive Schonhage-Strassen multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Ssa;

/// How a top-level SSA product or square picks its transform geometry.
///
/// Production uses [`Self::PLANNED`]. Tuning and tests can force the same
/// transform kernels below their crossover or select a specific product strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransformChoice {
    /// Transform even where the ring is narrow enough for the basecase.
    force: bool,
    strategy: SsaProductStrategyChoice,
}

impl TransformChoice {
    /// Let the planner decide both the geometry and whether to transform at all.
    pub const PLANNED: Self = Self {
        force: false,
        strategy: SsaProductStrategyChoice::Planned,
    };

    /// Force the transform while leaving its geometry to the planner.
    #[cfg(any(test, feature = "_internal-tune"))]
    pub const FORCED: Self = Self {
        force: true,
        strategy: SsaProductStrategyChoice::Planned,
    };

    /// Force the two-modulus CRT product strategy for paired tuning.
    #[cfg(feature = "_internal-tune")]
    pub const FORCED_CRT: Self = Self {
        force: true,
        strategy: SsaProductStrategyChoice::CrtTwoModuli,
    };

    /// Force the single full-width Fermat product strategy for paired tuning.
    #[cfg(any(test, feature = "_internal-tune"))]
    pub const FORCED_DIRECT_FERMAT: Self = Self {
        force: true,
        strategy: SsaProductStrategyChoice::DirectFermat,
    };

    /// Whether the selected geometry must use a transform below the normal
    /// transform crossover.
    #[must_use]
    pub const fn forces_transform(self) -> bool {
        self.force
    }

    /// Compares direct squares against the complete square CRT objective.
    #[must_use]
    pub fn use_direct_square(self, half_width: usize) -> bool {
        match self.strategy {
            SsaProductStrategyChoice::Planned => {
                SsaPlan::direct_fermat_is_cheaper(half_width, SsaOperation::Square)
            }
            #[cfg(feature = "_internal-tune")]
            SsaProductStrategyChoice::CrtTwoModuli => false,
            #[cfg(any(test, feature = "_internal-tune"))]
            SsaProductStrategyChoice::DirectFermat => true,
        }
    }

    /// Resolve the product strategy once the planner has found the CRT
    /// half-width and the executor-specific crossover.
    ///
    /// The measured direct-Fermat policy covers power-of-two half-widths.
    /// Irregular widths also admit whole-bit twists by widening their inner
    /// ring; their relative cost requires comparing the resulting geometries.
    #[must_use]
    pub const fn use_direct_fermat(self, half_width: usize, threshold: Option<usize>) -> bool {
        match self.strategy {
            SsaProductStrategyChoice::Planned => match threshold {
                Some(minimum) => half_width >= minimum && half_width.is_power_of_two(),
                None => false,
            },
            #[cfg(feature = "_internal-tune")]
            SsaProductStrategyChoice::CrtTwoModuli => false,
            #[cfg(any(test, feature = "_internal-tune"))]
            SsaProductStrategyChoice::DirectFermat => true,
        }
    }
}

impl Multiplication {
    /// Computes a transform product modulo `B^w - 1`, with `w >= minimum`.
    ///
    /// Returns false, without changing the destination, when an operand exceeds
    /// `minimum` limbs, the shared cyclic-product policy rejects it, or no
    /// shorter representable Mersenne geometry exists. On success the complete
    /// destination is the residue; zero may be represented by all zero or all
    /// maximum limbs. The output width determines the modulus. `FORCED`
    /// bypasses only the empirical crossover for kernel verification and tuning;
    /// mathematical admission is always checked. `RESERVE_FULL_PRODUCT` reserves
    /// room for callers that append reconstructed high limbs after the residue.
    pub fn try_mul_mod_bnm1<const FORCED: bool, const RESERVE_FULL_PRODUCT: bool>(
        a: &[Limb],
        b: &[Limb],
        minimum: usize,
        dst: &mut ScratchBuffer,
        scratch: &mut MulScratch,
    ) -> bool {
        if (!FORCED && (MUL_MOD_BNM1_THRESHOLD == 0 || minimum < MUL_MOD_BNM1_THRESHOLD))
            || a.len() > minimum
            || b.len() > minimum
        {
            return false;
        }
        let Some(required_bits) = minimum.checked_mul(LIMB_BITS) else {
            return false;
        };
        // B^(2h)-1 splits into B^h+1 and B^h-1, exactly the recurrence priced
        // by the CRT geometry chooser. Its legal half-width gives w=2h.
        let Some(width) = SsaPlan::best_crt_half_width(required_bits, SsaOperation::Multiply)
            .and_then(|half| half.checked_mul(2))
        else {
            return false;
        };
        // SAFETY: both materialized limb spans have at least two bytes per
        // limb; their combined length is below usize::MAX on every target.
        let full_len = unsafe { a.len().unchecked_add(b.len()) };
        if width >= full_len {
            return false;
        }
        DefaultExecutor::with_resolved(|executor| {
            let parallelism = executor.parallelism().get();
            let work = SsaCrt::mul_mod_bnm1_scratch_len_for_parallelism(width, parallelism);
            let Some(staging_len) = width.checked_mul(2) else {
                return false;
            };
            let Some(total) = staging_len.checked_add(work) else {
                return false;
            };
            // SAFETY: doubling the legal half-width preserves every required
            // halving; the scratch calculation validates all descendant bit widths.
            let plan = unsafe { CrtMulPlan::new(width).unwrap_unchecked() };
            if scratch.buf.capacity() < total {
                scratch.buf = ScratchBuffer::acquire(total);
            }
            // Initialize staging before exposing its limb slice. The inputs
            // overwrite their prefixes; only the implicit high limbs are zero.
            // Only the remaining new workspace needs initialization before
            // borrowing the complete arena as limbs.
            let old_len = scratch.buf.len();
            let initialized_len = old_len.max(staging_len);
            // SAFETY: the exclusive pooled allocation has capacity >= total
            // >= 2*width. width>=minimum>=a.len(),b.len(), so copies and padding
            // writes cover both disjoint width-limb operands exactly. The input
            // borrows exclude aliasing with scratch. Old exposed limbs remain
            // initialized; staging and the remaining new workspace are written
            // before the single length commit exposes the complete limb prefix.
            unsafe {
                let left = scratch.buf.as_mut_ptr();
                let right = left.add(width);
                copy_nonoverlapping(a.as_ptr(), left, a.len());
                write_bytes(left.add(a.len()), 0, width.unchecked_sub(a.len()));
                copy_nonoverlapping(b.as_ptr(), right, b.len());
                write_bytes(right.add(b.len()), 0, width.unchecked_sub(b.len()));
                if initialized_len < total {
                    write_bytes(
                        left.add(initialized_len),
                        0,
                        total.unchecked_sub(initialized_len),
                    );
                }
                scratch.buf.set_len(old_len.max(total));
            }
            // SAFETY: the committed initialized prefix contains at least
            // total=2*width+work limbs. Both width-limb operands and the
            // remaining child workspace are disjoint contiguous partitions.
            let (left, right, inner) = unsafe {
                let (left, after_left) = scratch.buf.split_at_mut_unchecked(width);
                let (right, inner) = after_left.split_at_mut_unchecked(width);
                (left, right, inner)
            };
            let output_capacity = if RESERVE_FULL_PRODUCT {
                full_len
            } else {
                width
            };
            dst.reset_with_capacity(output_capacity);
            // SAFETY: reset_with_capacity reserves width writable elements;
            // spare_capacity_mut retains their possibly uninitialized type.
            let output = unsafe { dst.spare_capacity_mut().get_unchecked_mut(..width) };
            // SAFETY: both initialized operands have the plan's exact width;
            // inner retains at least work limbs sized for this executor. Each
            // CRT leaf and merge writes every output limb before reading it;
            // output is disjoint reserved storage and remains MaybeUninit here.
            unsafe {
                SsaCrt::mul_mod_bnm1_prepared(output, left, right, inner, executor, &plan);
            }
            // SAFETY: the completed CRT product initialized all width limbs.
            // No limb slice was exposed while that destination was uninitialized.
            unsafe {
                dst.set_len(width);
            }
            true
        })
    }
}

impl Ssa {
    /// Resolve the direct-Fermat crossover for one executor width.
    ///
    /// Returns `None` below the configured minimum worker count.
    #[must_use]
    pub const fn direct_fermat_threshold(parallelism: usize) -> Option<usize> {
        if parallelism < SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS
            || SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD == 0
        {
            None
        } else {
            Some(SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD)
        }
    }

    /// Multiply two limb slices with recursive Fermat-ring FFT multiplication.
    ///
    /// `scratch` of [`Ssa::mul_scratch_len_for_parallelism`] limbs for the supplied
    /// executor supplies all numeric workspace. Transform planning can allocate
    /// recursive plan metadata. The caller owns and reuses the numeric arena.
    /// Returns `false` when these widths have no representable CRT half-width, or
    /// when a pinned exponent yields no usable geometry.
    /// The executor routes every transform fork; small tiers may remain
    /// sequential by design.
    pub fn try_mul_with_executor<E: ParallelExecutor>(
        dst: &mut [impl LimbOutput],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        choice: TransformChoice,
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        let Some(plan) =
            SsaMultiplicationPlan::try_new(a_limbs, b_limbs, choice, executor.parallelism())
        else {
            return false;
        };
        if dst.len() < plan.result_len || scratch.len() < plan.scratch_len {
            return false;
        }
        // SAFETY: this boundary validates destination, scratch, and
        // executor width against the exact operand-bound plan.
        unsafe {
            plan.run_with_scratch(dst, scratch, executor);
        }
        true
    }

    /// Whether [`Self::try_mul_with_executor`] can compute a product of these operand widths.
    ///
    /// The dispatcher applies `SSA_THRESHOLD` separately from geometry admission.
    ///
    /// Geometry and scratch follow declared widths; significant widths prune
    /// execution without changing the selected ring or its workspace.
    #[must_use]
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        let Some(product_width) = len_a.checked_add(len_b) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// The squaring counterpart of [`Self::admits_mul`].
    #[must_use]
    pub fn admits_sqr(len: usize) -> bool {
        let Some(product_width) = len.checked_mul(2) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// Scratch required by [`Self::try_mul_with_executor`] for an executor
    /// advertising `parallelism` scheduling lanes.
    #[must_use]
    pub fn mul_scratch_len_for_parallelism(
        len_a: usize,
        len_b: usize,
        parallelism: usize,
    ) -> usize {
        let Some(half_width) =
            SsaPlan::best_crt_half_width_for_operands(len_a, len_b, SsaOperation::Multiply)
        else {
            return 0;
        };
        // Forced calls transform even a narrow ring. Planned calls can use
        // the basecase there, whose arena can exceed the transform arena.
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        // Strategy selection uses declared widths, exactly as the operand-bound
        // plan does. Significant bits only prune work inside that geometry.
        if TransformChoice::PLANNED
            .use_direct_fermat(half_width, Self::direct_fermat_threshold(parallelism))
        {
            let Some(direct_bits) = ring_bits.checked_mul(2) else {
                return 0;
            };
            let scratch = FftPlan::new(direct_bits).transform_mul_scratch(parallelism);
            return if scratch == usize::MAX { 0 } else { scratch };
        }
        let ring_plan = FftPlan::new(ring_bits);
        let transformed = ring_plan.transform_mul_scratch(parallelism);
        let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
            transformed.max(ring_plan.required_mul_scratch())
        } else {
            transformed
        };
        // A parallel executor evaluates the two CRT halves concurrently, so the
        // caller workspace must cover both halves' staging and workspaces at
        // once. This mirrors the prepared plan's concurrent layout exactly.
        let crt_scratch = if parallelism > 1 {
            SsaCrt::layout_len_concurrent(half_width, ring_scratch, parallelism)
        } else {
            SsaCrt::layout_len(half_width, ring_scratch, parallelism)
        };
        if crt_scratch == usize::MAX {
            0
        } else {
            crt_scratch
        }
    }

    /// Scratch required by [`Self::try_sqr_with_executor`] for an executor
    /// advertising `parallelism` scheduling lanes.
    #[must_use]
    pub fn sqr_scratch_len_for_parallelism(len: usize, parallelism: usize) -> usize {
        let Some(required_bits) = len
            .checked_mul(2)
            .and_then(|width| width.checked_mul(LIMB_BITS))
        else {
            return 0;
        };
        let Some(half_width) = SsaPlan::best_crt_half_width(required_bits, SsaOperation::Square)
        else {
            return 0;
        };
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        // Reserve both forced-transform and unforced basecase execution for a
        // narrow ring. Squares minimize their own geometry.
        if TransformChoice::PLANNED.use_direct_square(half_width) {
            let Some(direct_bits) = ring_bits.checked_mul(2) else {
                return 0;
            };
            let scratch = FftPlan::new_for_square(direct_bits).transform_sqr_scratch(parallelism);
            return if scratch == usize::MAX { 0 } else { scratch };
        }
        let ring_plan = FftPlan::new_for_square(ring_bits);
        let transformed = ring_plan.transform_sqr_scratch(parallelism);
        let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
            transformed.max(ring_plan.required_sqr_scratch())
        } else {
            transformed
        };
        if ring_scratch == usize::MAX {
            return 0;
        }
        SsaCrt::sqr_layout_len(half_width, ring_scratch, parallelism)
    }

    /// Square a limb slice with recursive Fermat-ring FFT squaring.
    ///
    /// CRT uses smaller Fermat rings and a recursive Mersenne square. A direct
    /// Fermat square also retains one forward transform and pointwise squares;
    /// its tradeoff is the larger ring geometry and simultaneous storage.
    ///
    /// Takes a [`TransformChoice`] and caller-owned scratch, using a
    /// caller-selected synchronous executor. The transform and CRT geometry
    /// are identical for every executor.
    pub fn try_sqr_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        choice: TransformChoice,
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        if a_limbs.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(plan) = SsaSquaringPlan::try_new(a_limbs, choice, executor.parallelism().get())
        else {
            return false;
        };
        if dst.len() < plan.result_len || scratch.len() < plan.scratch_len {
            return false;
        }
        // SAFETY: the boundary validates the destination and caller-owned
        // scratch before entering the infallible prepared executor.
        unsafe {
            plan.run_with_scratch(dst, scratch, executor);
        }
        true
    }
}

/// Whether a product this many limbs wide has a representable CRT half-width.
fn admits_product_width(product_width: usize) -> bool {
    let Some(required_bits) = product_width.checked_mul(LIMB_BITS) else {
        return false;
    };
    let Some(half_width) = SsaPlan::crt_half_width(required_bits) else {
        return false;
    };
    half_width.checked_mul(LIMB_BITS).is_some()
}
