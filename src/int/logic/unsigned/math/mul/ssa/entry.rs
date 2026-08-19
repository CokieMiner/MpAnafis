//! The [`Ssa`] namespace: the tower's product and square, the geometry choice
//! they take, and the answers derived from operand widths alone.
//!
//! Capability and scratch length both follow from the declared widths, which is
//! what lets the dispatcher decide a tier and size a buffer before any operand is
//! inspected. Each entry point stages its operands into the `B^n + 1` and
//! `B^n - 1` halves and hands both residues to the shared exact reconstruction in
//! [`crt`](super::crt).
//!
//! The tuner-only surface — forcing a transform the planner would decline, and the
//! full-width Fermat product that bypasses the CRT split — is not here: it lives
//! on the multiplication tuning namespace.
//!
//! The whole module is gated off 16-bit targets by its declaration in the parent,
//! because the CRT half-widths it computes cannot be represented there.

use alloc::vec;

use crate::parallel::ParallelExecutor;

use super::{FftPlan, LIMB_BITS, Limb, SsaCrt, SsaMultiplicationPlan, SsaPlan, SsaSquaringPlan};

/// Namespace for recursive Schönhage-Strassen multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ssa;

/// How a top-level SSA product or square picks its transform geometry.
///
/// Production always passes [`Self::PLANNED`]. The tuner and the tier tests force
/// a transform on rings the planner would otherwise leave to the basecase, and can
/// pin the exponent, so a forced measurement runs through the *same*
/// implementation as production rather than a parallel copy of it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransformChoice {
    /// Transform even where the ring is narrow enough for the basecase.
    pub force: bool,
    /// Use this transform exponent instead of the planner's choice.
    pub exponent: Option<u32>,
}

impl TransformChoice {
    /// Let the planner decide both the geometry and whether to transform at all.
    pub const PLANNED: Self = Self {
        force: false,
        exponent: None,
    };

    /// Force the transform while leaving its geometry to the planner.
    #[cfg(any(test, feature = "_internal-tune"))]
    pub const FORCED: Self = Self {
        force: true,
        exponent: None,
    };

    /// Force one exact transform exponent for measurement.
    #[cfg(feature = "_internal-tune")]
    pub const fn forced_at(exponent: u32) -> Self {
        Self {
            force: true,
            exponent: Some(exponent),
        }
    }
}

impl Ssa {
    /// Multiply two limb slices with recursive Fermat-ring FFT multiplication.
    ///
    /// `scratch` of [`Ssa::mul_scratch_len`] limbs keeps the call allocation-free;
    /// `None` makes it allocate its own, which only the tuner and the tier tests do.
    /// Returns `false` when these widths have no representable CRT half-width, or
    /// when a pinned exponent yields no usable geometry.
    /// The executor routes every transform fork; small tiers may remain
    /// sequential by design.
    pub fn try_mul_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        choice: TransformChoice,
        scratch: Option<&mut [Limb]>,
        executor: &E,
    ) -> bool {
        let Some(plan) =
            SsaMultiplicationPlan::try_new(a_limbs, b_limbs, choice, executor.parallelism())
        else {
            return false;
        };
        if dst.len() < plan.destination_len() {
            return false;
        }
        match scratch {
            Some(workspace) if workspace.len() >= plan.scratch_len() => {
                // SAFETY: this boundary validates destination, scratch, and
                // executor width against the exact operand-bound plan.
                unsafe {
                    plan.run_with_scratch(dst, workspace, executor);
                }
            }
            _ => {
                // SAFETY: destination and executor were validated above; this
                // path allocates the plan's exact workspace itself.
                unsafe {
                    plan.run_allocating(dst, executor);
                }
            }
        }
        true
    }

    /// Whether [`Self::try_mul_with_executor`] can compute a product of these operand widths.
    ///
    /// This is a *capability* predicate, not a crossover: it reports whether the
    /// construction exists for these widths, and says nothing about whether it is
    /// the fastest tier for them. The dispatcher applies `SSA_THRESHOLD` on top.
    ///
    /// Keeping the two separate is what lets the dispatcher name a tier instead of
    /// merely attempting one. Every rejection below mirrors a `false` return in
    /// [`Self::try_mul_with_executor`], and `mul::dispatch::tests` sweeps the two against each
    /// other, so a plan naming this tier is a plan that runs.
    ///
    /// The bound is computed from the declared widths. A leading-zero-heavy operand
    /// only shortens the significant width, and [`SsaPlan::crt_half_width`] is monotone in
    /// it, so an accepted pair stays accepted once the exact widths are known.
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        let Some(product_width) = len_a.checked_add(len_b) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// The squaring counterpart of [`Self::admits_mul`].
    pub fn admits_sqr(len: usize) -> bool {
        let Some(product_width) = len.checked_mul(2) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// Scratch required for one [`Self::try_mul_with_executor`] call on operands of these limb
    /// widths, using the geometry the planner selects.
    ///
    /// Returns zero when the widths cannot describe a representable product, which
    /// is also the case the entry point declines.
    pub fn mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        Self::mul_scratch_len_for_parallelism(len_a, len_b, 1)
    }

    /// Scratch required by [`Self::try_mul_with_executor`] for an executor
    /// advertising `parallelism` scheduling lanes.
    pub fn mul_scratch_len_for_parallelism(
        len_a: usize,
        len_b: usize,
        parallelism: usize,
    ) -> usize {
        let Some(half_width) = SsaPlan::crt_half_width_for_operands(len_a, len_b) else {
            return 0;
        };
        // The top-level ring is always transformed, so it must be sized for the
        // transform layout even when it is narrow enough for the basecase.
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        let ring_plan = FftPlan::new(ring_bits);
        let ring_scratch =
            ring_plan.transform_mul_scratch_for_slots(ring_plan.parallel_slots(parallelism));
        if ring_scratch == usize::MAX {
            return 0;
        }
        SsaCrt::layout_len(half_width, ring_scratch)
    }

    /// Scratch required to square a full-width operand of `len` limbs.
    ///
    /// The significant value can only shorten the selected half-width, and
    /// [`SsaCrt::sqr_layout_len`] is monotone in it, so sizing from the complete
    /// declared width is a valid reusable upper bound.
    ///
    /// Returns zero when the width cannot describe a representable square, which is
    /// also the case the entry point declines.
    pub fn sqr_scratch_len(len: usize) -> usize {
        Self::sqr_scratch_len_for_parallelism(len, 1)
    }

    /// Scratch required by [`Self::try_sqr_with_executor`] for an executor
    /// advertising `parallelism` scheduling lanes.
    pub fn sqr_scratch_len_for_parallelism(len: usize, parallelism: usize) -> usize {
        let Some(required_bits) = len
            .checked_mul(2)
            .and_then(|width| width.checked_mul(LIMB_BITS))
        else {
            return 0;
        };
        let Some(half_width) = SsaPlan::crt_half_width(required_bits) else {
            return 0;
        };
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        // The top-level ring is always transformed, so it must be sized for the
        // transform layout even when it is narrow enough for the basecase.
        let ring_plan = FftPlan::new(ring_bits);
        let ring_scratch =
            ring_plan.transform_sqr_scratch_for_slots(ring_plan.parallel_slots(parallelism));
        if ring_scratch == usize::MAX {
            return 0;
        }
        SsaCrt::sqr_layout_len(half_width, ring_scratch)
    }

    /// Square a limb slice with recursive Fermat-ring FFT squaring.
    ///
    /// The CRT split is what makes a square cheaper than a general product. A single
    /// Fermat ring wide enough to hold the exact square would be simpler, but it
    /// discards that discount entirely, and sizing such a ring means rounding the
    /// product width up to the next power of two — which on any width that is not
    /// already one inflates the ring by the rounding ratio, worth 1.67x at five
    /// million limbs. Reaching [`SsaPlan::crt_half_width`] avoids both.
    ///
    /// Takes a [`TransformChoice`] and optional caller scratch, using a
    /// caller-selected synchronous executor. The transform and CRT geometry
    /// are identical for every executor. [`TransformChoice::exponent`] has no
    /// effect here because squaring plans its own ring geometry.
    pub fn try_sqr_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        choice: TransformChoice,
        scratch: Option<&mut [Limb]>,
        executor: &E,
    ) -> bool {
        debug_assert!(
            choice.exponent.is_none(),
            "squaring has no exponent override; see this function's documentation"
        );
        if a_limbs.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(plan) = SsaSquaringPlan::try_new(a_limbs, choice, executor.parallelism().get())
        else {
            return false;
        };
        if dst.len() < plan.destination_len() {
            return false;
        }
        // SAFETY: the boundary above checks the destination, and the branch
        // below either validates caller scratch or allocates the exact plan
        // width before entering the infallible prepared executor.
        unsafe {
            match scratch {
                Some(workspace) if workspace.len() >= plan.scratch_len() => {
                    plan.run_with_scratch(dst, workspace, executor);
                }
                _ => {
                    let mut owned = vec![0; plan.scratch_len()];
                    plan.run_with_scratch(dst, &mut owned, executor);
                }
            }
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
