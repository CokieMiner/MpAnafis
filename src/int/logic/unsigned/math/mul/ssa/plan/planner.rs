//! Validated FFT dimensions and overlapping phase workspace requirements.

#![expect(
    unsafe_code,
    reason = "Admitted transform powers and positive coefficient widths bound worker rounding and validated period construction"
)]

use core::{
    num::NonZeroUsize,
    ops::{Deref, Div},
};

use super::{
    Geometry, InverseTwist, LIMB_BITS, PointwiseMulPlan, PointwiseSquarePlan, ReconstructionBlocks,
    SSA_BASE_MODULUS_BITS, SsaOperation, SsaPointwise, SsaRing,
};

/// Depth at which the cost model stops expanding nested rings and prices the
/// remaining product with the basecase estimate.
///
/// [`super::SsaPlan::price_geometry`] follows only strictly shrinking inner
/// rings. This limit bounds recursive pricing work even when a further nested
/// transform remains executable; it does not limit the arithmetic recursion.
pub const MAX_COST_RECURSION_DEPTH: u32 = 6;

/// A parameter set for the recursive Fermat-ring FFT algorithm.
///
/// Geometry is copied into execution plans; all workspace sizes derive from it.
#[derive(Clone, Copy, Debug)]
pub struct FftPlan {
    geometry: FftGeometry,
}

/// Positive whole-bit and half-bit periods of an admitted Fermat ring.
#[derive(Clone, Copy, Debug)]
pub struct RingPeriods {
    pub whole: NonZeroUsize,
    pub half: NonZeroUsize,
}

/// Read-only dimensions exposed by a constructed FFT plan.
#[derive(Clone, Copy, Debug)]
pub struct FftGeometry {
    pub modulus_bits: usize,
    pub transform_len: usize,
    pub chunk_bits: NonZeroUsize,
    pub inner_bits: usize,
    /// `2*inner_bits` and `4*inner_bits`, established at plan construction.
    /// An overflowing, non-executable layout uses positive sentinel periods
    /// and `mat_limbs.get() == usize::MAX`.
    pub periods: RingPeriods,
    pub inner_cl: NonZeroUsize,
    /// Pre-twist step in *half-bit* units: `2 * inner_bits / transform_len`.
    /// Odd values carry a `sqrt(2)` factor; see `SsaRing::shift_sqrt2`.
    /// This is also the whole-bit transform root shift: omega = theta^2.
    pub twist_step_half: usize,
    pub mat_limbs: NonZeroUsize,
    pub recon_len: usize,
}

impl Deref for FftPlan {
    type Target = FftGeometry;

    fn deref(&self) -> &Self::Target {
        &self.geometry
    }
}

impl FftPlan {
    /// The inverse twiddle and scaling correction this geometry implies.
    #[must_use]
    pub const fn inverse_twist(&self) -> InverseTwist {
        #[expect(
            clippy::as_conversions,
            reason = "the trailing-zero count is at most 64 and fits both SSA pointer widths"
        )]
        let transform_log = self.geometry.transform_len.trailing_zeros() as usize;
        InverseTwist {
            inner_bits: self.geometry.inner_bits,
            transform_log,
            twist_step_half: self.geometry.twist_step_half,
        }
    }

    /// Builds the plan the cost model selects for this ring and operation.
    ///
    /// Total by construction: [`Geometry::best_for_operation`] always yields a
    /// geometry for a ring width that is a positive multiple of `LIMB_BITS`,
    /// which every caller in this module guarantees. Multiplication, squaring,
    /// and shared products minimize different objectives (`2F+I+KP`,
    /// `F+I+KS`, `3F+2I+2KP`), so each operation constructs its own geometry.
    pub fn new(modulus_bits: usize) -> Self {
        Self::from_geometry(
            modulus_bits,
            &Geometry::best_for_operation(modulus_bits, SsaOperation::Multiply),
        )
    }

    /// Builds the plan minimizing the squaring objective `F+I+KS`.
    pub fn new_for_square(modulus_bits: usize) -> Self {
        Self::from_geometry(
            modulus_bits,
            &Geometry::best_for_operation(modulus_bits, SsaOperation::Square),
        )
    }

    /// Builds the plan minimizing the shared-product objective `3F+2I+2KP`.
    pub fn new_for_pair(modulus_bits: usize) -> Self {
        Self::from_geometry(
            modulus_bits,
            &Geometry::best_for_operation(modulus_bits, SsaOperation::Pair),
        )
    }

    /// Scratch for the path an unforced executor-aware FFT multiplication call
    /// would take at this ring width.
    pub fn required_mul_scratch(&self) -> usize {
        if self.modulus_bits <= SSA_BASE_MODULUS_BITS {
            SsaPointwise::fermat_basecase_scratch_len(self.modulus_bits)
        } else {
            self.transform_mul_scratch(1)
        }
    }

    /// Scratch for a product transform at `parallelism` workers.
    ///
    /// Twiddle slots and pointwise workers derive separately from the same
    /// budget: slots keep the structural two-slot minimum for staging and odd
    /// twists, while pointwise leaves use the execution-worker count. A
    /// sequential transform therefore reserves one pointwise arena, not two.
    /// Two slots are the structural minimum: the radix-4 recursion can only fork
    /// when each child owns one private staging coefficient. Larger executor
    /// policies reserve a balanced contiguous arena for additional child ranges.
    pub fn transform_mul_scratch(&self, parallelism: usize) -> usize {
        self.transform_mul_scratch_with_pointwise(
            parallelism,
            PointwiseMulPlan::from(self.inner_bits).scratch_len,
        )
    }

    /// Sizes a retained multiplication tree from its precomputed leaf arena.
    pub fn transform_mul_scratch_with_pointwise(
        &self,
        parallelism: usize,
        leaf_scratch: NonZeroUsize,
    ) -> usize {
        let slot_count = self.parallel_slots(parallelism);
        let Some(matrix) = self.mat_limbs.get().checked_mul(2) else {
            return usize::MAX;
        };
        let Some(twiddle) = self
            .inner_cl
            .get()
            .checked_mul(slot_count)
            .and_then(|n| n.checked_mul(2))
        else {
            return usize::MAX;
        };
        let forward = matrix.saturating_add(twiddle);
        let pointwise =
            matrix.saturating_add(self.pointwise_scratch_with_leaf(parallelism, leaf_scratch));
        let reconstruct = self
            .mat_limbs
            .get()
            .saturating_add(twiddle.div_euclid(2))
            .saturating_add(self.reconstruction_scratch(parallelism));
        forward.max(pointwise).max(reconstruct)
    }

    /// Scratch for a two-by-one product transform at `parallelism` workers.
    ///
    /// Sized for three coefficient matrices plus twiddle, pointwise, and
    /// reconstruction workspaces. Twiddle slots and pointwise workers derive
    /// separately from the same budget, as in [`Self::transform_mul_scratch`].
    pub fn transform_mul_two_by_one_scratch(&self, parallelism: usize) -> usize {
        let slot_count = self.parallel_slots(parallelism);
        let Some(matrix) = self.mat_limbs.get().checked_mul(3) else {
            return usize::MAX;
        };
        let Some(twiddle) = self
            .inner_cl
            .get()
            .checked_mul(slot_count)
            .and_then(|n| n.checked_mul(2))
        else {
            return usize::MAX;
        };
        let forward = matrix.saturating_add(twiddle);
        let pointwise = matrix.saturating_add(self.pointwise_scratch_for_parallelism(parallelism));
        // A sequential pair reuses one inverse-twiddle arena and one
        // reconstruction workspace across its two output chains. A parallel
        // executor runs both chains concurrently, so each owns a private arena
        // and workspace that are live simultaneously.
        let reconstruct = if parallelism > 1 {
            self.mat_limbs
                .get()
                .saturating_mul(2)
                .saturating_add(twiddle)
                .saturating_add(self.reconstruction_scratch(parallelism).saturating_mul(2))
        } else {
            self.mat_limbs
                .get()
                .saturating_mul(2)
                .saturating_add(twiddle.div_euclid(2))
                .saturating_add(self.reconstruction_scratch(parallelism))
        };
        forward.max(pointwise).max(reconstruct)
    }

    /// Scratch for all pointwise leaves at this transform's scheduling width.
    /// Each leaf owns a complete product arena; nested coefficient products run
    /// sequentially so they cannot oversubscribe the outer executor.
    pub fn pointwise_scratch_for_parallelism(&self, parallelism: usize) -> usize {
        self.pointwise_scratch_with_leaf(
            parallelism,
            PointwiseMulPlan::from(self.inner_bits).scratch_len,
        )
    }

    /// Applies the executor leaf count to an already retained coefficient arena.
    pub fn pointwise_scratch_with_leaf(
        &self,
        parallelism: usize,
        leaf_scratch: NonZeroUsize,
    ) -> usize {
        let leaves = self.pointwise_leaf_count(parallelism);
        // `usize::MAX` is the planner's overflow sentinel. Each leaf owns one
        // complete sequential arena for its retained coefficient strategy.
        leaf_scratch.get().saturating_mul(leaves.get())
    }

    /// Bounds an executor hint by the number of independent coefficients.
    pub const fn pointwise_parallelism_budget(transform_len: usize, requested: usize) -> usize {
        let request = if requested == 0 { 1 } else { requested };
        let bounded = if request > transform_len {
            transform_len
        } else {
            request
        };
        if bounded == 0 { 1 } else { bounded }
    }

    /// Admits the power-of-two leaf budget for this complete transform.
    ///
    /// Execution carries this budget into every shortened spectrum. An active
    /// prefix may reduce it to its own largest power of two, but cannot reserve
    /// more coefficient workspaces than this full-plan admission.
    pub fn pointwise_leaf_count(&self, requested: usize) -> NonZeroUsize {
        if requested <= 1 {
            return NonZeroUsize::MIN;
        }
        let len = self.geometry.transform_len;
        // SAFETY: constructed geometries have power-of-two len>=2, and the
        // serial request returned above. Thus 2<=min(len,requested)<=len.
        let workers = unsafe { NonZeroUsize::new_unchecked(len.min(requested)) };
        let quotient = len.div(workers);
        // SAFETY: workers<=len proves quotient>=1. Adding the remainder flag
        // gives ceil(len/workers)<=len, so the sum fits and remains positive.
        let leaf_len = unsafe {
            NonZeroUsize::new_unchecked(quotient.unchecked_add(usize::from(len % workers != 0)))
        };
        // Write q=ceil(len/workers) and h=floor(log2(q)). Since len is a
        // power of two, 2^h<=q<2^(h+1) gives
        // len/2^(h+1)<ceil(len/q)<=len/2^h. Thus the smallest admitted
        // power-of-two leaf count is len/2^h, without a second division.
        // SAFETY: 1<=leaf_len<=len, so h<=log2(len)<usize::BITS.
        // Shifting the positive power-of-two len by h remains positive.
        unsafe { NonZeroUsize::new_unchecked(len.unchecked_shr(leaf_len.ilog2())) }
    }

    /// Returns the per-operand twiddle-slot budget for an executor.
    ///
    /// The budget is bounded by both the executor hint and transform width, then
    /// rounded up to the next power of two so a non-power-of-two executor is not
    /// artificially under-filled by the binary transform tree. A two-slot
    /// minimum is structural: operand splitting and odd half-bit twists require
    /// a staging coefficient plus one factor coefficient even when execution is
    /// sequential.
    pub const fn parallel_slots(&self, parallelism: usize) -> usize {
        let budget = Self::pointwise_parallelism_budget(self.geometry.transform_len, parallelism);
        if self.geometry.transform_len <= 1 {
            return budget;
        }
        let target = if budget < 2 { 2 } else { budget };
        // SAFETY: 2<=target<=transform_len, a representable usize power of two.
        // Thus bit_length(target-1)<usize::BITS, and its rounded power fits.
        unsafe {
            let log = usize::BITS.unchecked_sub(target.unchecked_sub(1).leading_zeros());
            1_usize.unchecked_shl(log)
        }
    }

    pub fn required_sqr_scratch(&self) -> usize {
        if self.modulus_bits <= SSA_BASE_MODULUS_BITS {
            PointwiseSquarePlan::from(self.modulus_bits)
                .scratch_len
                .get()
        } else {
            self.transform_sqr_scratch(1)
        }
    }

    /// Scratch for a square transform at `parallelism` workers.
    pub fn transform_sqr_scratch(&self, parallelism: usize) -> usize {
        self.transform_sqr_scratch_with_pointwise(
            parallelism,
            PointwiseSquarePlan::from(self.inner_bits).scratch_len,
        )
    }

    /// Sizes a retained square tree from its precomputed leaf arena.
    pub fn transform_sqr_scratch_with_pointwise(
        &self,
        parallelism: usize,
        leaf_scratch: NonZeroUsize,
    ) -> usize {
        let slot_count = self.parallel_slots(parallelism);
        let Some(twiddle) = self.inner_cl.get().checked_mul(slot_count) else {
            return usize::MAX;
        };
        self.mat_limbs.get().saturating_add(
            twiddle
                .saturating_add(self.reconstruction_scratch(parallelism))
                .max(self.pointwise_scratch_with_leaf(parallelism, leaf_scratch)),
        )
    }

    /// Sizes serial reconstruction or disjoint parallel blocks and carry merging.
    pub fn reconstruction_scratch(&self, parallelism: usize) -> usize {
        ReconstructionBlocks::new(
            self.transform_len,
            self.transform_len,
            self.chunk_bits,
            self.inner_bits,
            parallelism,
        )
        .map_or(self.recon_len, |blocks| {
            let outer = SsaRing::coeff_limbs(self.modulus_bits).get();
            outer
                .saturating_add(self.inner_cl.get())
                .saturating_add(1)
                .saturating_add(blocks.scratch_len())
                .max(self.recon_len)
        })
    }

    /// Expands a validated geometry into the full scratch-aware plan.
    fn from_geometry(modulus_bits: usize, geometry: &Geometry) -> Self {
        let cl = SsaRing::coeff_limbs(modulus_bits);
        let inner_cl = SsaRing::coeff_limbs(geometry.inner_bits);
        // Half-bit twists have period 4n. Overflow rejects the executable
        // layout through the same sentinel as an unrepresentable matrix.
        let (periods, mat_limbs) = geometry.inner_bits.checked_mul(4).map_or(
            (
                RingPeriods {
                    whole: NonZeroUsize::MIN,
                    half: NonZeroUsize::MIN,
                },
                NonZeroUsize::MAX,
            ),
            |half| {
                // SAFETY: geometry supplies inner_bits>=LIMB_BITS>0. The checked
                // half-bit period is therefore >=4*LIMB_BITS, and its half is
                // positive on both admitted SSA pointer widths.
                let periods = unsafe {
                    RingPeriods {
                        whole: NonZeroUsize::new_unchecked(half >> 1),
                        half: NonZeroUsize::new_unchecked(half),
                    }
                };
                // SAFETY: Geometry admits K>=2 and coeff_limbs carries its
                // positive guard width. Their product and its overflow sentinel
                // usize::MAX are positive, so the matrix retains that invariant.
                let mat_limbs = unsafe {
                    NonZeroUsize::new_unchecked(
                        geometry.transform_len.saturating_mul(inner_cl.get()),
                    )
                };
                (periods, mat_limbs)
            },
        );
        // Reconstruction streams magnitude digits into the accumulator and
        // folds its disjoint high half directly. Only the accumulator is live
        // in serial execution: one complete coefficient at the last chunk
        // offset, plus the sub-limb shift's top limb and the outer bias guard.
        let recon_len = geometry
            .transform_len
            .checked_sub(1)
            .and_then(|last| last.checked_mul(geometry.chunk_bits.get()))
            .and_then(|bits| bits.checked_div(LIMB_BITS))
            .and_then(|offset| offset.checked_add(inner_cl.get()))
            .and_then(|span| span.checked_add(1))
            .map_or(usize::MAX, |span| span.max(cl.get()));

        Self {
            geometry: FftGeometry {
                modulus_bits,
                transform_len: geometry.transform_len,
                chunk_bits: geometry.chunk_bits,
                inner_bits: geometry.inner_bits,
                periods,
                inner_cl,
                twist_step_half: geometry.twist_step_half,
                mat_limbs,
                recon_len,
            },
        }
    }
}
