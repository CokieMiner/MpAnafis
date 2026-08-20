//! FFT parameter orchestration logic for recursive Fermat-ring SSA.
//!
//! The planner picks the radix-2 transform length for a Fermat ring
//! $\mathbb{Z}/(2^{\text{modulus\_bits}} + 1)$ by minimising a closed-form cost
//! model instead of trusting a single analytic formula. The model is recursive:
//! the pointwise stage of a transform is priced by planning the inner ring it
//! would produce, so a geometry that hides an expensive nested FFT behind a
//! cheap-looking butterfly count is rejected on its true cost.

#[cfg(target_has_atomic = "8")]
use core::sync::atomic::{AtomicU8, Ordering};

use super::{InverseTwist, LIMB_BITS, SSA_BASE_MODULUS_BITS, SsaPlan, SsaPointwise, SsaRing};

// `SSA_COEFFICIENT_VISIT_OVERHEAD` prices the fixed setup in every
// coefficient visit relative to one limb multiplication. The model would
// otherwise over-reward long transforms with artificially narrow coefficients.

/// Depth at which the cost model stops expanding nested rings and prices the
/// remaining product with the basecase estimate.
///
/// **Portable.** A pure termination bound: [`SsaPlan::price_geometry`] rejects any
/// candidate whose inner ring is not strictly narrower, so every level at least
/// halves and no representable operand reaches this depth.
pub const MAX_COST_RECURSION_DEPTH: u32 = 6;

/// Cached geometry code for power-of-two ring widths, indexed by `log2(bits)`.
///
/// The cost model is deterministic for a build, so a benign race can only
/// store the same exponent and alignment choice. Zero means uninitialized.
#[cfg(target_has_atomic = "8")]
static POWER_OF_TWO_GEOMETRIES: [AtomicU8; LIMB_BITS] = [const { AtomicU8::new(0) }; LIMB_BITS];

/// A parameter set for the recursive Fermat-ring FFT algorithm.
///
/// Plain arithmetic data, so it is `Copy`: the transform takes its geometry by
/// value rather than borrowing it, which keeps a caller-forced plan and a
/// planner-derived one interchangeable at the call site.
#[derive(Clone, Copy, Debug)]
pub struct FftPlan {
    pub modulus_bits: usize,
    pub transform_len: usize,
    pub transform_log: usize,
    pub chunk_bits: usize,
    pub inner_bits: usize,
    pub inner_cl: usize,
    /// Pre-twist step in *half-bit* units: `2 * inner_bits / transform_len`.
    /// Odd values carry a `sqrt(2)` factor; see `ring::fermat_shift_half`.
    pub twist_step_half: usize,
    pub omega_shift: usize,
    pub mat_limbs: usize,
    pub recon_len: usize,
}

impl FftPlan {
    /// The inverse twiddle and scaling correction this geometry implies.
    #[must_use]
    pub const fn inverse_twist(&self) -> InverseTwist {
        InverseTwist {
            inner_bits: self.inner_bits,
            transform_log: self.transform_log,
            twist_step_half: self.twist_step_half,
        }
    }

    /// Builds the plan the cost model selects for this ring.
    ///
    /// Total by construction: [`Geometry::best_for`] always yields a geometry
    /// for a ring width that is a positive multiple of `LIMB_BITS`, which every
    /// caller in this module guarantees.
    pub fn new(modulus_bits: usize) -> Self {
        Self::from_geometry(modulus_bits, &Geometry::best_for(modulus_bits))
    }

    /// Builds the plan for one caller-forced transform exponent.
    ///
    /// Returns `None` when the exponent cannot produce a usable transform for
    /// this ring: it does not divide the ring width, it leaves no room for the
    /// coefficient bound, or the inner ring has no primitive root of that order.
    pub fn try_forced(modulus_bits: usize, transform_exponent: u32) -> Option<Self> {
        let geometry = forced_geometry(modulus_bits, transform_exponent)?;
        Some(Self::from_geometry(modulus_bits, &geometry))
    }

    /// Scratch for the path an unforced executor-aware FFT multiplication call
    /// would take at this ring width.
    pub fn required_mul_scratch(&self) -> usize {
        if self.modulus_bits <= SSA_BASE_MODULUS_BITS {
            SsaPointwise::fermat_basecase_scratch_len(self.modulus_bits)
        } else {
            self.transform_mul_scratch()
        }
    }

    /// Scratch for the transform path specifically.
    ///
    /// A ring narrow enough for the basecase still runs the transform when the
    /// caller forces it, and the transform layout is not bounded by the
    /// basecase product buffer, so the two must be asked for separately.
    pub fn transform_mul_scratch(&self) -> usize {
        self.transform_mul_scratch_for_slots(2)
    }

    /// Scratch for a product transform with `slots` twiddle slots per operand.
    ///
    /// Two slots are the structural minimum: the radix-4 recursion can only fork
    /// when each child owns one private staging coefficient. Larger executor
    /// policies reserve a balanced contiguous arena for additional child ranges.
    pub fn transform_mul_scratch_for_slots(&self, slots: usize) -> usize {
        let slot_count = slots.max(1);
        let Some(matrix) = self.mat_limbs.checked_mul(2) else {
            return usize::MAX;
        };
        let Some(twiddle) = self
            .inner_cl
            .checked_mul(slot_count)
            .and_then(|n| n.checked_mul(2))
        else {
            return usize::MAX;
        };
        matrix
            .checked_add(twiddle)
            .and_then(|n| n.checked_add(self.pointwise_scratch_for_parallelism(slot_count)))
            .and_then(|n| n.checked_add(self.recon_len))
            .unwrap_or(usize::MAX)
    }

    /// Scratch for one pointwise coefficient leaf. Nested products use the
    /// sequential child executor and therefore need only the baseline arena.
    pub fn pointwise_leaf_scratch(&self) -> usize {
        // `usize::MAX` is the planner's overflow sentinel; saturating addition
        // preserves it while keeping the calculation total for malformed widths.
        self.inner_cl
            .saturating_add(self.nested_transform_scratch())
    }

    /// Scratch for all pointwise leaves at this transform's scheduling width.
    /// Each leaf owns a complete product arena; nested coefficient products run
    /// sequentially so they cannot oversubscribe the outer executor.
    pub fn pointwise_scratch_for_parallelism(&self, parallelism: usize) -> usize {
        let workers = Self::pointwise_parallelism_budget(self.transform_len, parallelism);
        let leaf_len = self.transform_len.div_ceil(workers).max(1);
        let leaves = Self::pointwise_leaf_count(self.transform_len, leaf_len);
        self.pointwise_leaf_scratch().saturating_mul(leaves)
    }

    /// Rounds an executor hint to a power-of-two pointwise split budget.
    pub const fn pointwise_parallelism_budget(transform_len: usize, requested: usize) -> usize {
        let request = if requested == 0 { 1 } else { requested };
        let bounded = if request > transform_len {
            transform_len
        } else {
            request
        };
        if bounded <= 1 {
            return 1;
        }
        let mut budget = 1;
        while budget <= bounded.div_euclid(2) {
            budget = budget.saturating_mul(2);
        }
        budget
    }

    /// Counts the coefficient-aligned leaves produced by a pointwise splitter.
    pub fn pointwise_leaf_count(transform_len: usize, leaf_len: usize) -> usize {
        let leaf_width = leaf_len.max(1);
        if transform_len <= leaf_width {
            1
        } else {
            let left = transform_len.div_euclid(2);
            Self::pointwise_leaf_count(left, leaf_width).saturating_add(Self::pointwise_leaf_count(
                transform_len.saturating_sub(left),
                leaf_width,
            ))
        }
    }

    /// Scratch for the coefficient square stage with the given slot budget.
    pub fn square_scratch_for_slots(&self, slots: usize) -> usize {
        self.pointwise_scratch_for_parallelism(slots)
    }

    /// Returns the per-operand twiddle-slot budget for an executor.
    ///
    /// The budget is bounded by both the executor's scheduling hint and the
    /// transform width, then rounded down to a power of two so recursive halves
    /// remain coefficient-aligned. A two-slot minimum is structural rather than
    /// tuned: it is exactly what one safe fork/join requires for disjoint scratch.
    pub const fn parallel_slots(&self, parallelism: usize) -> usize {
        let budget = Self::pointwise_parallelism_budget(self.transform_len, parallelism);
        if self.transform_len <= 1 {
            return budget;
        }
        // A power-of-two arena keeps every recursive half coefficient-aligned.
        // Round the scheduling hint down to the largest such arena that still
        // exposes at least the structural two-way fork.
        if budget < 2 { 2 } else { budget }
    }

    pub fn required_sqr_scratch(&self) -> usize {
        if self.modulus_bits <= SSA_BASE_MODULUS_BITS {
            SsaPointwise::fermat_basecase_scratch_len(self.modulus_bits)
        } else {
            self.transform_sqr_scratch()
        }
    }

    /// Scratch for the squaring transform path specifically.
    pub fn transform_sqr_scratch(&self) -> usize {
        self.transform_sqr_scratch_for_slots(2)
    }

    /// Scratch for a square transform with `slots` twiddle slots.
    pub fn transform_sqr_scratch_for_slots(&self, slots: usize) -> usize {
        let slot_count = slots.max(1);
        let Some(twiddle) = self.inner_cl.checked_mul(slot_count) else {
            return usize::MAX;
        };
        self.mat_limbs
            .checked_add(twiddle)
            .and_then(|n| n.checked_add(self.square_scratch_for_slots(slot_count)))
            .and_then(|n| n.checked_add(self.recon_len))
            .unwrap_or(usize::MAX)
    }

    /// Computes the nested coefficient scratch needed by the pointwise stage.
    ///
    /// Nested coefficient products use a sequential child executor, so their
    /// scratch stays at the one-worker baseline and cannot oversubscribe the
    /// outer executor. Planning that nested geometry here keeps the check at
    /// the outer scratch boundary.
    fn nested_transform_scratch(&self) -> usize {
        if self.inner_bits <= SSA_BASE_MODULUS_BITS {
            SsaPointwise::fermat_basecase_scratch_len(self.inner_bits)
        } else {
            let nested_plan = Self::new(self.inner_bits);
            // Even a sequential outer executor reserves the two structural
            // twiddle slots used by the transform entry point.
            nested_plan.transform_mul_scratch_for_slots(2)
        }
    }

    /// Expands a validated geometry into the full scratch-aware plan.
    fn from_geometry(modulus_bits: usize, geometry: &Geometry) -> Self {
        let cl = SsaRing::coeff_limbs(modulus_bits);
        let inner_cl = SsaRing::coeff_limbs(geometry.inner_bits);
        let mat_limbs = geometry.transform_len.saturating_mul(inner_cl);
        let recon_len = cl
            .checked_add(inner_cl)
            .and_then(|n| n.checked_mul(2))
            .map_or(usize::MAX, |n| n.max(cl));

        Self {
            modulus_bits,
            transform_len: geometry.transform_len,
            transform_log: geometry.transform_log,
            chunk_bits: geometry.chunk_bits,
            inner_bits: geometry.inner_bits,
            inner_cl,
            twist_step_half: geometry.twist_step_half,
            // omega = theta^2, so the transform root is a whole-bit shift of
            // exactly the half-bit twist step.
            omega_shift: geometry.twist_step_half,
            mat_limbs,
            recon_len,
        }
    }
}

/// The geometric core of a plan: everything derived purely from the transform
/// exponent and the ring width, with no scratch accounting.
pub struct Geometry {
    pub transform_len: usize,
    pub transform_log: usize,
    pub chunk_bits: usize,
    pub inner_bits: usize,
    pub twist_step_half: usize,
    #[cfg(target_has_atomic = "8")]
    pub whole_step: bool,
}

impl Geometry {
    /// The geometry the cost model selects for this ring.
    ///
    /// The windowed search can come up empty for a ring whose width has too few
    /// trailing zeros for the window it lands in, so this falls back to the
    /// two-point transform. That fallback is always available: for any
    /// `modulus_bits` that is a positive multiple of `LIMB_BITS`, exponent 1
    /// divides the ring width, leaves `chunk_bits = modulus_bits / 2 > 0`, and
    /// produces an inner ring at least `LIMB_BITS` wide, which is `>= 2` and has
    /// an even width, so a primitive square root of unity exists.
    fn best_for(modulus_bits: usize) -> Self {
        debug_assert!(
            modulus_bits >= LIMB_BITS && modulus_bits.is_multiple_of(LIMB_BITS),
            "ring widths are always whole, non-empty limb counts"
        );
        if let Some(geometry) = Self::cached(modulus_bits) {
            return geometry;
        }

        let geometry = SsaPlan::best_exponent(modulus_bits, 0)
            .map_or_else(|| Self::two_point(modulus_bits), |(_, geometry)| geometry);
        #[cfg(target_has_atomic = "8")]
        geometry.cache(modulus_bits);
        geometry
    }

    /// Reads the optional planner cache on targets with byte atomics.
    #[cfg(target_has_atomic = "8")]
    fn cached(modulus_bits: usize) -> Option<Self> {
        let slot = if modulus_bits.is_power_of_two() {
            #[allow(
                clippy::as_conversions,
                reason = "a usize trailing-zero count is strictly below LIMB_BITS"
            )]
            POWER_OF_TWO_GEOMETRIES.get(modulus_bits.trailing_zeros() as usize)
        } else {
            None
        }?;
        let encoded = slot.load(Ordering::Relaxed);
        if encoded == 0 {
            return None;
        }
        let payload = encoded.wrapping_sub(1);
        let exponent = u32::from(payload.wrapping_shr(1));
        let whole_step = payload & 1 != 0;
        Self::for_exponent(exponent, modulus_bits, whole_step)
    }

    /// Recomputes plans on architectures that cannot implement a byte cache.
    #[cfg(not(target_has_atomic = "8"))]
    const fn cached(_modulus_bits: usize) -> Option<Self> {
        None
    }

    /// Stores this geometry in the optional power-of-two planner cache.
    #[cfg(target_has_atomic = "8")]
    fn cache(&self, modulus_bits: usize) {
        if modulus_bits.is_power_of_two() {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "a valid transform exponent and its alignment bit fit in u8"
            )]
            let encoded = ((self.transform_log as u8) << 1)
                .wrapping_add(u8::from(self.whole_step))
                .wrapping_add(1);
            #[allow(
                clippy::as_conversions,
                reason = "a usize trailing-zero count is strictly below LIMB_BITS"
            )]
            if let Some(slot) = POWER_OF_TWO_GEOMETRIES.get(modulus_bits.trailing_zeros() as usize)
            {
                slot.store(encoded, Ordering::Relaxed);
            }
        }
    }

    /// Builds the two-point geometry directly, without going through the
    /// fallible [`Self::for_exponent`].
    ///
    /// This is the fallback [`Self::best_for`] needs and the reason it is total.
    /// Exponent 1 satisfies every condition in `for_exponent` for any ring width
    /// that is a positive multiple of `LIMB_BITS`, so rather than assert that
    /// through an unchecked unwrap, the case is constructed outright: the
    /// derivation below is the `exponent == 1` specialization of the general one.
    fn two_point(modulus_bits: usize) -> Self {
        let chunk_bits = modulus_bits.wrapping_shr(1);
        // `alignment` is `max(2, LIMB_BITS) == LIMB_BITS`, and `modulus_bits` is
        // a multiple of `LIMB_BITS`, so `2 * chunk_bits + 1 == modulus_bits + 1`
        // rounds up to exactly one limb past the ring width.
        let inner_bound = modulus_bits.wrapping_add(1);
        let aligned = modulus_bits.wrapping_add(LIMB_BITS);
        let inner_bits = if aligned <= SSA_BASE_MODULUS_BITS {
            aligned
        } else {
            inner_bound.checked_next_power_of_two().unwrap_or(aligned)
        };
        Self {
            transform_len: 2,
            transform_log: 1,
            chunk_bits,
            inner_bits,
            // `2 * inner_bits / transform_len` at `transform_len == 2`.
            twist_step_half: inner_bits,
            #[cfg(target_has_atomic = "8")]
            whole_step: true,
        }
    }

    /// Derives the geometry for one transform exponent, or `None` when the
    /// exponent cannot produce a usable Fermat transform for this ring.
    ///
    /// `whole_step` asks for an inner ring aligned to a whole transform length
    /// rather than a half, which makes the pre-twist a plain shift at the price
    /// of a wider ring. Both are valid; [`SsaPlan::best_exponent`] prices each and
    /// keeps the cheaper, because the `sqrt(2)` factor pays for itself only once
    /// the pointwise stage is large enough to dominate the twist passes.
    pub fn for_exponent(exponent: u32, modulus_bits: usize, whole_step: bool) -> Option<Self> {
        let transform_len = 1_usize.checked_shl(exponent)?;
        if transform_len < 2 || !modulus_bits.is_multiple_of(transform_len) {
            return None;
        }
        // `checked_shl` above rejects every exponent at or above `usize::BITS`,
        // so the trailing-zero count of `transform_len` is exactly `exponent`
        // and is in range for `usize` on every target.
        #[allow(
            clippy::as_conversions,
            reason = "a usize trailing-zero count is at most usize::BITS and always fits"
        )]
        let transform_log = transform_len.trailing_zeros() as usize;

        let chunk_bits = modulus_bits.wrapping_shr(exponent);
        if chunk_bits == 0 {
            return None;
        }

        // Each coefficient must hold the full product of two chunks plus the
        // carries accumulated over `transform_log` butterfly stages.
        let inner_bound = chunk_bits.checked_mul(2)?.checked_add(transform_log)?;

        // The ring width must be a whole number of limbs and, once the
        // transform is wider than two limbs, a whole number of *half* transform
        // steps. Half is enough because the pre-twist is applied in half-bit
        // units: an odd step is a genuine `sqrt(2)` factor rather than an
        // impossible one. Demanding whole steps instead would round the inner
        // ring up to the next multiple of `transform_len`, which at the widest
        // operands is close to a factor of two of wasted pointwise work.
        let step_alignment = if whole_step {
            transform_len
        } else {
            transform_len.wrapping_shr(1)
        };
        let alignment = step_alignment.max(LIMB_BITS);
        let mask = alignment.wrapping_sub(1);
        let aligned = inner_bound.checked_add(mask)? & !mask;

        let inner_bits = if aligned <= SSA_BASE_MODULUS_BITS {
            aligned
        } else {
            // Above the basecase width the inner ring is itself transformed, so
            // it must also admit a nested geometry. Widening it to the next
            // power of two guarantees that and costs up to a factor of two of
            // pointwise work; `nested_ring_bits` buys the same guarantee at the
            // granularity the nested transform actually needs.
            nested_ring_bits(aligned, alignment)?
        };

        // The ring must contain a primitive `transform_len`-th root of unity:
        // 2 is a `2 * inner_bits`-th root, so `transform_len` must divide
        // `2 * inner_bits`. The pre-twist is the square root of that root, and
        // since the ring also contains `sqrt(2)` it exists whenever the twist
        // step is a whole number of *half* bits — no stricter condition.
        let doubled = inner_bits.checked_mul(2)?;
        if inner_bits < transform_len.wrapping_shr(1) || !doubled.is_multiple_of(transform_len) {
            return None;
        }
        // `transform_len >= 2` was established above, so the divisor is
        // non-zero and the quotient is at least one.
        let twist_step_half = doubled.div_euclid(transform_len);

        Some(Self {
            transform_len,
            transform_log,
            chunk_bits,
            inner_bits,
            twist_step_half,
            #[cfg(target_has_atomic = "8")]
            whole_step,
        })
    }
}

/// Rounds a nested inner ring up to a width the nested planner can partition.
///
/// A ring wider than `SSA_BASE_MODULUS_BITS` has its pointwise products
/// computed by another transform, so its width must be divisible by that
/// transform's length. Rounding up to the next power of two satisfies that
/// unconditionally, but it is up to twice as wide as required, and the inner
/// ring width multiplies *every* coefficient visit and *every* pointwise
/// product in the enclosing transform.
///
/// Rounding instead to a multiple of the nested transform length is the same
/// guarantee at a fraction of the cost. The nested length is estimated from
/// [`SsaPlan::search_centre`] rather than from a full nested search, because the
/// search prices geometries by calling back into this function. Rounding can move
/// the estimate, so the adjustment is iterated to a fixed point, bounded because
/// each round strictly increases the width and `checked_add` fails before it can
/// run away.
fn nested_ring_bits(width: usize, alignment: usize) -> Option<usize> {
    let mut width = width;
    loop {
        let nested_log = SsaPlan::search_centre(width).saturating_add(2);
        // A nested length that does not fit `usize` cannot describe a transform
        // on this target, so the alignment it would demand is not a constraint.
        let nested_len = 1_usize.checked_shl(nested_log).unwrap_or(1);
        let unit = alignment.max(nested_len).max(LIMB_BITS);
        let mask = unit.checked_sub(1)?;
        let rounded = width.checked_add(mask)? & !mask;
        if rounded == width {
            return Some(width);
        }
        width = rounded;
    }
}

/// Chooses the cheaper half-step alignment for one forced exponent.
fn forced_geometry(modulus_bits: usize, exponent: u32) -> Option<Geometry> {
    let half = SsaPlan::price_geometry(exponent, modulus_bits, 0, false);
    let whole = SsaPlan::price_geometry(exponent, modulus_bits, 0, true);
    match (half, whole) {
        (Some((half_cost, half_geometry)), Some((whole_cost, whole_geometry))) => {
            if whole_cost < half_cost {
                Some(whole_geometry)
            } else {
                Some(half_geometry)
            }
        }
        (Some((_, geometry)), None) | (None, Some((_, geometry))) => Some(geometry),
        (None, None) => None,
    }
}
