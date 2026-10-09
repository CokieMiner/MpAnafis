//! Recursive cost model used to select an SSA transform geometry.

#![expect(
    unsafe_code,
    reason = "Validated capacity bits bound significant-width arithmetic; finite exponent searches bound recursion and candidate slots"
)]

use core::{mem::size_of, num::NonZeroUsize};

use super::{
    CostMemo, Geometry, LIMB_BITS, Limb, MAX_COST_RECURSION_DEPTH, Multiplication, NegacyclicPlan,
    SSA_BASE_MODULUS_BITS, SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS,
    SSA_COEFFICIENT_VISIT_OVERHEAD, SSA_NESTED_COST_PENALTY_16THS, SsaRing,
};

/// Exact arithmetic pass overhead for sqrt(2) shifts across forward and inverse transforms.
const SSA_SQRT2_TWIST_PASSES: usize = 4;

/// Operation whose transform geometry the cost model prices.
///
/// Multiplication, squaring, and shared products traverse different numbers
/// of forward transforms, inverse transforms, and pointwise products, so the
/// cheapest geometry differs even at a fixed ring width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SsaOperation {
    /// Two forward transforms, one inverse, one pointwise product per frequency.
    Multiply,
    /// One forward transform, one inverse, one pointwise square per frequency.
    Square,
    /// Three forward transforms, two inverses, two pointwise products.
    Pair,
}

/// Global planning and sizing routines for SSA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaPlan;

impl SsaOperation {
    /// Operations that require distinct geometry cache slots.
    #[cfg(feature = "std")]
    pub const ALL: [Self; 3] = [Self::Multiply, Self::Square, Self::Pair];
}

impl SsaPlan {
    /// Compares a complete direct ring with the legal CRT candidate. Both
    /// costs retain operation-specific forwards, inverses, and point products.
    pub fn direct_fermat_is_cheaper(width: usize, operation: SsaOperation) -> bool {
        let Some(bits) = width
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.checked_mul(2))
        else {
            return false;
        };
        let mut memo = CostMemo::new();
        // A direct exact product transforms even below the ring/basecase gate.
        let direct =
            Self::best_exponent(bits, 0, operation, &mut memo).map_or(usize::MAX, |(cost, _)| cost);
        direct < Self::crt_work_for_width(width, operation, &mut memo)
    }

    /// Structural cost of an exact square, including its square CRT recurrence.
    pub fn square_work(len: usize) -> Option<usize> {
        let required = len.checked_mul(LIMB_BITS)?.checked_mul(2)?;
        let width = Self::best_crt_half_width(required, SsaOperation::Square)?;
        Some(
            Self::crt_work_for_width(width, SsaOperation::Square, &mut CostMemo::new())
                .saturating_add(len),
        )
    }

    /// Prices the Fermat children, terminal Mersenne leaf, and linear
    /// folding and reconstruction visits for a legal CRT half-width.
    pub fn crt_work_for_width(width: usize, operation: SsaOperation, memo: &mut CostMemo) -> usize {
        let (staging, merge): (usize, usize) = match operation {
            SsaOperation::Multiply => (2, 2),
            SsaOperation::Square => (1, 2),
            SsaOperation::Pair => (3, 4),
        };
        let Some(bits) = width.checked_mul(LIMB_BITS) else {
            return usize::MAX;
        };
        let mut cost = Self::ring_cost_for_operation(bits, 0, operation, memo)
            .saturating_add(width.saturating_mul(staging.saturating_add(merge)));
        let mut remaining = width;
        while remaining > SSA_BNM1_BASECASE_LIMBS {
            let parent = remaining;
            remaining >>= 1;
            // The top width-to-bits product proves the halved product fits.
            let child_bits = remaining.saturating_mul(LIMB_BITS);
            cost = cost
                .saturating_add(Self::ring_cost_for_operation(
                    child_bits, 0, operation, memo,
                ))
                .saturating_add(
                    parent.saturating_mul(staging.saturating_mul(2).saturating_add(merge)),
                );
        }
        let leaf = match operation {
            SsaOperation::Multiply => Multiplication::structural_product_work(remaining, remaining),
            SsaOperation::Square => Multiplication::structural_square_work(remaining),
            SsaOperation::Pair => {
                Multiplication::structural_product_work(remaining, remaining).saturating_mul(2)
            }
        };
        cost.saturating_add(leaf)
    }

    /// Computes the number of significant bits in a limb slice.
    ///
    /// # Safety
    /// `limbs.len() * LIMB_BITS` must be representable. Operand-bound plans
    /// validate this capacity conversion before inspecting significant widths.
    pub unsafe fn significant_bits_of_slice(limbs: &[Limb]) -> usize {
        limbs
            .iter()
            .enumerate()
            .rfind(|(_, limb)| **limb != 0)
            .map_or(0, |(last_nonzero_idx, &top_limb)| {
                #[expect(
                    clippy::as_conversions,
                    reason = "LIMB_BITS is at most 64; leading_zeros fits in usize across all supported pointer widths"
                )]
                let leading_zeros = top_limb.leading_zeros() as usize;
                // SAFETY: the selected limb is nonzero, so leading_zeros is
                // below LIMB_BITS. The caller admits limbs.len()*LIMB_BITS;
                // index*LIMB_BITS+top_bits is at most that capacity bound.
                unsafe {
                    let top_bits = LIMB_BITS.unchecked_sub(leading_zeros);
                    last_nonzero_idx.unchecked_mul(LIMB_BITS).unchecked_add(top_bits)
                }
            })
    }

    /// Returns the structural work of one complete exact SSA product.
    ///
    /// The top Fermat product is followed by the `B^n - 1` recursion, whose
    /// successive `B^(n/2) + 1` products use the same geometry planner. Linear
    /// terms account for operand staging, folds, and CRT reconstruction. This
    /// estimate retains the actual irregular CRT widths and nested geometries.
    pub fn product_work(len_a: usize, len_b: usize) -> Option<usize> {
        let half_width =
            Self::best_crt_half_width_for_operands(len_a, len_b, SsaOperation::Multiply)?;
        Some(
            Self::crt_work_for_width(half_width, SsaOperation::Multiply, &mut CostMemo::new())
                .saturating_add(len_a)
                .saturating_add(len_b),
        )
    }

    /// Returns the cheapest geometry for a ring together with its modelled cost,
    /// scanning every representable transform exponent for the width.
    /// `operation` selects the forward/inverse/pointwise weights (`2F+I+KP`
    /// for products, `F+I+KS` for squares, `3F+2I+2KP` for pairs), so each
    /// operation minimizes its own transform and pointwise work.
    pub fn best_exponent(
        modulus_bits: usize,
        depth: u32,
        operation: SsaOperation,
        memo: &mut CostMemo,
    ) -> Option<(usize, Geometry)> {
        let exponent_ceiling = modulus_bits.trailing_zeros();
        let low = 1_u32;
        let high = exponent_ceiling.saturating_sub(1);

        let mut winner: Option<(usize, Geometry)> = None;
        let mut probe = low;
        while probe <= high {
            // Half-step and whole-step alignment can produce the same geometry,
            // as can neighboring nested-alignment bumps; pricing either twice
            // only doubles the recursive search graph.
            let candidates = [
                Geometry::for_exponent_candidates(probe, modulus_bits, false),
                Geometry::for_exponent_candidates(probe, modulus_bits, true),
            ];
            for (index, slot) in candidates.iter().flatten().enumerate() {
                let Some(candidate) = slot else {
                    continue;
                };
                // The exponent fixes every dimension except inner_bits.
                // Compare all preceding slots: the whole-step set can repeat
                // several half-step widths in a different position.
                if candidates
                    .iter()
                    .flatten()
                    .take(index)
                    .flatten()
                    .any(|previous| previous.inner_bits == candidate.inner_bits)
                {
                    continue;
                }
                if let Some((cost, _)) =
                    Self::price_geometry(candidate, modulus_bits, depth, operation, memo)
                    && winner.as_ref().is_none_or(|(best, _)| cost < *best)
                {
                    winner = Some((cost, *candidate));
                }
            }
            // SAFETY: probe<=high<=usize::BITS-1<=63 (15 on 16-bit targets),
            // so the next search exponent fits u32.
            probe = unsafe { probe.unchecked_add(1) };
        }
        winner
    }

    /// Centre of the exponent search at every recursion level.
    ///
    /// The classical SSA split cuts `N` bits into approximately `sqrt(N)`
    /// coefficients of approximately `sqrt(N)` bits, so the transform exponent
    /// is centred at `floor(log2(N) / 2)`. Nested-ring alignment and geometry
    /// pricing use this same centre.
    pub fn search_centre(modulus_bits: usize) -> u32 {
        if modulus_bits == 0 {
            return 1;
        }
        modulus_bits.ilog2().div_euclid(2).max(1)
    }

    /// Prices one constructed geometry, including its recursive pointwise products.
    ///
    /// `operation` weights the transform and pointwise terms: products pay two
    /// forwards, one inverse, and one product per frequency; squares pay one
    /// forward, one inverse, and one square; pairs pay three forwards, two
    /// inverses, and two products. The pointwise term names its executed
    /// strategy, including the factor-3 and factor-5 decomposition costs.
    pub fn price_geometry(
        geometry: &Geometry,
        modulus_bits: usize,
        depth: u32,
        operation: SsaOperation,
        memo: &mut CostMemo,
    ) -> Option<(usize, Geometry)> {
        let geometry = *geometry;
        // Let `K = 2^exponent`, `M = modulus_bits / K`, and
        // `required = 2M + log2(K)`. The inner ring width is the aligned value
        // `n >= required`. If `required / n < 1/2`, the `K / 2` transform fits
        // in the same ring: its requirement is `4M + log2(K) - 1`, which is
        // strictly below `n`. It performs half as many pointwise products at
        // the same coefficient width, so this geometry is dominated.
        let required_inner_bits = geometry
            .chunk_bits
            .get()
            .checked_mul(2)?
            .checked_add(geometry.transform_log)?;
        if required_inner_bits.checked_mul(2)? < geometry.inner_bits {
            return None;
        }
        let nests = geometry.inner_bits > SSA_BASE_MODULUS_BITS;
        if nests && geometry.inner_bits >= modulus_bits {
            return None;
        }
        let inner_cl = SsaRing::coeff_limbs(geometry.inner_bits).get();

        // Count each executed traversal independently. A forward includes its
        // split/twist pass; an inverse includes its untwist and reconstruction.
        // Odd half-steps charge each twisted operand and output separately.
        let sqrt2_passes = if geometry.twist_step_half.is_multiple_of(2) {
            0
        } else {
            SSA_SQRT2_TWIST_PASSES
        };
        let visit_cost = inner_cl.checked_add(SSA_COEFFICIENT_VISIT_OVERHEAD)?;
        let (forwards, inverses): (usize, usize) = match operation {
            SsaOperation::Multiply => (2, 1),
            SsaOperation::Square => (1, 1),
            SsaOperation::Pair => (3, 2),
        };
        let forward = geometry
            .transform_log
            .checked_add(1)?
            .checked_add(sqrt2_passes)?;
        let inverse = geometry
            .transform_log
            .checked_add(2)?
            .checked_add(sqrt2_passes)?;
        let passes = forwards
            .checked_mul(forward)?
            .checked_add(inverses.checked_mul(inverse)?)?;
        let transform_cost = geometry
            .transform_len
            .checked_mul(passes)?
            .checked_mul(visit_cost)?;
        let pointwise_factor: usize = match operation {
            SsaOperation::Pair => 2,
            SsaOperation::Multiply | SsaOperation::Square => 1,
        };

        // The nested penalty weights overhead beyond the modeled arithmetic.
        // Squares use their own recurrence; pairs pay two products per frequency.
        let unit_product = {
            let modelled = Self::ring_cost_for_operation(
                geometry.inner_bits,
                // SAFETY: planning begins at depth zero; ring_cost_for_operation
                // stops recursion at MAX_COST_RECURSION_DEPTH before pricing
                // another geometry, so this next depth is at most that bound.
                unsafe { depth.unchecked_add(1) },
                match operation {
                    SsaOperation::Square => SsaOperation::Square,
                    SsaOperation::Multiply | SsaOperation::Pair => SsaOperation::Multiply,
                },
                memo,
            );
            if nests {
                modelled
                    .saturating_mul(SSA_NESTED_COST_PENALTY_16THS)
                    .div_euclid(16)
            } else {
                modelled
            }
        };
        let pointwise_cost = geometry
            .transform_len
            .checked_mul(unit_product)?
            .checked_mul(pointwise_factor)?;
        Some((transform_cost.checked_add(pointwise_cost)?, geometry))
    }

    /// Interpolate between `n^1.5` and `n^1.75` lower-tower cost models.
    pub const fn basecase_product_cost(coefficient_limbs: usize) -> usize {
        let three_halves = coefficient_limbs.saturating_mul(coefficient_limbs.isqrt());
        let seven_fourths = coefficient_limbs.saturating_mul(three_halves.isqrt());
        let interpolation = seven_fourths
            .saturating_sub(three_halves)
            .saturating_mul(SSA_BASECASE_COST_WEIGHT_16THS)
            .div_euclid(16);
        three_halves.saturating_add(interpolation)
    }

    /// Smallest CRT half-width, in limbs, that can carry a `required_bits`-wide
    /// product through the `B^n - 1` / `B^n + 1` decomposition.
    ///
    /// Two constraints bound the choice:
    ///
    /// * `2 * n * LIMB_BITS >= required_bits`, so the product is recovered exactly.
    ///   With `a < 2^sig_a` and `b < 2^sig_b`, the product is at most
    ///   `2^(sig_a + sig_b) - 2^sig_a - 2^sig_b + 1`, strictly below `B^(2n) - 1`,
    ///   which is the modulus the two CRT halves reconstruct against.
    /// * `mul_mod_bnm1` halves its width at every level until it reaches
    ///   `SSA_BNM1_BASECASE_LIMBS`, so every level must stay even. Rounding `n` up
    ///   to a multiple of a power of two makes its odd part no larger than the
    ///   quotient, and requiring that quotient to fit the basecase guarantees the
    ///   recursion never lands on an odd width above it.
    ///
    /// Legality and pricing stay separate: this is the minimum legal width.
    /// [`Self::best_crt_half_width`] compares neighboring legal widths with the
    /// complete CRT recurrence, since a slightly larger width can enable a
    /// better transform length, remove a square-root twist, or avoid an
    /// unfavorable nested alignment whose savings outweigh the extra limbs.
    pub fn crt_half_width(required_bits: usize) -> Option<usize> {
        let minimum = required_bits.div_ceil(LIMB_BITS.checked_mul(2)?).max(2);
        // ceil(minimum / 2^k) <= T iff minimum <= T*2^k. Thus the
        // smallest legal alignment is the next power of two at or above
        // ceil(minimum / T); no candidate array or pricing search is needed.
        let divisor = NonZeroUsize::new(SSA_BNM1_BASECASE_LIMBS)?;
        let step = minimum
            .div_ceil(divisor.get())
            .checked_next_power_of_two()?;
        minimum.div_ceil(step).checked_mul(step)
    }

    /// Cheapest legal CRT half-width for `required_bits` under `operation`.
    ///
    /// Prices each candidate from [`Self::crt_candidates`] with the complete
    /// recurrence for its operation (Fermat ring cost plus Mersenne levels and
    /// linear staging), keeping the minimum. The minimum legal width minimizes
    /// padding, not execution cost.
    pub fn best_crt_half_width(required_bits: usize, operation: SsaOperation) -> Option<usize> {
        if let Some(width) = Self::cached_crt_half_width(required_bits, operation) {
            return Some(width);
        }
        let candidates = Self::crt_candidates(required_bits);
        let mut memo = CostMemo::new();
        let mut winner: Option<(usize, usize)> = None;
        for candidate in candidates {
            let cost = Self::crt_work_for_width(candidate, operation, &mut memo);
            if cost == usize::MAX {
                continue;
            }
            if winner.is_none_or(|(best, _)| cost < best) {
                winner = Some((cost, candidate));
            }
        }
        let width = winner.map(|(_, width)| width);
        // The search is deterministic in its inputs, so caching the winning
        // width can never change a later result for the same query.
        if let Some(cached) = width {
            Self::cache_crt_half_width(required_bits, operation, cached);
        }
        width
    }

    /// Cheapest legal CRT half-width covering these operand widths.
    pub fn best_crt_half_width_for_operands(
        len_a: usize,
        len_b: usize,
        operation: SsaOperation,
    ) -> Option<usize> {
        let product_bits = len_a.checked_add(len_b)?.checked_mul(LIMB_BITS)?;
        Self::best_crt_half_width(product_bits, operation)
    }

    /// All legal CRT half-widths near the minimum for `required_bits`.
    ///
    /// Each candidate satisfies both legality constraints above. The set covers
    /// the minimum, nearby alignment boundaries from the power-of-two step
    /// search, and the next power-of-two width, which can admit a shorter
    /// transform or a whole-bit twist.
    pub fn crt_candidates(required_bits: usize) -> impl Iterator<Item = usize> {
        // At most one candidate per representable power-of-two alignment.
        // A fixed array removes a heap allocation from every sizing query.
        #[expect(
            clippy::manual_bits,
            reason = "const array length avoids a primitive cast and covers 16-, 32-, and 64-bit usize"
        )]
        let mut candidates = [0; size_of::<usize>() * 8];
        let mut count = 0_usize;
        let half_limb_bits = LIMB_BITS * 2;
        let minimum = required_bits.div_ceil(half_limb_bits).max(2);
        if minimum <= SSA_BNM1_BASECASE_LIMBS {
            // SAFETY: every supported pointer width provides a nonempty array.
            unsafe {
                *candidates.get_unchecked_mut(0) = minimum;
            }
            return candidates.into_iter().take(1);
        }
        let mut previous = 0;
        let mut step_log = 0_u32;
        while let Some(step) = 1_usize.checked_shl(step_log) {
            let blocks = minimum.div_ceil(step);
            if blocks <= SSA_BNM1_BASECASE_LIMBS
                && let Some(candidate) = blocks.checked_mul(step)
                && candidate != previous
            {
                // Nested power-of-two alignments yield nondecreasing widths.
                // SAFETY: each iteration inserts at most once, and there are
                // at most usize::BITS representable shifts, matching the array.
                unsafe {
                    *candidates.get_unchecked_mut(count) = candidate;
                }
                // SAFETY: at most usize::BITS<=64 candidates are inserted,
                // which is representable on every supported pointer width.
                count = unsafe { count.unchecked_add(1) };
                previous = candidate;
            }
            // Include the first alignment >= minimum: it is the next useful
            // power-of-two candidate and subsumes every smaller alignment.
            if step >= minimum {
                break;
            }
            // SAFETY: checked_shl admitted step_log<usize::BITS<=64, so its
            // successor fits u32; checked_shl retains the real width boundary.
            step_log = unsafe { step_log.unchecked_add(1) };
        }
        candidates.into_iter().take(count)
    }
    /// Cost of one Fermat-ring product under `operation`.
    ///
    /// Squares use the square recurrence; pairs minimize the pair geometry itself
    /// rather than discounting a single-product optimum.
    pub fn ring_cost_for_operation(
        modulus_bits: usize,
        depth: u32,
        operation: SsaOperation,
        memo: &mut CostMemo,
    ) -> usize {
        if modulus_bits <= SSA_BASE_MODULUS_BITS {
            let limbs = SsaRing::mod_limbs(modulus_bits);
            return match operation {
                SsaOperation::Multiply => NegacyclicPlan::product_cost(limbs),
                SsaOperation::Square => Multiplication::structural_square_work(limbs),
                SsaOperation::Pair => NegacyclicPlan::product_cost(limbs).saturating_mul(2),
            };
        }
        if depth >= MAX_COST_RECURSION_DEPTH {
            let base = Self::basecase_product_cost(SsaRing::mod_limbs(modulus_bits));
            return match operation {
                SsaOperation::Pair => base.saturating_mul(2),
                SsaOperation::Multiply | SsaOperation::Square => base,
            };
        }
        if let Some(cost) = memo.get(modulus_bits, depth, operation) {
            return cost;
        }
        let cost = Self::best_exponent(modulus_bits, depth, operation, memo).map_or_else(
            || {
                let base = Self::basecase_product_cost(SsaRing::coeff_limbs(modulus_bits).get());
                match operation {
                    SsaOperation::Pair => base.saturating_mul(2),
                    SsaOperation::Multiply | SsaOperation::Square => base,
                }
            },
            |(cost, _)| cost,
        );
        memo.insert(modulus_bits, depth, operation, cost);
        cost
    }
}
