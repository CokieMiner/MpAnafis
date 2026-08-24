//! Recursive cost model used to select an SSA transform geometry.

use super::{
    Geometry, LIMB_BITS, Limb, MAX_COST_RECURSION_DEPTH, SSA_BASE_MODULUS_BITS,
    SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD,
    SSA_NESTED_COST_PENALTY_16THS, SsaRing,
};

/// Exact arithmetic pass overhead for sqrt(2) shifts across forward and inverse transforms.
const SSA_SQRT2_TWIST_PASSES: usize = 4;

/// Global planning and sizing routines for SSA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaPlan;

impl SsaPlan {
    /// Computes the number of significant bits in a limb slice.
    pub fn significant_bits_of_slice(limbs: &[Limb]) -> usize {
        limbs
            .iter()
            .rposition(|l| *l != 0)
            .map_or(0, |last_nonzero_idx| {
                // SAFETY: last_nonzero_idx < limbs.len().
                let top_limb = unsafe { *limbs.get_unchecked(last_nonzero_idx) };
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "LIMB_BITS is at most 64; leading_zeros fits in u32"
                )]
                let top_bits = LIMB_BITS.wrapping_sub(top_limb.leading_zeros() as usize);
                last_nonzero_idx
                    .wrapping_mul(LIMB_BITS)
                    .wrapping_add(top_bits)
            })
    }
    /// CRT half-width covering the widest product these operand widths can produce.
    pub fn crt_half_width_for_operands(len_a: usize, len_b: usize) -> Option<usize> {
        let product_bits = len_a.checked_add(len_b)?.checked_mul(LIMB_BITS)?;
        Self::crt_half_width(product_bits)
    }

    /// Returns the cheapest geometry for a ring together with its modelled cost,
    /// scanning every representable transform exponent for the width.
    ///
    /// Geometries whose pointwise products nest another transform are priced
    /// against those that keep them in the multiplication tower, and the cheaper
    /// wins. Neither class is preferred a priori: since
    /// `nested_ring_bits` rounds a nested inner ring to a
    /// multiple of the nested transform length rather than to a power of two,
    /// nesting costs what it actually costs, and preferring the tower
    /// unconditionally selects exponents two to three above the measured optimum at
    /// every RAM-resident width.
    pub fn best_exponent(modulus_bits: usize, depth: u32) -> Option<(usize, Geometry)> {
        let exponent_ceiling = modulus_bits.trailing_zeros();
        let low = 1_u32;
        let high = exponent_ceiling.saturating_sub(1);

        let mut winner: Option<(usize, Geometry)> = None;
        let mut probe = low;
        while probe <= high {
            for whole_step in [false, true] {
                if let Some((cost, geometry)) =
                    Self::price_geometry(probe, modulus_bits, depth, whole_step)
                    && winner.as_ref().is_none_or(|(best, _)| cost < *best)
                {
                    winner = Some((cost, geometry));
                }
            }
            probe = probe.wrapping_add(1);
        }
        winner
    }

    /// Centre of the exponent search, at every level of the recursion.
    ///
    /// The classical Schönhage-Strassen split cuts `N` bits into `sqrt(N)` pieces
    /// of `sqrt(N)` bits, so the transform exponent tracks `log2(N) / 2`. The
    /// discrete cost model evaluates all candidate exponents around this analytic
    /// minimum.
    ///
    /// The nested search used to anchor to the basecase width instead, at
    /// `log2(bits) - log2(SSA_BASE_MODULUS_BITS) + 2`. That deviates from the
    /// optimum by `log2/2 - 11`, which is unbounded, so a nested ring wide enough
    /// was priced with a window that could not reach its own optimum — a 2^26-bit
    /// inner ring searched exponents 1 to 5 for an optimum near 13.
    ///
    /// Moving the nested search here could not be done alone: the better-placed
    /// window finds cheaper nested geometries, so nesting looked cheaper still and
    /// top-level decisions tipped onto slower nested geometries. It is
    /// `SSA_NESTED_COST_PENALTY_16THS` that pays for the correction, and the two
    /// only work together.
    ///
    /// `nested_ring_bits` rounds an inner ring
    /// to a multiple of whatever transform length the nested planner will choose,
    /// so it must use this same centre and no other.
    pub fn search_centre(modulus_bits: usize) -> u32 {
        if modulus_bits == 0 {
            return 1;
        }
        modulus_bits.ilog2().div_euclid(2).max(1)
    }

    /// Prices one geometry, including its recursive pointwise products.
    pub fn price_geometry(
        exponent: u32,
        modulus_bits: usize,
        depth: u32,
        whole_step: bool,
    ) -> Option<(usize, Geometry)> {
        let geometry = Geometry::for_exponent(exponent, modulus_bits, whole_step)?;
        // Let `K = 2^exponent`, `M = modulus_bits / K`, and
        // `required = 2M + log2(K)`. The inner ring width is the aligned value
        // `n >= required`. If `required / n < 1/2`, the `K / 2` transform fits
        // in the same ring: its requirement is `4M + log2(K) - 1`, which is
        // strictly below `n`. It performs half as many pointwise products at
        // the same coefficient width, so this geometry is dominated.
        let required_inner_bits = geometry
            .chunk_bits
            .checked_mul(2)?
            .checked_add(geometry.transform_log)?;
        if required_inner_bits.checked_mul(2)? < geometry.inner_bits {
            return None;
        }
        let nests = geometry.inner_bits > SSA_BASE_MODULUS_BITS;
        if nests && geometry.inner_bits >= modulus_bits {
            return None;
        }
        let inner_cl = SsaRing::coeff_limbs(geometry.inner_bits);

        // Two forward transforms, one inverse, forward twist, inverse untwist, and
        // reconstruction visit each coefficient. Odd half-steps add sqrt(2) work.
        let sqrt2_passes = if geometry.twist_step_half.is_multiple_of(2) {
            0
        } else {
            SSA_SQRT2_TWIST_PASSES
        };
        let passes = geometry
            .transform_log
            .checked_mul(2)?
            .checked_add(3)?
            .checked_add(sqrt2_passes)?;
        let visit_cost = inner_cl.checked_add(SSA_COEFFICIENT_VISIT_OVERHEAD)?;
        let transform_cost = geometry
            .transform_len
            .checked_mul(passes)?
            .checked_mul(visit_cost)?;

        // A nested pointwise stage is charged its own modelled transform cost and
        // nothing else, which omits everything nesting costs outside the
        // arithmetic. `SSA_NESTED_COST_PENALTY_16THS` restores it.
        let unit_product = {
            let modelled = ring_cost(geometry.inner_bits, depth.wrapping_add(1));
            if nests {
                modelled
                    .saturating_mul(SSA_NESTED_COST_PENALTY_16THS)
                    .div_euclid(16)
            } else {
                modelled
            }
        };
        let pointwise_cost = geometry.transform_len.checked_mul(unit_product)?;
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
    /// Rounding to the smallest such multiple rather than to the next power of two
    /// is what keeps a 98,304-limb product from costing the same as a 131,072-limb
    /// one.
    pub fn crt_half_width(required_bits: usize) -> Option<usize> {
        let half_limb_bits = LIMB_BITS.checked_mul(2)?;
        let minimum = required_bits.div_ceil(half_limb_bits).max(2);
        if minimum <= SSA_BNM1_BASECASE_LIMBS {
            return Some(minimum);
        }

        let mut winner: Option<usize> = None;
        let mut step_log = 0_u32;
        while let Some(step) = 1_usize.checked_shl(step_log) {
            if step > minimum {
                break;
            }
            let blocks = minimum.div_ceil(step);
            if blocks <= SSA_BNM1_BASECASE_LIMBS
                && let Some(candidate) = blocks.checked_mul(step)
                && winner.is_none_or(|current| candidate < current)
            {
                winner = Some(candidate);
            }
            step_log = step_log.checked_add(1)?;
        }
        winner
    }
}

/// Cost of one product in a Fermat ring of the given width.
fn ring_cost(modulus_bits: usize, depth: u32) -> usize {
    if modulus_bits <= SSA_BASE_MODULUS_BITS || depth >= MAX_COST_RECURSION_DEPTH {
        return SsaPlan::basecase_product_cost(SsaRing::coeff_limbs(modulus_bits));
    }
    SsaPlan::best_exponent(modulus_bits, depth).map_or_else(
        || SsaPlan::basecase_product_cost(SsaRing::coeff_limbs(modulus_bits)),
        |(cost, _)| cost,
    )
}
