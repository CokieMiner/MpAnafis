//! Typed tuning-profile schema and semantic validation.

use super::{valid_finite, valid_optional_crossover, valid_threshold_chain};

/// Complete integer performance profile consumed by generated builds.
///
/// The profile contains only independent performance inputs: measured algorithm
/// crossovers, recursion geometry, and planner coefficients. Exact relationships
/// are derived at their consumer instead of becoming profile fields (for example,
/// the Mulders cutoff derives from the Karatsuba crossover and SSA coefficient
/// widths derive from a prepared transform plan). Presence here does not imply
/// that the host tuner can identify a field independently; model coefficients
/// retain architecture defaults until an isolated calibration exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TuningProfile {
    pub radix_decimal_recursive: usize,
    pub radix_small_recursive: usize,
    pub radix_large_recursive: usize,
    pub karatsuba: usize,
    pub toom_cook_3: usize,
    pub toom_cook_4: usize,
    /// Entry width for Toom-6. Equal Toom-6 and Toom-8.5 thresholds intentionally
    /// shadow Toom-6 because dispatch offers the higher tier first.
    pub toom_cook_6: usize,
    pub toom_cook_85: usize,
    /// Optional balanced top-level Toom-8 entry that bypasses lower Toom
    /// gates without changing their recursive-child crossovers.
    pub balanced_toom8: usize,
    pub toom85_paired_reconstruction_min_limbs: usize,
    pub toom8_full_guard_product_min_split_limbs: usize,
    pub sqr_karatsuba: usize,
    pub sqr_toom_cook_3: usize,
    pub sqr_toom_cook_4: usize,
    pub sqr_toom_cook_6: usize,
    pub sqr_toom_cook_85: usize,
    pub burnikel_ziegler: usize,
    pub newton_raphson: usize,
    /// Base block size for Burnikel-Ziegler recursion, independent of dispatch.
    pub burnikel_ziegler_block: usize,
    /// Reciprocal basecase cutoff for Newton-Raphson iteration.
    pub newton_reciprocal_basecase: usize,
    /// Conventional multiplication tower to SSA crossover; zero disables SSA.
    pub ssa: usize,
    /// Conventional squaring tower to SSA crossover; zero disables SSA.
    pub sqr_ssa: usize,
    /// Shorter-operand floor below which a transform loses to blocking.
    pub transform_min_smaller_limbs: usize,
    /// Largest longer-to-shorter operand ratio admitted to one transform.
    pub transform_max_operand_ratio: usize,
    /// Widest inner ring left to the multiplication tower, in bits.
    pub ssa_base_modulus_bits: usize,
    pub ssa_bnm1_basecase_limbs: usize,
    pub ssa_negacyclic_factor3: usize,
    pub ssa_negacyclic_factor5: usize,
    pub ssa_coefficient_visit_overhead: usize,
    pub ssa_basecase_cost_weight_16ths: usize,
    pub ssa_nested_cost_penalty_16ths: usize,
    /// Largest coefficient width using the direct Fermat shift loop; zero
    /// disables that loop.
    pub ssa_direct_shift_max_limbs: usize,
    /// Minimum CRT half-width for direct Fermat multiplication with a parallel
    /// executor; zero disables the strategy.
    pub ssa_direct_fermat_parallel_threshold: usize,
    /// Minimum Rayon worker count allowed to select the parallel direct-Fermat
    /// strategy. Ignored when its threshold is zero.
    pub ssa_direct_fermat_parallel_min_workers: usize,
    /// Minimum estimated limb-work per Rayon worker before an SSA pass forks.
    pub ssa_parallel_min_limb_work: usize,
}

impl TuningProfile {
    /// Validate contracts shared by generated and built-in profiles.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.radix_decimal_recursive == 0
            || self.radix_small_recursive == 0
            || self.radix_large_recursive == 0
        {
            return Err("formatting thresholds must be nonzero");
        }
        if !valid_threshold_chain(&[
            self.karatsuba,
            self.toom_cook_3,
            self.toom_cook_4,
            self.toom_cook_6,
            self.toom_cook_85,
        ]) {
            return Err("multiplication thresholds are unordered or contain an invalid sentinel");
        }
        if !valid_threshold_chain(&[
            self.sqr_karatsuba,
            self.sqr_toom_cook_3,
            self.sqr_toom_cook_4,
            self.sqr_toom_cook_6,
            self.sqr_toom_cook_85,
        ]) {
            return Err("squaring thresholds are unordered or contain an invalid sentinel");
        }
        if !valid_finite(self.toom85_paired_reconstruction_min_limbs)
            || !valid_finite(self.toom8_full_guard_product_min_split_limbs)
        {
            return Err("Toom geometry thresholds must be finite and nonzero");
        }
        if !valid_optional_crossover(self.balanced_toom8, self.karatsuba) {
            return Err("balanced Toom-8 must be disabled or follow Karatsuba");
        }
        if self.balanced_toom8 != 0 && self.ssa != 0 && self.balanced_toom8 >= self.ssa {
            return Err("balanced Toom-8 must precede an enabled SSA crossover");
        }
        if self.burnikel_ziegler == 0
            || self.newton_raphson == 0
            || self.burnikel_ziegler >= self.newton_raphson
        {
            return Err("division thresholds must be nonzero and ordered");
        }
        if !valid_finite(self.burnikel_ziegler_block)
            || !valid_finite(self.newton_reciprocal_basecase)
        {
            return Err("division geometry must be finite and nonzero");
        }
        if !valid_optional_crossover(self.ssa, self.toom_cook_85)
            || !valid_optional_crossover(self.sqr_ssa, self.sqr_toom_cook_85)
        {
            return Err("disabled transform crossovers must be zero or valid later thresholds");
        }
        if !valid_finite(self.transform_min_smaller_limbs)
            || !valid_finite(self.transform_max_operand_ratio)
            || !valid_finite(self.ssa_base_modulus_bits)
            || !valid_finite(self.ssa_bnm1_basecase_limbs)
            || !valid_finite(self.ssa_negacyclic_factor3)
            || !valid_finite(self.ssa_negacyclic_factor5)
            || !valid_finite(self.ssa_coefficient_visit_overhead)
            || !valid_finite(self.ssa_basecase_cost_weight_16ths)
            || !valid_finite(self.ssa_nested_cost_penalty_16ths)
            || self.ssa_direct_shift_max_limbs >= usize::MAX - 1
            || self.ssa_direct_fermat_parallel_threshold >= usize::MAX - 1
            || !valid_finite(self.ssa_direct_fermat_parallel_min_workers)
            || !valid_finite(self.ssa_parallel_min_limb_work)
        {
            return Err("SSA geometry and planner coefficients are invalid");
        }
        Ok(())
    }
}

impl Default for TuningProfile {
    fn default() -> Self {
        Self::portable()
    }
}
