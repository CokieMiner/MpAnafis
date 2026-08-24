//! Single built-in default and retained target-selection entry point.

use super::TuningProfile;

impl TuningProfile {
    /// Current built-in default, seeded from the documented Zen 5 measurements.
    ///
    /// This is deliberately the only built-in default. Target-specific dispatch
    /// remains in [`profile_for_target`] so independently measured profiles can
    /// be added later without changing the build script, tuner, generated source,
    /// or production consumers.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            radix_decimal_recursive: 8,
            radix_small_recursive: 10,
            radix_large_recursive: 12,
            karatsuba: 20,
            toom_cook_3: 536,
            toom_cook_4: 800,
            toom_cook_6: 2_049,
            toom_cook_85: 2_419,
            // A forced Karatsuba/Toom-8 sweep found the first sustained
            // top-level win here. Recursive child thresholds remain separate.
            balanced_toom8: 288,
            toom85_paired_reconstruction_min_limbs: 256,
            toom8_full_guard_product_min_split_limbs: 384,
            sqr_karatsuba: 56,
            sqr_toom_cook_3: 202,
            sqr_toom_cook_4: 320,
            sqr_toom_cook_6: 398,
            sqr_toom_cook_85: 398,
            burnikel_ziegler: 192,
            newton_raphson: 3_072,
            burnikel_ziegler_block: 64,
            newton_reciprocal_basecase: 40,
            // Forced Toom-8/SSA pairs sustain wins for both one- and
            // sixteen-worker execution from this width onward.
            ssa: 2_976,
            sqr_ssa: 2_304,
            transform_min_smaller_limbs: 1_100,
            transform_max_operand_ratio: 32,
            ssa_base_modulus_bits: 65_536,
            ssa_bnm1_basecase_limbs: 128,
            ssa_negacyclic_factor3: 48,
            ssa_negacyclic_factor5: 32,
            ssa_coefficient_visit_overhead: 16,
            ssa_basecase_cost_weight_16ths: 16,
            ssa_nested_cost_penalty_16ths: 20,
            // Full-ladder forced scoring retained nine as the crossover
            // between direct and fused coefficient shifts.
            ssa_direct_shift_max_limbs: 9,
            // Direct Fermat is admitted only for the measured wide Rayon pool;
            // narrower pools have no separate disabled threshold field.
            ssa_direct_fermat_parallel_threshold: 1_048_576,
            ssa_direct_fermat_parallel_min_workers: 8,
            // A 16-worker sweep retained 512 after cache-sized and
            // million-limb guard cells rejected the nearby candidates.
            ssa_parallel_min_limb_work: 512,
        }
    }
}

/// Resolve the built-in default for a target.
///
/// The target arguments are intentionally retained as the stable dispatch
/// boundary used by the build script and tuner. They become active when that
/// target has its own independently measured profile.
#[must_use]
pub const fn profile_for_target(_target_arch: &str, _pointer_width: &str) -> TuningProfile {
    TuningProfile::portable()
}
