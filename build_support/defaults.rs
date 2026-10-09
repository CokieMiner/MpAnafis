//! Built-in tuning values shared across target selections.

use super::TuningProfile;

impl TuningProfile {
    /// Resolves the built-in default for a target.
    ///
    /// The build script and tuner share this selection boundary. All architecture
    /// and pointer-width inputs currently select the same profile.
    #[must_use]
    pub const fn for_target(_target_arch: &str, _pointer_width: &str) -> Self {
        Self::portable()
    }

    /// Built-in performance policies, grouped by arithmetic family.
    ///
    /// Parameter units and admissible values are defined by [`TuningProfile`].
    /// Measured uncertainty is recorded in tuning-session reports.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            // Multiplication tower and coefficient reconstruction, in limbs.
            karatsuba: 18,
            toom_cook_3: 259,
            toom_cook_4: 313,
            toom_cook_6: 512,
            toom_cook_85: 512,
            balanced_toom8: 314,
            toom85_paired_reconstruction_min_limbs: 128,
            toom8_full_guard_product_min_split_limbs: 448,
            low_product_recursive: 72,
            low_product_full: 259,
            mul_mod_bnm1: 3_073,

            // Independent squaring tower, in limbs.
            sqr_karatsuba: 28,
            sqr_toom_cook_3: 191,
            sqr_toom_cook_4: 278,
            sqr_toom_cook_6: 512,
            sqr_toom_cook_85: 512,

            // Transform admission, ring geometry, and coefficient cost policies.
            ssa: 3_073,
            sqr_ssa: 2_049,
            transform_min_smaller_limbs: 1_100,
            transform_max_operand_ratio: 32,
            lopsided_transform_block_ratio: 16,
            ssa_base_modulus_bits: 32_768,
            ssa_bnm1_basecase_limbs: 128,
            ssa_negacyclic_factor3: 48,
            ssa_negacyclic_factor5: 64,
            ssa_coefficient_visit_overhead: 16,
            ssa_basecase_cost_weight_16ths: 16,
            ssa_nested_cost_penalty_16ths: 20,
            ssa_direct_shift_max_limbs: 9,
            ssa_shift_scalar_threshold: 128,
            ssa_shift_block_width: 64,
            cache_block_bytes: 512 * 1024,

            // Parallel transform admission and minimum work per worker.
            ssa_direct_fermat_parallel_threshold: 1_048_576,
            ssa_direct_fermat_parallel_min_workers: 8,
            ssa_parallel_min_limb_work: 128,

            // Division output dispatch, recursive geometry, and primitive leaves.
            burnikel_ziegler: 48,
            newton_raphson: 1_800,
            burnikel_quotient: 144,
            burnikel_long_quotient: 48,
            newton_quotient: 1_800,
            division_small_quotient_max: 8,
            burnikel_ziegler_block: 64,
            newton_reciprocal_basecase: 64,
            approximate_division_block: 64,
            newton_small_quotient_block_ratio: 3,
            division_truncation_ratio: 2,
            division_divisible: 48,
            division_stack_limbs: 128,
            division_single_normalized_preinverse: 2,
            division_single_unnormalized_preinverse: 2,
            division_basecase_quotient_max_limbs: 2,

            // Modular exponentiation crossover, in modulus limbs.
            montgomery_pow_mod: 149,
            montgomery_cios_max_limbs: 32,

            // Plain GCD recursion and Lehmer simulation/update policies.
            hgcd_crossover: 123,
            hgcd_block_threshold: 24,
            lehmer_fused_update_max: 7,
            wide_lehmer_threshold: 178,
            lehmer_branchless_threshold: 64,
            binary_euclid_division_shift: 16,

            // Extended Euclid entry and cofactor completion policies.
            extended_hgcd_crossover: 64,
            extended_gcd_wide_threshold: 48,
            extended_gcd_cofactor_batch_min: 128,
            extended_gcd_cofactor_batch_ratio: 2,

            // Formatting inputs are limb widths; parsing inputs are radix chunks.
            radix_format_decimal_recursive: 49,
            radix_format_small_recursive: 5,
            radix_format_large_recursive: 7,
            radix_parse_decimal_recursive: 320,
            radix_parse_small_recursive: 320,
            radix_parse_large_recursive: 384,
            radix_parse_leaf: 32,
        }
    }
}
