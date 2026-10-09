//! One field-to-constant registry shared by profile I/O and compiled searches.

use super::TuningProfile;

/// Identity and typed accessors of one independent performance input.
#[derive(Clone, Copy, Debug)]
pub struct Parameter {
    /// Stable generated Rust constant name.
    pub name: &'static str,
    /// Reads exactly the registered profile field.
    pub get: fn(TuningProfile) -> usize,
    /// Writes exactly the registered profile field.
    pub set: fn(&mut TuningProfile, usize),
    /// Excludes transform-only constants from 16-bit library builds.
    pub wide_only: bool,
}

// This host-side registry generates metadata only. Library builds receive plain
// usize constants, with no runtime lookup or function-pointer dispatch.
macro_rules! parameters {
    ($($field:ident => $constant:ident, $wide_only:literal;)+) => {
        impl Parameter {
            $(
                #[doc = concat!("Typed access to `TuningProfile::", stringify!($field), "`.")]
                pub const $constant: Self = Self {
                    name: stringify!($constant),
                    get: |profile| profile.$field,
                    set: |profile, value| profile.$field = value,
                    wide_only: $wide_only,
                };
            )+
            /// Complete profile declaration order, including target exclusions.
            pub const ALL: &'static [Self] = &[$(Self::$constant,)+];
        }
    };
}

parameters! {
    radix_format_decimal_recursive => RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD, false;
    radix_format_small_recursive => RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD, false;
    radix_format_large_recursive => RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD, false;
    radix_parse_decimal_recursive => RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD, false;
    radix_parse_small_recursive => RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD, false;
    radix_parse_large_recursive => RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD, false;
    radix_parse_leaf => RADIX_PARSE_LEAF_CHUNKS, false;
    hgcd_crossover => HGCD_CROSSOVER_THRESHOLD, false;
    extended_hgcd_crossover => EXTENDED_HGCD_CROSSOVER_THRESHOLD, false;
    extended_gcd_wide_threshold => EXTENDED_GCD_WIDE_THRESHOLD, false;
    extended_gcd_cofactor_batch_min => EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS, false;
    extended_gcd_cofactor_batch_ratio => EXTENDED_GCD_COFACTOR_BATCH_RATIO, false;
    hgcd_block_threshold => HGCD_BLOCK_THRESHOLD, false;
    lehmer_fused_update_max => LEHMER_FUSED_UPDATE_MAX_LIMBS, false;
    wide_lehmer_threshold => WIDE_LEHMER_THRESHOLD, false;
    lehmer_branchless_threshold => LEHMER_BRANCHLESS_THRESHOLD, false;
    binary_euclid_division_shift => BINARY_EUCLID_DIVISION_SHIFT, false;
    karatsuba => KARATSUBA_THRESHOLD, false;
    low_product_recursive => LOW_PRODUCT_RECURSIVE_THRESHOLD, false;
    low_product_full => LOW_PRODUCT_FULL_THRESHOLD, false;
    mul_mod_bnm1 => MUL_MOD_BNM1_THRESHOLD, true;
    toom_cook_3 => TOOM_COOK_THRESHOLD, false;
    toom_cook_4 => TOOM_COOK_4_THRESHOLD, false;
    toom_cook_6 => TOOM_COOK_6_THRESHOLD, false;
    toom_cook_85 => TOOM_COOK_85_THRESHOLD, false;
    balanced_toom8 => BALANCED_TOOM8_THRESHOLD, false;
    lopsided_transform_block_ratio => LOPSIDED_TRANSFORM_BLOCK_RATIO, false;
    toom85_paired_reconstruction_min_limbs => TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, false;
    toom8_full_guard_product_min_split_limbs => TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, false;
    sqr_karatsuba => SQR_KARATSUBA_THRESHOLD, false;
    sqr_toom_cook_3 => SQR_TOOM_COOK_THRESHOLD, false;
    sqr_toom_cook_4 => SQR_TOOM_COOK_4_THRESHOLD, false;
    sqr_toom_cook_6 => SQR_TOOM_COOK_6_THRESHOLD, false;
    sqr_toom_cook_85 => SQR_TOOM_COOK_85_THRESHOLD, false;
    burnikel_ziegler => BURNIKEL_ZIEGLER_THRESHOLD, false;
    newton_raphson => NEWTON_RAPHSON_THRESHOLD, false;
    burnikel_quotient => BURNIKEL_QUOTIENT_THRESHOLD, false;
    burnikel_long_quotient => BURNIKEL_LONG_QUOTIENT_THRESHOLD, false;
    newton_quotient => NEWTON_QUOTIENT_THRESHOLD, false;
    division_small_quotient_max => DIVISION_SMALL_QUOTIENT_MAX, false;
    burnikel_ziegler_block => BURNIKEL_ZIEGLER_BLOCK_LIMBS, false;
    newton_reciprocal_basecase => NEWTON_RAPHSON_BASECASE_LIMBS, false;
    approximate_division_block => APPROXIMATE_DIVISION_BLOCK_LIMBS, false;
    newton_small_quotient_block_ratio => NEWTON_SMALL_QUOTIENT_BLOCK_RATIO, false;
    division_truncation_ratio => DIVISION_TRUNCATION_RATIO, false;
    division_divisible => DIVISION_DIVISIBLE_THRESHOLD, false;
    division_stack_limbs => DIVISION_STACK_LIMBS, false;
    division_single_normalized_preinverse => DIVISION_SINGLE_NORMALIZED_PREINVERSE, false;
    division_single_unnormalized_preinverse => DIVISION_SINGLE_UNNORMALIZED_PREINVERSE, false;
    division_basecase_quotient_max_limbs => DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, false;
    montgomery_pow_mod => MONTGOMERY_POW_MOD_THRESHOLD, false;
    montgomery_cios_max_limbs => MONTGOMERY_CIOS_MAX_LIMBS, false;
    ssa => SSA_THRESHOLD, true;
    sqr_ssa => SQR_SSA_THRESHOLD, true;
    transform_min_smaller_limbs => TRANSFORM_MIN_SMALLER_LIMBS, true;
    transform_max_operand_ratio => TRANSFORM_MAX_OPERAND_RATIO, true;
    ssa_base_modulus_bits => SSA_BASE_MODULUS_BITS, true;
    ssa_bnm1_basecase_limbs => SSA_BNM1_BASECASE_LIMBS, true;
    ssa_negacyclic_factor3 => SSA_NEGACYCLIC_FACTOR3_THRESHOLD, true;
    ssa_negacyclic_factor5 => SSA_NEGACYCLIC_FACTOR5_THRESHOLD, true;
    ssa_coefficient_visit_overhead => SSA_COEFFICIENT_VISIT_OVERHEAD, true;
    ssa_basecase_cost_weight_16ths => SSA_BASECASE_COST_WEIGHT_16THS, true;
    ssa_nested_cost_penalty_16ths => SSA_NESTED_COST_PENALTY_16THS, true;
    ssa_direct_shift_max_limbs => SSA_DIRECT_SHIFT_MAX_LIMBS, true;
    ssa_shift_scalar_threshold => SSA_SHIFT_SCALAR_THRESHOLD, true;
    ssa_shift_block_width => SSA_SHIFT_BLOCK_WIDTH, true;
    ssa_direct_fermat_parallel_threshold => SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD, true;
    ssa_direct_fermat_parallel_min_workers => SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS, true;
    ssa_parallel_min_limb_work => SSA_PARALLEL_MIN_LIMB_WORK, true;
    cache_block_bytes => CACHE_BLOCK_BYTES, true;
}
