//! Render and parse complete generated Rust profiles.

use super::TuningProfile;

pub const REQUIRED_DEFINITIONS: [&str; 35] = [
    "const RADIX_DECIMAL_RECURSIVE_THRESHOLD:",
    "const RADIX_SMALL_RECURSIVE_THRESHOLD:",
    "const RADIX_LARGE_RECURSIVE_THRESHOLD:",
    "const KARATSUBA_THRESHOLD:",
    "const TOOM_COOK_THRESHOLD:",
    "const TOOM_COOK_4_THRESHOLD:",
    "const TOOM_COOK_6_THRESHOLD:",
    "const TOOM_COOK_85_THRESHOLD:",
    "const BALANCED_TOOM8_THRESHOLD:",
    "const TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS:",
    "const TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS:",
    "const SQR_KARATSUBA_THRESHOLD:",
    "const SQR_TOOM_COOK_THRESHOLD:",
    "const SQR_TOOM_COOK_4_THRESHOLD:",
    "const SQR_TOOM_COOK_6_THRESHOLD:",
    "const SQR_TOOM_COOK_85_THRESHOLD:",
    "const BURNIKEL_ZIEGLER_THRESHOLD:",
    "const NEWTON_RAPHSON_THRESHOLD:",
    "const BURNIKEL_ZIEGLER_BLOCK_LIMBS:",
    "const NEWTON_RAPHSON_BASECASE_LIMBS:",
    "const SSA_THRESHOLD:",
    "const SQR_SSA_THRESHOLD:",
    "const TRANSFORM_MIN_SMALLER_LIMBS:",
    "const TRANSFORM_MAX_OPERAND_RATIO:",
    "const SSA_BASE_MODULUS_BITS:",
    "const SSA_BNM1_BASECASE_LIMBS:",
    "const SSA_NEGACYCLIC_FACTOR3_THRESHOLD:",
    "const SSA_NEGACYCLIC_FACTOR5_THRESHOLD:",
    "const SSA_COEFFICIENT_VISIT_OVERHEAD:",
    "const SSA_BASECASE_COST_WEIGHT_16THS:",
    "const SSA_NESTED_COST_PENALTY_16THS:",
    "const SSA_DIRECT_SHIFT_MAX_LIMBS:",
    "const SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD:",
    "const SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS:",
    "const SSA_PARALLEL_MIN_LIMB_WORK:",
];

impl TuningProfile {
    /// Parse a complete rendered profile and return its typed values.
    pub fn from_source(source: &str) -> Result<Self, String> {
        let mut profile = Self::portable();
        macro_rules! read {
            ($field:ident, $name:literal) => {
                profile.$field = parse_constant(source, $name)?;
            };
        }
        read!(radix_decimal_recursive, "RADIX_DECIMAL_RECURSIVE_THRESHOLD");
        read!(radix_small_recursive, "RADIX_SMALL_RECURSIVE_THRESHOLD");
        read!(radix_large_recursive, "RADIX_LARGE_RECURSIVE_THRESHOLD");
        read!(karatsuba, "KARATSUBA_THRESHOLD");
        read!(toom_cook_3, "TOOM_COOK_THRESHOLD");
        read!(toom_cook_4, "TOOM_COOK_4_THRESHOLD");
        read!(toom_cook_6, "TOOM_COOK_6_THRESHOLD");
        read!(toom_cook_85, "TOOM_COOK_85_THRESHOLD");
        read!(balanced_toom8, "BALANCED_TOOM8_THRESHOLD");
        read!(
            toom85_paired_reconstruction_min_limbs,
            "TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS"
        );
        read!(
            toom8_full_guard_product_min_split_limbs,
            "TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS"
        );
        read!(sqr_karatsuba, "SQR_KARATSUBA_THRESHOLD");
        read!(sqr_toom_cook_3, "SQR_TOOM_COOK_THRESHOLD");
        read!(sqr_toom_cook_4, "SQR_TOOM_COOK_4_THRESHOLD");
        read!(sqr_toom_cook_6, "SQR_TOOM_COOK_6_THRESHOLD");
        read!(sqr_toom_cook_85, "SQR_TOOM_COOK_85_THRESHOLD");
        read!(burnikel_ziegler, "BURNIKEL_ZIEGLER_THRESHOLD");
        read!(newton_raphson, "NEWTON_RAPHSON_THRESHOLD");
        read!(burnikel_ziegler_block, "BURNIKEL_ZIEGLER_BLOCK_LIMBS");
        read!(newton_reciprocal_basecase, "NEWTON_RAPHSON_BASECASE_LIMBS");
        read!(ssa, "SSA_THRESHOLD");
        read!(sqr_ssa, "SQR_SSA_THRESHOLD");
        read!(transform_min_smaller_limbs, "TRANSFORM_MIN_SMALLER_LIMBS");
        read!(transform_max_operand_ratio, "TRANSFORM_MAX_OPERAND_RATIO");
        read!(ssa_base_modulus_bits, "SSA_BASE_MODULUS_BITS");
        read!(ssa_bnm1_basecase_limbs, "SSA_BNM1_BASECASE_LIMBS");
        read!(ssa_negacyclic_factor3, "SSA_NEGACYCLIC_FACTOR3_THRESHOLD");
        read!(ssa_negacyclic_factor5, "SSA_NEGACYCLIC_FACTOR5_THRESHOLD");
        read!(
            ssa_coefficient_visit_overhead,
            "SSA_COEFFICIENT_VISIT_OVERHEAD"
        );
        read!(
            ssa_basecase_cost_weight_16ths,
            "SSA_BASECASE_COST_WEIGHT_16THS"
        );
        read!(
            ssa_nested_cost_penalty_16ths,
            "SSA_NESTED_COST_PENALTY_16THS"
        );
        read!(ssa_direct_shift_max_limbs, "SSA_DIRECT_SHIFT_MAX_LIMBS");
        read!(
            ssa_direct_fermat_parallel_threshold,
            "SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD"
        );
        read!(
            ssa_direct_fermat_parallel_min_workers,
            "SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS"
        );
        read!(ssa_parallel_min_limb_work, "SSA_PARALLEL_MIN_LIMB_WORK");
        Ok(profile)
    }

    /// Render a complete Rust source profile with `header` above the constants.
    #[must_use]
    pub fn render(self, header: &str) -> String {
        format!(
            "{header}\n\
             pub const RADIX_DECIMAL_RECURSIVE_THRESHOLD: usize = {};\n\
             pub const RADIX_SMALL_RECURSIVE_THRESHOLD: usize = {};\n\
             pub const RADIX_LARGE_RECURSIVE_THRESHOLD: usize = {};\n\
             pub const KARATSUBA_THRESHOLD: usize = {};\n\
             pub const TOOM_COOK_THRESHOLD: usize = {};\n\
             pub const TOOM_COOK_4_THRESHOLD: usize = {};\n\
             pub const TOOM_COOK_6_THRESHOLD: usize = {};\n\
             pub const TOOM_COOK_85_THRESHOLD: usize = {};\n\
             pub const BALANCED_TOOM8_THRESHOLD: usize = {};\n\
             pub const TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS: usize = {};\n\
             pub const TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS: usize = {};\n\
             pub const SQR_KARATSUBA_THRESHOLD: usize = {};\n\
             pub const SQR_TOOM_COOK_THRESHOLD: usize = {};\n\
             pub const SQR_TOOM_COOK_4_THRESHOLD: usize = {};\n\
             pub const SQR_TOOM_COOK_6_THRESHOLD: usize = {};\n\
             pub const SQR_TOOM_COOK_85_THRESHOLD: usize = {};\n\
             pub const BURNIKEL_ZIEGLER_THRESHOLD: usize = {};\n\
             pub const NEWTON_RAPHSON_THRESHOLD: usize = {};\n\
             pub const BURNIKEL_ZIEGLER_BLOCK_LIMBS: usize = {};\n\
             pub const NEWTON_RAPHSON_BASECASE_LIMBS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SQR_SSA_THRESHOLD: usize = {};\n\
             pub const TRANSFORM_MIN_SMALLER_LIMBS: usize = {};\n\
             pub const TRANSFORM_MAX_OPERAND_RATIO: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_BASE_MODULUS_BITS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_BNM1_BASECASE_LIMBS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_NEGACYCLIC_FACTOR3_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_NEGACYCLIC_FACTOR5_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_COEFFICIENT_VISIT_OVERHEAD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_BASECASE_COST_WEIGHT_16THS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_NESTED_COST_PENALTY_16THS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_DIRECT_SHIFT_MAX_LIMBS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             pub const SSA_PARALLEL_MIN_LIMB_WORK: usize = {};",
            format_constant(self.radix_decimal_recursive),
            format_constant(self.radix_small_recursive),
            format_constant(self.radix_large_recursive),
            format_constant(self.karatsuba),
            format_constant(self.toom_cook_3),
            format_constant(self.toom_cook_4),
            format_constant(self.toom_cook_6),
            format_constant(self.toom_cook_85),
            format_constant(self.balanced_toom8),
            format_constant(self.toom85_paired_reconstruction_min_limbs),
            format_constant(self.toom8_full_guard_product_min_split_limbs),
            format_constant(self.sqr_karatsuba),
            format_constant(self.sqr_toom_cook_3),
            format_constant(self.sqr_toom_cook_4),
            format_constant(self.sqr_toom_cook_6),
            format_constant(self.sqr_toom_cook_85),
            format_constant(self.burnikel_ziegler),
            format_constant(self.newton_raphson),
            format_constant(self.burnikel_ziegler_block),
            format_constant(self.newton_reciprocal_basecase),
            format_constant(self.ssa),
            format_constant(self.sqr_ssa),
            format_constant(self.transform_min_smaller_limbs),
            format_constant(self.transform_max_operand_ratio),
            format_constant(self.ssa_base_modulus_bits),
            format_constant(self.ssa_bnm1_basecase_limbs),
            format_constant(self.ssa_negacyclic_factor3),
            format_constant(self.ssa_negacyclic_factor5),
            format_constant(self.ssa_coefficient_visit_overhead),
            format_constant(self.ssa_basecase_cost_weight_16ths),
            format_constant(self.ssa_nested_cost_penalty_16ths),
            format_constant(self.ssa_direct_shift_max_limbs),
            format_constant(self.ssa_direct_fermat_parallel_threshold),
            format_constant(self.ssa_direct_fermat_parallel_min_workers),
            format_constant(self.ssa_parallel_min_limb_work),
        )
    }
}

fn format_constant(value: usize) -> String {
    let s = value.to_string();
    if s.len() <= 4 {
        return s;
    }
    let mut result = String::new();
    let char_count = s.chars().count();
    for (i, ch) in s.chars().enumerate() {
        let rev_pos = char_count.saturating_sub(i);
        if i > 0 && rev_pos.is_multiple_of(3) {
            result.push('_');
        }
        result.push(ch);
    }
    result
}

fn parse_constant(source: &str, name: &str) -> Result<usize, String> {
    let marker = format!("const {name}: usize =");
    let Some(start) = source.find(&marker) else {
        return Err(format!("missing constant {name}"));
    };
    let offset = start
        .checked_add(marker.len())
        .ok_or_else(|| format!("constant {name} offset overflows usize"))?;
    let expression = source
        .get(offset..)
        .ok_or_else(|| format!("constant {name} has an invalid UTF-8 offset"))?
        .split_once(';')
        .map(|(value, _)| value.trim())
        .ok_or_else(|| format!("constant {name} has no terminating semicolon"))?;
    let compact: String = expression
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact == "usize::MAX-1" {
        return Ok(usize::MAX - 1);
    }
    compact
        .replace('_', "")
        .parse::<usize>()
        .map_err(|_error| format!("constant {name} is not a usize literal: {expression}"))
}

/// Return the first missing definition in a tuned profile source.
#[must_use]
pub fn missing_definition(source: &str) -> Option<&'static str> {
    REQUIRED_DEFINITIONS
        .iter()
        .find(|definition| !source.contains(**definition))
        .copied()
}
