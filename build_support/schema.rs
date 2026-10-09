//! Typed tuning-profile schema and semantic validation.

use super::Validation;

/// Complete integer performance profile consumed by generated builds.
///
/// The profile contains only independent performance inputs: measured algorithm
/// crossovers, recursion geometry, and planner coefficients. Exact relationships
/// are derived at their consumer instead of becoming profile fields (for example,
/// SSA coefficient widths derive from a prepared transform plan). The host tuner
/// may search several fields jointly. Model coefficients
/// are searched as coupled execution policies rather than fitted physical costs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TuningProfile {
    /// Recursion cutoff for decimal string formatting.
    pub radix_format_decimal_recursive: usize,
    /// Recursion cutoff for small radix string formatting (radix <= 36).
    pub radix_format_small_recursive: usize,
    /// Recursion cutoff for arbitrary-base radix string formatting.
    pub radix_format_large_recursive: usize,
    /// Decimal parsing entry width, in limb-sized radix chunks.
    pub radix_parse_decimal_recursive: usize,
    /// Parsing entry width for non-power-of-two radices 3..=9, in chunks.
    pub radix_parse_small_recursive: usize,
    /// Parsing entry width for non-power-of-two radices 11..=36, in chunks.
    pub radix_parse_large_recursive: usize,
    /// Largest decoded chunk block accumulated without recursive multiplication.
    pub radix_parse_leaf: usize,
    /// Lehmer-to-half-GCD production crossover in limbs.
    pub hgcd_crossover: usize,
    /// Retained-matrix half-GCD crossover for extended Euclid, in limbs.
    pub extended_hgcd_crossover: usize,
    /// Initial second-operand width selecting wide Lehmer simulation in extended Euclid.
    pub extended_gcd_wide_threshold: usize,
    /// Minimum cofactor width admitting recursive completion of extended Euclid.
    pub extended_gcd_cofactor_batch_min: usize,
    /// Cofactor-to-remainder width ratio admitting recursive completion.
    pub extended_gcd_cofactor_batch_ratio: usize,
    /// Minimum block width for half-GCD matrix construction and recursive cutoff.
    pub hgcd_block_threshold: usize,
    /// Largest equal-width Lehmer update fused into one paired limb pass.
    pub lehmer_fused_update_max: usize,
    /// Width at which Lehmer switches from single-limb to double-limb quotient simulation.
    pub wide_lehmer_threshold: usize,
    /// Initial problem width selecting masked narrow-quotient corrections.
    pub lehmer_branchless_threshold: usize,
    /// Bit gap selecting scalar Euclidean reduction before binary GCD/Jacobi.
    /// A value at least the target limb width disables the initial division.
    pub binary_euclid_division_shift: usize,
    /// Basecase-to-Karatsuba multiplication crossover in limbs.
    pub karatsuba: usize,
    /// Triangular-to-recursive low-product crossover in limbs.
    pub low_product_recursive: usize,
    /// Recursive low-product to full-multiplication crossover in limbs.
    /// Zero disables the full-product bypass.
    pub low_product_full: usize,
    /// Minimum requested modulus width for a cyclic product modulo B^n-1.
    /// Shared by Newton refinement and Montgomery arithmetic; zero disables it.
    pub mul_mod_bnm1: usize,
    /// Karatsuba-to-Toom-Cook-3 multiplication crossover in limbs.
    pub toom_cook_3: usize,
    /// Toom-Cook-3 to Toom-Cook-4 multiplication crossover in limbs.
    pub toom_cook_4: usize,
    /// Entry width for Toom-6. Equal Toom-6 and Toom-8.5 thresholds intentionally
    /// shadow Toom-6 because dispatch offers the higher tier first.
    pub toom_cook_6: usize,
    /// Entry width for Toom-8.5 multiplication.
    pub toom_cook_85: usize,
    /// Optional balanced top-level Toom-8 entry that bypasses lower Toom
    /// gates without changing their recursive-child crossovers.
    pub balanced_toom8: usize,
    /// Preferred transform-block width as a multiple of the shorter operand.
    pub lopsided_transform_block_ratio: usize,
    /// Minimum operand limb width for paired coefficient reconstruction in Toom-8.5.
    pub toom85_paired_reconstruction_min_limbs: usize,
    /// Minimum split-block width in limbs for full guard products during Toom-8 evaluation.
    pub toom8_full_guard_product_min_split_limbs: usize,
    /// Basecase-to-Karatsuba squaring crossover in limbs.
    pub sqr_karatsuba: usize,
    /// Karatsuba-to-Toom-Cook-3 squaring crossover in limbs.
    pub sqr_toom_cook_3: usize,
    /// Toom-Cook-3 to Toom-Cook-4 squaring crossover in limbs.
    pub sqr_toom_cook_4: usize,
    /// Entry width for Toom-6 squaring.
    pub sqr_toom_cook_6: usize,
    /// Entry width for Toom-8.5 squaring.
    pub sqr_toom_cook_85: usize,
    /// Basecase-to-Burnikel-Ziegler division crossover in limbs.
    pub burnikel_ziegler: usize,
    /// Burnikel-Ziegler to Newton-Raphson division crossover in limbs.
    pub newton_raphson: usize,
    /// Short Algorithm D to recursive quotient-only division crossover.
    pub burnikel_quotient: usize,
    /// Algorithm D to recursive quotient division for quotient spans wider
    /// than the divisor.
    pub burnikel_long_quotient: usize,
    /// Recursive quotient-only division to Newton crossover.
    pub newton_quotient: usize,
    /// Largest leading-limb quotient estimate admitted by the scalar correction path.
    pub division_small_quotient_max: usize,
    /// Finite even width of the smallest divisor that divide-and-conquer
    /// division splits; narrower blocks use Algorithm D. An even width keeps
    /// the halves of that smallest split equal.
    pub burnikel_ziegler_block: usize,
    /// Reciprocal basecase cutoff for Newton-Raphson iteration.
    pub newton_reciprocal_basecase: usize,
    /// Last divisor width evaluated by an approximate quotient leaf.
    pub approximate_division_block: usize,
    /// One Newton block is selected when its quotient width is at most
    /// the divisor width divided by this ratio.
    pub newton_small_quotient_block_ratio: usize,
    /// Divisor-to-retained-prefix ratio admitting basecase truncation.
    pub division_truncation_ratio: usize,
    /// Divisor and excess-numerator widths selecting recursive divisibility.
    pub division_divisible: usize,
    /// Normalization stack capacity in limbs, including the high guard.
    pub division_stack_limbs: usize,
    /// Minimum remaining numerator width amortizing a normalized limb reciprocal.
    pub division_single_normalized_preinverse: usize,
    /// Minimum remaining numerator width amortizing an unnormalized limb reciprocal.
    pub division_single_unnormalized_preinverse: usize,
    /// Largest quotient span sent directly to Algorithm D at dispatch.
    pub division_basecase_quotient_max_limbs: usize,
    /// Montgomery-to-Barrett modular exponentiation crossover in limbs.
    pub montgomery_pow_mod: usize,
    /// Largest modulus width using integrated scalar Montgomery multiplication.
    /// At least one retains the one-limb domain without reciprocal refinement.
    pub montgomery_cios_max_limbs: usize,
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
    /// Basecase cutoff for SSA convolution stages.
    pub ssa_bnm1_basecase_limbs: usize,
    /// Threshold limb count for activating radices with factor 3 in negacyclic convolution.
    pub ssa_negacyclic_factor3: usize,
    /// Threshold limb count for activating radices with factor 5 in negacyclic convolution.
    pub ssa_negacyclic_factor5: usize,
    /// Estimated overhead weight per coefficient butterfly pass.
    pub ssa_coefficient_visit_overhead: usize,
    /// Relative cost multiplier (in units of 1/16) for basecase operations.
    pub ssa_basecase_cost_weight_16ths: usize,
    /// Relative penalty multiplier (in units of 1/16) for nested recursive transforms.
    pub ssa_nested_cost_penalty_16ths: usize,
    /// Largest coefficient width using the direct Fermat shift loop; zero
    /// disables that loop.
    pub ssa_direct_shift_max_limbs: usize,
    /// Negated shift width below which scalar shift/subtract replaces staged blocks.
    pub ssa_shift_scalar_threshold: usize,
    /// Power-of-two stack block width for staged negated shifts, at most 256 limbs.
    pub ssa_shift_block_width: usize,
    /// Minimum CRT half-width for direct Fermat multiplication with a parallel
    /// executor; zero disables the strategy.
    pub ssa_direct_fermat_parallel_threshold: usize,
    /// Minimum Rayon worker count allowed to select the parallel direct-Fermat
    /// strategy. Ignored when its threshold is zero.
    pub ssa_direct_fermat_parallel_min_workers: usize,
    /// Minimum estimated limb-work per Rayon worker before an SSA pass forks.
    pub ssa_parallel_min_limb_work: usize,
    /// Cache budget in bytes for cache-blocked operations.
    pub cache_block_bytes: usize,
}

impl TuningProfile {
    /// Validates contracts shared by generated and built-in profiles.
    ///
    /// # Errors
    ///
    /// Returns an error if a required input is zero, a crossover is out of order,
    /// or a geometry or planner input violates its range constraints. Optional
    /// strategies may use zero to disable execution.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.radix_format_decimal_recursive == 0
            || self.radix_format_small_recursive == 0
            || self.radix_format_large_recursive == 0
        {
            return Err("formatting thresholds must be nonzero");
        }
        for entry in [
            self.radix_parse_decimal_recursive,
            self.radix_parse_small_recursive,
            self.radix_parse_large_recursive,
        ] {
            if !Validation::valid_finite(entry)
                || !Validation::valid_finite(self.radix_parse_leaf)
                || entry <= self.radix_parse_leaf
            {
                return Err("parsing entry thresholds must exceed finite nonzero leaf cutoffs");
            }
        }
        self.validate_gcd()?;
        self.validate_division()?;
        if self.low_product_recursive < 4 || !Validation::valid_finite(self.low_product_recursive) {
            return Err("low-product recursion requires a finite cutoff of at least four limbs");
        }
        if self.low_product_full >= usize::MAX - 1 {
            return Err("low-product full multiplication cutoff must fit finite limb bounds");
        }
        if self.mul_mod_bnm1 >= usize::MAX - 1 {
            return Err("cyclic-product cutoff must fit finite limb bounds; zero disables it");
        }
        if !Validation::valid_finite(self.lopsided_transform_block_ratio) {
            return Err("lopsided transform block ratio must be finite and nonzero");
        }
        let mul_chain = [
            self.karatsuba,
            self.toom_cook_3,
            self.toom_cook_4,
            self.toom_cook_6,
            self.toom_cook_85,
        ];
        if !Validation::valid_threshold_chain(&mul_chain) {
            return Err("multiplication thresholds are unordered or contain an invalid sentinel");
        }
        let sqr_chain = [
            self.sqr_karatsuba,
            self.sqr_toom_cook_3,
            self.sqr_toom_cook_4,
            self.sqr_toom_cook_6,
            self.sqr_toom_cook_85,
        ];
        if !Validation::valid_threshold_chain(&sqr_chain) {
            return Err("squaring thresholds are unordered or contain an invalid sentinel");
        }
        if !Validation::valid_finite(self.toom85_paired_reconstruction_min_limbs)
            || !Validation::valid_finite(self.toom8_full_guard_product_min_split_limbs)
        {
            return Err("Toom geometry thresholds must be finite and nonzero");
        }
        if !Validation::valid_optional_crossover(self.balanced_toom8, self.karatsuba) {
            return Err("balanced Toom-8 must be disabled or follow Karatsuba");
        }
        if self.balanced_toom8 != 0 && self.ssa != 0 && self.balanced_toom8 >= self.ssa {
            return Err("balanced Toom-8 must precede an enabled SSA crossover");
        }
        if !Validation::valid_finite(self.montgomery_pow_mod) {
            return Err("modular exponentiation crossover must be finite and nonzero");
        }
        if !Validation::valid_finite(self.montgomery_cios_max_limbs) {
            return Err("Montgomery CIOS width must be finite and at least one limb");
        }
        if !Validation::valid_transform_crossover(self.ssa, self.toom_cook_85, &mul_chain)
            || !Validation::valid_transform_crossover(
                self.sqr_ssa,
                self.sqr_toom_cook_85,
                &sqr_chain,
            )
        {
            return Err("disabled transform crossovers must be zero or valid later thresholds");
        }
        if !Validation::valid_finite(self.transform_min_smaller_limbs)
            || !Validation::valid_finite(self.transform_max_operand_ratio)
            || !Validation::valid_finite(self.ssa_base_modulus_bits)
            || !Validation::valid_finite(self.ssa_bnm1_basecase_limbs)
            || !Validation::valid_finite(self.ssa_negacyclic_factor3)
            || !Validation::valid_finite(self.ssa_negacyclic_factor5)
            || !Validation::valid_finite(self.ssa_coefficient_visit_overhead)
            || !Validation::valid_finite(self.ssa_basecase_cost_weight_16ths)
            || !Validation::valid_finite(self.ssa_nested_cost_penalty_16ths)
            || self.ssa_direct_shift_max_limbs >= usize::MAX - 1
            || !Validation::valid_finite(self.ssa_shift_scalar_threshold)
            || !self.ssa_shift_block_width.is_power_of_two()
            || self.ssa_shift_block_width > 256
            || self.ssa_direct_fermat_parallel_threshold >= usize::MAX - 1
            || !Validation::valid_finite(self.ssa_direct_fermat_parallel_min_workers)
            || !Validation::valid_finite(self.ssa_parallel_min_limb_work)
            || !Validation::valid_finite(self.cache_block_bytes)
        {
            return Err("SSA geometry and planner coefficients are invalid");
        }
        Ok(())
    }

    /// Validates output dispatch, reciprocal geometry, and bounded stack storage.
    fn validate_division(self) -> Result<(), &'static str> {
        if self.division_small_quotient_max < 2
            || self.division_small_quotient_max > usize::from(u16::MAX)
        {
            return Err(
                "scalar quotient limit must be between 2 and the smallest supported limb maximum",
            );
        }
        if !Validation::valid_finite(self.burnikel_ziegler)
            || !Validation::valid_finite(self.newton_raphson)
            || self.burnikel_ziegler > self.newton_raphson
        {
            return Err("division thresholds must be finite, nonzero and ordered");
        }
        if !Validation::valid_finite(self.burnikel_quotient)
            || !Validation::valid_finite(self.newton_quotient)
            || self.burnikel_quotient > self.newton_quotient
            || !Validation::valid_finite(self.burnikel_long_quotient)
            || self.burnikel_long_quotient > self.newton_quotient
            || self.approximate_division_block < 4
            || !Validation::valid_finite(self.approximate_division_block)
            || !Validation::valid_finite(self.newton_small_quotient_block_ratio)
            || !Validation::valid_finite(self.division_truncation_ratio)
            || !Validation::valid_finite(self.division_divisible)
            || !(4..=512).contains(&self.division_stack_limbs)
            || self.division_single_normalized_preinverse < 2
            || !Validation::valid_finite(self.division_single_normalized_preinverse)
            || self.division_single_unnormalized_preinverse < 2
            || !Validation::valid_finite(self.division_single_unnormalized_preinverse)
            || !Validation::valid_finite(self.division_basecase_quotient_max_limbs)
        {
            return Err(
                "division output cutoffs, block ratios, and bounded stack capacity are invalid",
            );
        }
        if !Validation::valid_finite(self.burnikel_ziegler_block)
            || !self.burnikel_ziegler_block.is_multiple_of(2)
            || !Validation::valid_finite(self.newton_reciprocal_basecase)
        {
            return Err(
                "division geometry must be finite and nonzero; Burnikel-Ziegler block width must be even",
            );
        }
        Ok(())
    }

    /// Validates reduction cutoffs, scalar shift range, and HGCD geometry.
    fn validate_gcd(self) -> Result<(), &'static str> {
        if !Validation::valid_finite(self.hgcd_crossover)
            || !Validation::valid_finite(self.extended_hgcd_crossover)
            || !Validation::valid_finite(self.extended_gcd_wide_threshold)
            || !Validation::valid_finite(self.extended_gcd_cofactor_batch_min)
            || !Validation::valid_finite(self.extended_gcd_cofactor_batch_ratio)
            || !Validation::valid_finite(self.hgcd_block_threshold)
            || !Validation::valid_finite(self.lehmer_fused_update_max)
            || !Validation::valid_finite(self.wide_lehmer_threshold)
            || !Validation::valid_finite(self.lehmer_branchless_threshold)
        {
            return Err("GCD thresholds must be finite and nonzero");
        }
        if !(1..=64).contains(&self.binary_euclid_division_shift) {
            return Err("scalar division shift must be in 1..=64");
        }
        if self.hgcd_block_threshold > self.hgcd_crossover {
            return Err("HGCD block threshold cannot exceed the HGCD crossover threshold");
        }
        Ok(())
    }
}

impl Default for TuningProfile {
    fn default() -> Self {
        Self::portable()
    }
}
