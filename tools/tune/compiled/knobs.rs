//! Candidate grids attached to the shared typed parameter registry.

use super::{Parameter, ScoreDomain, TuningProfile};

/// Candidate SSA Fermat-modulus bit widths.
pub const BASE_MODULUS_BITS: &[usize] = &[8_192, 16_384, 24_576, 32_768, 49_152, 65_536];
/// Candidate Bernstein-Nussbaumer basecase limb widths.
pub const BNM1_BASECASE_LIMBS: &[usize] = &[16, 24, 32, 48, 64, 96];
/// Candidate factor-3 transform admission thresholds.
pub const FACTOR3_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
/// Candidate factor-5 transform admission thresholds.
pub const FACTOR5_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
/// Candidate direct-shift limb limits.
pub const DIRECT_SHIFT_LIMBS: &[usize] = &[0, 4, 6, 8, 9, 10, 12, 16];
/// Candidate scalar-shift crossover widths.
pub const SHIFT_SCALAR_THRESHOLDS: &[usize] = &[32, 64, 96, 128, 192, 256];
/// Candidate blocked-shift widths.
pub const SHIFT_BLOCK_WIDTHS: &[usize] = &[16, 32, 64, 128, 256];
/// Candidate Toom-8.5 paired-reconstruction limb floors.
pub const PAIRED_RECONSTRUCTION_LIMBS: &[usize] = &[64, 96, 128, 192, 256, 384, 512, 768, 1_024];
/// Candidate Toom-8 full-guard product split floors.
pub const FULL_GUARD_PRODUCT_SPLIT_LIMBS: &[usize] = &[128, 192, 256, 320, 384, 448, 512, 640];
/// Candidate division recursion geometry widths.
pub const DIVISION_RECURSION_LIMBS: &[usize] = &[16, 24, 32, 40, 48, 64, 96];
const BURNIKEL_THRESHOLDS: &[usize] = &[8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256];
const NEWTON_THRESHOLDS: &[usize] = &[
    128, 192, 256, 384, 512, 640, 768, 1_024, 1_536, 2_048, 3_072, 4_096, 8_192, 16_384, 32_768,
    65_536,
];
const PARSING_ENTRIES: &[usize] = &[
    8, 16, 19, 32, 48, 64, 96, 128, 192, 224, 256, 288, 320, 384, 512, 768, 1_024,
];

/// One compiled-profile coordinate and its scoring catalog.
#[derive(Clone, Copy, Debug)]
pub struct Knob<'candidates> {
    /// Registered constant name recorded in the run report.
    pub name: &'static str,
    /// Discrete values the coordinate search may install.
    pub candidates: &'candidates [usize],
    /// Rebuild-worker domain that times this coordinate.
    pub domain: ScoreDomain,
    /// Registered profile reader.
    pub get: fn(TuningProfile) -> usize,
    /// Registered profile writer.
    pub set: fn(&mut TuningProfile, usize),
}

impl<'candidates> Knob<'candidates> {
    /// Binds candidates to one registry entry without repeating field accessors.
    pub const fn new(
        parameter: Parameter,
        candidates: &'candidates [usize],
        domain: ScoreDomain,
    ) -> Self {
        Self {
            name: parameter.name,
            candidates,
            domain,
            get: parameter.get,
            set: parameter.set,
        }
    }
}

/// The shared parsing leaf precedes the radix-group entries.
pub const PARSING_KNOBS: &[Knob<'static>] = &[
    Knob::new(
        Parameter::RADIX_PARSE_LEAF_CHUNKS,
        &[1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
        ScoreDomain::Parsing,
    ),
    Knob::new(
        Parameter::RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD,
        PARSING_ENTRIES,
        ScoreDomain::Parsing,
    ),
    Knob::new(
        Parameter::RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD,
        PARSING_ENTRIES,
        ScoreDomain::Parsing,
    ),
    Knob::new(
        Parameter::RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD,
        PARSING_ENTRIES,
        ScoreDomain::Parsing,
    ),
];

/// Compiled Toom reconstruction coordinates.
pub const TOOM_KNOBS: [Knob<'static>; 2] = [
    Knob::new(
        Parameter::TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS,
        PAIRED_RECONSTRUCTION_LIMBS,
        ScoreDomain::Toom85,
    ),
    Knob::new(
        Parameter::TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS,
        FULL_GUARD_PRODUCT_SPLIT_LIMBS,
        ScoreDomain::Toom85Mul,
    ),
];

/// Compiled SSA kernel and geometry coordinates.
pub const SSA_KNOBS: [Knob<'static>; 10] = [
    Knob::new(
        Parameter::SSA_BASE_MODULUS_BITS,
        BASE_MODULUS_BITS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_BNM1_BASECASE_LIMBS,
        BNM1_BASECASE_LIMBS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_NEGACYCLIC_FACTOR3_THRESHOLD,
        FACTOR3_THRESHOLDS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_NEGACYCLIC_FACTOR5_THRESHOLD,
        FACTOR5_THRESHOLDS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_DIRECT_SHIFT_MAX_LIMBS,
        DIRECT_SHIFT_LIMBS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_SHIFT_SCALAR_THRESHOLD,
        SHIFT_SCALAR_THRESHOLDS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_SHIFT_BLOCK_WIDTH,
        SHIFT_BLOCK_WIDTHS,
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_COEFFICIENT_VISIT_OVERHEAD,
        &[1, 2, 4, 8, 16, 32, 64],
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_BASECASE_COST_WEIGHT_16THS,
        &[4, 8, 12, 16, 24, 32, 48, 64],
        ScoreDomain::Ssa,
    ),
    Knob::new(
        Parameter::SSA_NESTED_COST_PENALTY_16THS,
        &[8, 12, 16, 20, 24, 32, 48, 64],
        ScoreDomain::Ssa,
    ),
];

/// Production shapes for transform admission and lopsided blocking policies.
pub const TRANSFORM_SHAPE_KNOBS: [Knob<'static>; 4] = [
    Knob::new(
        Parameter::BALANCED_TOOM8_THRESHOLD,
        &[0, 64, 128, 192, 256, 320, 384, 512, 768, 1_024],
        ScoreDomain::ProductionShapes,
    ),
    Knob::new(
        Parameter::LOPSIDED_TRANSFORM_BLOCK_RATIO,
        &[1, 2, 4, 8, 16, 32, 64],
        ScoreDomain::ProductionShapes,
    ),
    Knob::new(
        Parameter::TRANSFORM_MIN_SMALLER_LIMBS,
        &[256, 512, 1_100, 2_048, 4_096],
        ScoreDomain::ProductionShapes,
    ),
    Knob::new(
        Parameter::TRANSFORM_MAX_OPERAND_RATIO,
        &[8, 16, 32, 64, 128],
        ScoreDomain::ProductionShapes,
    ),
];

/// Parallel admission and work budgets, measured on an explicit CPU set.
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
pub const PARALLEL_KNOBS: [Knob<'static>; 3] = [
    Knob::new(
        Parameter::SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD,
        &[
            0, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304,
        ],
        ScoreDomain::ParallelSsa,
    ),
    Knob::new(
        Parameter::SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS,
        &[1, 2, 3, 4, 6, 8, 12, 16],
        ScoreDomain::ParallelSsa,
    ),
    Knob::new(
        Parameter::SSA_PARALLEL_MIN_LIMB_WORK,
        &[128, 512, 2_048, 8_192, 32_768, 131_072, 524_288, 2_097_152],
        ScoreDomain::ParallelSsa,
    ),
];

/// Independent division geometry; mathematical guard bounds are excluded.
pub const DIVISION_KNOBS: [Knob<'static>; 16] = [
    Knob::new(
        Parameter::BURNIKEL_ZIEGLER_BLOCK_LIMBS,
        DIVISION_RECURSION_LIMBS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::NEWTON_RAPHSON_BASECASE_LIMBS,
        DIVISION_RECURSION_LIMBS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::BURNIKEL_QUOTIENT_THRESHOLD,
        BURNIKEL_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::BURNIKEL_LONG_QUOTIENT_THRESHOLD,
        BURNIKEL_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::NEWTON_QUOTIENT_THRESHOLD,
        NEWTON_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::APPROXIMATE_DIVISION_BLOCK_LIMBS,
        DIVISION_RECURSION_LIMBS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::NEWTON_SMALL_QUOTIENT_BLOCK_RATIO,
        &[1, 2, 3, 4, 6, 8],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_TRUNCATION_RATIO,
        &[1, 2, 3, 4],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_DIVISIBLE_THRESHOLD,
        BURNIKEL_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_STACK_LIMBS,
        &[4, 8, 16, 32, 64, 96, 128, 192, 256, 512],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_SINGLE_NORMALIZED_PREINVERSE,
        &[2, 3, 4, 5, 8, 16, 32, 64],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_SINGLE_UNNORMALIZED_PREINVERSE,
        &[2, 3, 4, 5, 8, 16, 32, 64],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_BASECASE_QUOTIENT_MAX_LIMBS,
        &[1, 2, 3, 4, 6, 8, 16],
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::NEWTON_RAPHSON_THRESHOLD,
        NEWTON_THRESHOLDS,
        ScoreDomain::ProductionDivision,
    ),
    Knob::new(
        Parameter::DIVISION_SMALL_QUOTIENT_MAX,
        &[2, 4, 8, 16, 32, 64],
        ScoreDomain::ProductionDivision,
    ),
];

/// Direct product objectives, with Montgomery and Newton regression guards.
pub const PRODUCT_KNOBS: [Knob<'static>; 2] = [
    Knob::new(
        Parameter::MONTGOMERY_CIOS_MAX_LIMBS,
        &[1, 4, 8, 16, 24, 32, 48, 64, 96, 128],
        ScoreDomain::Products,
    ),
    Knob::new(
        Parameter::MUL_MOD_BNM1_THRESHOLD,
        &[
            0, 64, 128, 256, 512, 1_024, 1_536, 2_048, 3_072, 4_096, 8_192, 16_384,
        ],
        ScoreDomain::Products,
    ),
];

/// GCD policies include growing cofactors and shrinking remainders.
pub const GCD_KNOBS: [Knob<'static>; 10] = [
    Knob::new(
        Parameter::BINARY_EUCLID_DIVISION_SHIFT,
        &[1, 4, 8, 12, 16, 24, 32, 64],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::LEHMER_BRANCHLESS_THRESHOLD,
        &[1, 8, 16, 32, 48, 64, 96, 128, 256, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::WIDE_LEHMER_THRESHOLD,
        &[3, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1_024, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::LEHMER_FUSED_UPDATE_MAX_LIMBS,
        &[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 256, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::HGCD_BLOCK_THRESHOLD,
        &[16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::HGCD_CROSSOVER_THRESHOLD,
        &[16, 32, 48, 64, 96, 128, 192, 256, 512, 1_024, 2_048, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::EXTENDED_GCD_WIDE_THRESHOLD,
        &[3, 8, 16, 24, 32, 48, 64, 128, 256, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::EXTENDED_HGCD_CROSSOVER_THRESHOLD,
        &[16, 32, 48, 64, 96, 128, 256, 512, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS,
        &[3, 32, 64, 128, 256, 512, 16_384],
        ScoreDomain::Gcd,
    ),
    Knob::new(
        Parameter::EXTENDED_GCD_COFACTOR_BATCH_RATIO,
        &[1, 2, 3, 4, 8],
        ScoreDomain::Gcd,
    ),
];
