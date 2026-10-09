//! Mutually exclusive tuner modes and worker argument selection.

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use core::mem::size_of;
use std::env::args;

/// Search, validation, rendering and subprocess-worker modes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    All,
    TiersOnly,
    CompiledOnly,
    DivisionOnly,
    ToomOnly,
    FormattingOnly,
    ParsingOnly,
    ModularOnly,
    GcdOnly,
    ParallelOnly(String),
    ValidateParallelOnly(String),
    ValidateOnly,
    Check,
    ScoreSsa,
    ScoreSsaCoarse,
    ScoreSsaMul,
    ScoreSsaMulCoarse,
    ScoreToom85,
    ScoreToom85Mul,
    ScoreBurnikel,
    ScoreNewton,
    ScoreProductionDivision(String),
    ScoreProducts(String),
    ScoreParallelSsa(String),
    ScoreParallelProduction(String),
    ScoreProduction(String),
    ScoreProductionShapes,
    ScoreGcd,
    ScoreConsumers,
    ScoreHoldout,
    ScoreParsing(String),
    ScoreMulPair(String),
    ScoreLowPair(String),
    ScoreSqrPair(String),
    ScorePowPair(String),
    ScoreFmtPair(String),
    ScoreGcdPair(String),
    ProfileFor(String),
    Help,
}

impl Mode {
    /// Validate the selected command-line mode before host discovery or builds.
    pub fn selected() -> Result<Self, String> {
        let mut selected = Self::All;
        let mut arguments = args().skip(1);
        while let Some(argument) = arguments.next() {
            let candidate = if let Some(specification) = argument.strip_prefix("--parallel-only=") {
                Self::ParallelOnly(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--validate-parallel-only=") {
                Self::ValidateParallelOnly(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-parallel-ssa=") {
                Self::ScoreParallelSsa(specification.to_owned())
            } else if let Some(specification) =
                argument.strip_prefix("--score-parallel-production=")
            {
                Self::ScoreParallelProduction(specification.to_owned())
            } else if let Some(specification) =
                argument.strip_prefix("--score-production-division=")
            {
                Self::ScoreProductionDivision(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-products=") {
                Self::ScoreProducts(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-production=") {
                Self::ScoreProduction(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-mul-pair=") {
                Self::ScoreMulPair(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-low-pair=") {
                Self::ScoreLowPair(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-sqr-pair=") {
                Self::ScoreSqrPair(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-pow-pair=") {
                Self::ScorePowPair(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-fmt-pair=") {
                Self::ScoreFmtPair(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-parsing=") {
                Self::ScoreParsing(specification.to_owned())
            } else if let Some(specification) = argument.strip_prefix("--score-gcd-pair=") {
                Self::ScoreGcdPair(specification.to_owned())
            } else {
                match argument.as_str() {
                    "--tiers-only" => Self::TiersOnly,
                    "--compiled-only" => Self::CompiledOnly,
                    "--division-only" => Self::DivisionOnly,
                    "--toom-only" => Self::ToomOnly,
                    "--formatting-only" => Self::FormattingOnly,
                    "--parsing-only" => Self::ParsingOnly,
                    "--modular-only" => Self::ModularOnly,
                    "--gcd-only" => Self::GcdOnly,
                    "--validate-only" => Self::ValidateOnly,
                    "--check" => Self::Check,
                    "--score-ssa" => Self::ScoreSsa,
                    "--score-ssa-coarse" => Self::ScoreSsaCoarse,
                    "--score-ssa-mul" => Self::ScoreSsaMul,
                    "--score-ssa-mul-coarse" => Self::ScoreSsaMulCoarse,
                    "--score-toom85" => Self::ScoreToom85,
                    "--score-toom85-mul" => Self::ScoreToom85Mul,
                    "--score-burnikel" => Self::ScoreBurnikel,
                    "--score-newton" => Self::ScoreNewton,
                    "--score-production-division" => Self::ScoreProductionDivision(String::new()),
                    "--score-production" => Self::ScoreProduction(String::new()),
                    "--score-products" => Self::ScoreProducts(String::new()),
                    "--score-production-shapes" => Self::ScoreProductionShapes,
                    "--score-gcd" => Self::ScoreGcd,
                    "--score-consumers" => Self::ScoreConsumers,
                    "--score-holdout" => Self::ScoreHoldout,
                    "--score-parsing" => Self::ScoreParsing(String::new()),
                    "--profile-for" => {
                        let Some(target) = arguments.next() else {
                            return Err("--profile-for requires a target architecture".to_owned());
                        };
                        Self::ProfileFor(target)
                    }
                    "--help" | "-h" => Self::Help,
                    _ => return Err(format!("unknown mp-tune option: {argument}; use --help")),
                }
            };
            if selected != Self::All && selected != candidate {
                return Err("mp-tune modes are mutually exclusive".to_owned());
            }
            selected = candidate;
        }
        Ok(selected)
    }

    /// Parse explicit fixture widths and prove the largest constructed span fits.
    ///
    /// Empty input selects the built-in grid; malformed or unaddressable widths
    /// are rejected before operand allocation or arithmetic dispatch.
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    pub fn limb_widths(specification: &str) -> Result<Vec<usize>, String> {
        if specification.is_empty() {
            return Ok(Vec::new());
        }
        let widths = specification
            .split(',')
            .map(|field| {
                field
                    .parse::<usize>()
                    .map_err(|error| format!("invalid fixture width: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if widths.iter().any(|&width| {
            width == 0
                || width
                    .checked_mul(4)
                    .and_then(|span| span.checked_add(2))
                    .and_then(|span| span.checked_mul(size_of::<usize>()))
                    .is_none_or(|bytes| isize::try_from(bytes).is_err())
        }) {
            return Err("fixture widths exceed the addressable span".to_owned());
        }
        Ok(widths)
    }
}
