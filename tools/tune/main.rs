//! Host hardware tuning through compiled profiles and paired measurements.
//!
//! Search follows arithmetic dependencies. Fresh median-ratio bounds confirm
//! candidates, and production validation gates profile installation.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "the tuner is an interactive command-line measurement tool"
)]

use std::env::var;

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use self::worker::ParallelWorker;
#[cfg(not(target_pointer_width = "16"))]
use self::worker::{
    DivisionGrid, DivisionScoreDomain, DivisionWorker, HoldoutWorker, ProductGrid, ProductWorker,
};
use self::{
    arguments::Mode,
    compiled::CompiledTuner,
    crossovers::{Calibration, Crossovers, Request},
    harness::{
        CandidateHarness, ComparisonRule, FormattingPairSpec, SCORE_SCALE, ScoreDomain, ScoreLimit,
        TierPairSpec,
    },
    measure::{CrossoverMeasure, InterleavedMeasure, ProbeQuality},
    platform::Platform,
    session::TuneSession,
    smoke::SmokeCheck,
    store::{ProfileWriter, SCORE_CACHE_NAME, ScoreStore},
    tuning_profile::{Parameter, TuningProfile},
    validation::Validation,
    worker::{
        CONSUMER_SCORE_COUNT, ConsumerWorker, GCD_SCORE_CASES, GcdWorker, HASH_A, HASH_B,
        PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, PairWorkers, ParsingWorker, ProfileWorkers,
        ScoreCell, SsaScoreQuality, TOOM85_MUL_SCORE_CELLS,
    },
};

#[path = "../../build_support/mod.rs"]
mod tuning_profile;

mod arguments;
mod compiled;
mod crossovers;
mod harness;
mod measure;
mod platform;
mod session;
mod smoke;
mod store;
mod tiers;
mod validation;
mod worker;

fn main() -> Result<(), String> {
    let mode = Mode::selected()?;
    let selection = var("MP_TUNING_CELLS").unwrap_or_default();
    match &mode {
        Mode::ScoreSsa => ProfileWorkers::print_ssa_score(SsaScoreQuality::Precise, &selection),
        Mode::ScoreSsaCoarse => {
            ProfileWorkers::print_ssa_score(SsaScoreQuality::Coarse, &selection)
        }
        Mode::ScoreSsaMul => {
            ProfileWorkers::print_ssa_mul_score(SsaScoreQuality::Precise);
            Ok(())
        }
        Mode::ScoreSsaMulCoarse => {
            ProfileWorkers::print_ssa_mul_score(SsaScoreQuality::Coarse);
            Ok(())
        }
        Mode::ScoreToom85 => ProfileWorkers::print_toom85_score(&selection),
        Mode::ScoreToom85Mul => ProfileWorkers::print_toom85_mul_score(&selection),
        Mode::ScoreBurnikel => DivisionWorker::print_score(
            DivisionScoreDomain::Burnikel,
            &DivisionGrid::default(),
            &selection,
        ),
        Mode::ScoreNewton => DivisionWorker::print_score(
            DivisionScoreDomain::Newton,
            &DivisionGrid::default(),
            &selection,
        ),
        Mode::ScoreProductionDivision(specification) => {
            let grid = DivisionGrid::parse(specification)?;
            DivisionWorker::print_score(DivisionScoreDomain::Production, &grid, &selection)
        }
        Mode::ScoreProduction(specification) => {
            let grid = DivisionGrid::parse(specification)?;
            ProfileWorkers::print_production_score(&grid);
            Ok(())
        }
        Mode::ScoreProducts(specification) => ProductWorker::print_score(specification, &selection),
        Mode::ScoreProductionShapes => ProfileWorkers::print_production_shapes_score(&selection),
        Mode::ScoreParallelSsa(specification) | Mode::ScoreParallelProduction(specification) => {
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            {
                let (cpus, width_text) =
                    specification.split_once(';').unwrap_or((specification, ""));
                let widths = Mode::limb_widths(width_text)?;
                ParallelWorker::print_score(
                    cpus,
                    matches!(mode, Mode::ScoreParallelProduction(_)),
                    &widths,
                    &selection,
                )
            }
            #[cfg(not(all(feature = "rayon", not(target_pointer_width = "16"))))]
            {
                let _ = specification;
                Err("parallel tuning requires rayon and at least 32-bit pointers".to_owned())
            }
        }
        Mode::ScoreGcd => GcdWorker::print_score(&selection),
        Mode::ScoreConsumers => {
            ConsumerWorker::print_score();
            Ok(())
        }
        Mode::ScoreHoldout => HoldoutWorker::print_score(&selection),
        Mode::ScoreParsing(specification) => ParsingWorker::print_score(specification, &selection),
        Mode::ScoreMulPair(specification) => PairWorkers::print_mul_pair_score(specification),
        Mode::ScoreLowPair(specification) => PairWorkers::print_low_pair_score(specification),
        Mode::ScoreSqrPair(specification) => PairWorkers::print_sqr_pair_score(specification),
        Mode::ScorePowPair(specification) => PairWorkers::print_pow_pair_score(specification),
        Mode::ScoreFmtPair(specification) => PairWorkers::print_fmt_pair_score(specification),
        Mode::ScoreGcdPair(specification) => PairWorkers::print_gcd_pair_score(specification),
        Mode::ProfileFor(target_arch) => ProfileWriter::write_target_profile(target_arch)
            .map_err(|error| format!("cannot write target profile: {error}")),
        Mode::Help => {
            print_help();
            Ok(())
        }
        Mode::Check => SmokeCheck::check_workers(),
        Mode::ValidateParallelOnly(specification) => {
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            {
                TuneSession::run_parallel_validation(specification)
            }
            #[cfg(not(all(feature = "rayon", not(target_pointer_width = "16"))))]
            {
                let _ = specification;
                Err("parallel validation requires rayon and at least 32-bit pointers".to_owned())
            }
        }
        Mode::All
        | Mode::TiersOnly
        | Mode::CompiledOnly
        | Mode::DivisionOnly
        | Mode::ToomOnly
        | Mode::FormattingOnly
        | Mode::ParsingOnly
        | Mode::ModularOnly
        | Mode::GcdOnly
        | Mode::ParallelOnly(_)
        | Mode::ValidateOnly => TuneSession::run(&mode),
    }
}

fn print_help() {
    println!(
        "Usage: mp-tune [--check | --tiers-only | --compiled-only | --division-only | --toom-only | --modular-only | --gcd-only | --formatting-only | --parsing-only | --validate-only | --profile-for <arch>]\n\
         \n\
         With no mode, tunes the complete profile in 8 phases:\n\
         1. Toom-8.5 kernels and multiplication\n\
         2. Squaring\n\
         3. SSA, transforms, and direct product policies\n\
         4. Division geometry and dispatch\n\
         5. Modular exponentiation\n\
         6. Greatest common divisor (GCD)\n\
         7. Radix formatting and parsing\n\
         8. Production validation\n\
         --check          Check rebuild workers and paired execution without installing a profile.\n\
         --tiers-only     Tune arithmetic tiers, radix formatting, and parsing.\n\
         --compiled-only  Tune kernel and geometry constants that require rebuilding.\n\
         --division-only  Tune shared product policies, division geometry and dispatch.\n\
         --parallel-only=<cpu-list> Tune parallel SSA policies on that exact pinned CPU set; requires rayon.\n\
         --validate-parallel-only=<cpu-list> Validate production products at every pool width; requires MP_TUNING_START and rayon.\n\
         --toom-only      Tune only compiled Toom-8.5 kernel crossovers.\n\
         --modular-only   Tune direct product policies, then Montgomery-to-Barrett exponentiation.\n\
         --gcd-only       Tune only greatest common divisor (GCD) thresholds and crossovers.\n\
         --formatting-only Tune only formatting recursion thresholds.\n\
         --parsing-only   Tune only parsing entry and schoolbook-leaf cutoffs.\n\
         MP_TUNING_START  Complete profile used as the starting candidate for partial or full tuning.\n\
         --validate-only  Validate MP_TUNING_START for installation without searching.\n\
         --profile-for <arch>\n\
                         Render the current built-in default for a target that cannot\n\
                         run the tuner and install it as the local build override.\n\
                         This command performs no measurements.\n\
                         Known archs: x86_64, aarch64, powerpc64le, s390x, riscv64,\n\
                         x86, arm, wasm32, riscv32, avr, msp430, and the other sets\n\
                         listed in build_support/."
    );
}
