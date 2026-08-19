//! Host-side automated hardware tuner for `mp-anafis`.
//!
//! Run `cargo run --bin mp-tune --release --features _internal-tune` to
//! generate a complete machine-specific profile.
//!
//! # Phase order
//!
//! 1. Compiled Toom-8.5 reconstruction, then the multiplication Toom tower
//!    ([`compiled`], [`tiers`]).
//! 2. The independent squaring Toom tower ([`tiers`]).
//! 3. Compiled division recursion geometry, then rebuild-based production
//!    division dispatch thresholds ([`compiled`]).
//! 4. Compiled SSA kernel and geometry constants, then multiplication and
//!    squaring SSA crossovers ([`compiled`], [`tiers`]).
//! 5. Radix formatting after arithmetic and transform crossovers stabilize.
//! 6. End-to-end production-dispatch validation: the production dispatcher with the
//!    tuned profile must beat the architecture defaults; otherwise the profile
//!    is rejected rather than installed ([`validate`]).
//!
//! Every crossover probe runs at two qualities (see [`measure`]): the ladder
//! scan and bisection use cheap directional probes, and only the confirmation
//! and guard cells run the full measurement, so the accepted thresholds keep
//! full precision while the search costs a fraction of the time.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "the tuner is an interactive command-line measurement tool"
)]

use std::env::{self, consts};

use mp_anafis::tune_api::Limb;

#[path = "../../build_support/tuning.rs"]
mod tuning_profile;

mod compiled;
mod crossovers;
mod harness;
mod measure;
mod output;
mod platform;
mod session;
mod store;
mod tiers;
mod validate;
mod worker;

use tuning_profile::profile_for_target;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Mode {
    All,
    TiersOnly,
    CompiledOnly,
    DivisionOnly,
    ToomOnly,
    FormattingOnly,
    ScoreSsa,
    ScoreSsaCoarse,
    ScoreToom85,
    ScoreToom85Mul,
    ScoreBurnikel,
    ScoreNewton,
    ScoreProductionDivision,
    ScoreProduction,
    ScoreMulPair(String),
    ScoreSqrPair(String),
    ScoreFmtPair(String),
    ProfileFor(String),
    Help,
}

fn main() -> Result<(), String> {
    let mode = selected_mode()?;
    match &mode {
        Mode::ScoreSsa => {
            worker::print_ssa_score(worker::SsaScoreQuality::Precise);
            Ok(())
        }
        Mode::ScoreSsaCoarse => {
            worker::print_ssa_score(worker::SsaScoreQuality::Coarse);
            Ok(())
        }
        Mode::ScoreToom85 => {
            worker::print_toom85_score();
            Ok(())
        }
        Mode::ScoreToom85Mul => {
            worker::print_toom85_mul_score();
            Ok(())
        }
        Mode::ScoreBurnikel => {
            worker::print_division_score(worker::DivisionScoreDomain::Burnikel);
            Ok(())
        }
        Mode::ScoreNewton => {
            worker::print_division_score(worker::DivisionScoreDomain::Newton);
            Ok(())
        }
        Mode::ScoreProductionDivision => {
            worker::print_division_score(worker::DivisionScoreDomain::Production);
            Ok(())
        }
        Mode::ScoreProduction => {
            worker::print_production_score();
            Ok(())
        }
        Mode::ScoreMulPair(specification) => worker::print_mul_pair_score(specification),
        Mode::ScoreSqrPair(specification) => worker::print_sqr_pair_score(specification),
        Mode::ScoreFmtPair(specification) => worker::print_fmt_pair_score(specification),
        Mode::ProfileFor(target_arch) => {
            write_target_profile(target_arch);
            Ok(())
        }
        Mode::Help => {
            print_help();
            Ok(())
        }
        Mode::All
        | Mode::TiersOnly
        | Mode::CompiledOnly
        | Mode::DivisionOnly
        | Mode::ToomOnly
        | Mode::FormattingOnly => {
            run_tuner(&mode);
            Ok(())
        }
    }
}

fn run_tuner(mode: &Mode) {
    let mut session = start_tuning_session();
    run_tuning_phases(mode, &mut session);
    let semantically_valid = match session.profile.validate() {
        Ok(()) => true,
        Err(reason) => {
            println!("Validation failed: candidate profile is invalid: {reason}");
            session.record(
                "PROFILE_SEMANTIC_VALIDATION".to_owned(),
                format!("rejected: {reason}"),
            );
            false
        }
    };
    let validated = semantically_valid
        && if matches!(mode, Mode::All | Mode::FormattingOnly) {
            validate::end_to_end(&mut session)
        } else {
            println!(
                "Partial tuner mode: preserving the candidate as rejected output; \
                 no local profile will be installed"
            );
            session.record(
                "PARTIAL_MODE_INSTALL".to_owned(),
                "rejected: complete end-to-end validation was not requested".to_owned(),
            );
            false
        };
    let rendered = session.profile.render("// tuner candidate round-trip");
    let parsed = tuning_profile::TuningProfile::from_source(&rendered)
        .expect("tuner candidate must round-trip through the profile parser");
    assert_eq!(
        parsed, session.profile,
        "tuner candidate profile round-trip changed values"
    );
    session.harness.save_cache();
    store::write_report(
        &session
            .machine_dir
            .join(format!("report-{}.json", session.date)),
        &session.cpu,
        &session.date,
        &session.profile,
        &session.decisions,
    );
    println!("\nTuning finished");
    if validated {
        output::write_profile(&session.profile, &session.cpu, &session.date);
    } else {
        output::write_rejected_profile(
            &session.profile,
            &session.machine_dir,
            &session.cpu,
            &session.date,
        );
    }
}

fn start_tuning_session() -> session::TuneSession {
    println!("mp-anafis hardware autotuner");
    println!("Target limb width: {} bits", Limb::BITS);

    let affinity = match platform::single_cpu_affinity() {
        Ok(identity) => {
            println!("Pinned measurement context: {}", identity.description());
            Some(identity)
        }
        Err(reason) => {
            println!("WARNING: {reason}");
            None
        }
    };
    let calibration = crossovers::calibrate_noise();

    let pointer_width = Limb::BITS.to_string();
    let defaults = profile_for_target(consts::ARCH, &pointer_width);
    let cpu_model = platform::cpu_model();
    let cpu = affinity.map_or_else(
        || cpu_model.clone(),
        |identity| format!("{cpu_model} [{}]", identity.description()),
    );
    let date = platform::today();
    let machine_dir = store::machine_dir(&cpu);
    let mut session = session::TuneSession::new(defaults, calibration, machine_dir, cpu, date);
    session.record(
        "MEASUREMENT_CALIBRATION",
        format!(
            "noise_cv={:.6}; timing_bucket_ms={}",
            session.calibration.noise_cv, session.calibration.timing_bucket_ms
        ),
    );
    session
}

fn run_tuning_phases(mode: &Mode, session: &mut session::TuneSession) {
    if matches!(mode, Mode::All | Mode::CompiledOnly | Mode::ToomOnly) {
        compiled::tune_toom(session);
    }
    if matches!(mode, Mode::All | Mode::TiersOnly) {
        tiers::tune_multiplication(session);
        tiers::tune_squaring(session);
    }
    if matches!(mode, Mode::All | Mode::CompiledOnly | Mode::DivisionOnly) {
        compiled::tune_division_geometry(session);
    }
    if matches!(mode, Mode::All | Mode::TiersOnly | Mode::DivisionOnly) {
        compiled::tune_division_dispatch(session);
    }
    if matches!(mode, Mode::All | Mode::CompiledOnly) {
        compiled::tune_ssa(session);
    }
    if matches!(mode, Mode::All | Mode::TiersOnly) {
        tiers::tune_transforms(session);
    }
    if matches!(mode, Mode::All | Mode::TiersOnly | Mode::FormattingOnly) {
        tiers::tune_radix_formatting(session);
    }
    if matches!(mode, Mode::All | Mode::FormattingOnly) {
        validate::report_formatting_boundaries(session);
    }
}

/// Install the conservative architecture profile for a target that cannot run
/// the tuner itself.
///
/// Embedded, wasm, and other-ISA targets have no way to execute the measured
/// phases on their own hardware through this tool, so their starting point is
/// the reasoned architecture profile in `build_support/tuning.rs` (whose
/// implementation is split under `build_support/tuning/`). This mode
/// renders that profile for `target_arch` and installs it as the local build
/// override, which is what a cross-compilation workflow wants: the build reads
/// one committed profile instead of re-deriving defaults, and the header makes
/// it explicit that nothing was measured.
fn write_target_profile(target_arch: &str) {
    let profile = profile_for_target(target_arch, conventional_pointer_width(target_arch));
    println!(
        "Installing the architecture profile for {target_arch} \
         (reasoned defaults; no measurements on this host)"
    );
    output::write_arch_profile(&profile, target_arch);
}

/// Conventional width for an architecture-only profile request.
///
/// Real builds use the target's actual pointer width. `--profile-for` receives
/// only an architecture name, so ambiguous ILP32 ABIs use their conventional
/// architecture width.
fn conventional_pointer_width(target_arch: &str) -> &'static str {
    match target_arch {
        "avr" | "msp430" => "16",
        "x86" | "arm" | "armv7" | "thumbv7neon" | "thumbv7a" | "thumbv7m" | "mips" | "mipsel"
        | "sparc" | "powerpc" | "sh" | "or1k" | "nios2" | "hexagon" | "csky" | "wasm32"
        | "riscv32" | "loongarch32" | "xtensa" | "m68k" => "32",
        _ => "64",
    }
}

fn selected_mode() -> Result<Mode, String> {
    let mut selected = Mode::All;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let candidate = if let Some(specification) = argument.strip_prefix("--score-mul-pair=") {
            Mode::ScoreMulPair(specification.to_owned())
        } else if let Some(specification) = argument.strip_prefix("--score-sqr-pair=") {
            Mode::ScoreSqrPair(specification.to_owned())
        } else if let Some(specification) = argument.strip_prefix("--score-fmt-pair=") {
            Mode::ScoreFmtPair(specification.to_owned())
        } else {
            match argument.as_str() {
                "--tiers-only" => Mode::TiersOnly,
                "--compiled-only" => Mode::CompiledOnly,
                "--division-only" => Mode::DivisionOnly,
                "--toom-only" => Mode::ToomOnly,
                "--formatting-only" => Mode::FormattingOnly,
                "--score-ssa" => Mode::ScoreSsa,
                "--score-ssa-coarse" => Mode::ScoreSsaCoarse,
                "--score-toom85" => Mode::ScoreToom85,
                "--score-toom85-mul" => Mode::ScoreToom85Mul,
                "--score-burnikel" => Mode::ScoreBurnikel,
                "--score-newton" => Mode::ScoreNewton,
                "--score-production-division" => Mode::ScoreProductionDivision,
                "--score-production" => Mode::ScoreProduction,
                "--profile-for" => {
                    let Some(target) = arguments.next() else {
                        return Err(
                            "--profile-for requires a target architecture, e.g. --profile-for avr"
                                .to_owned(),
                        );
                    };
                    Mode::ProfileFor(target)
                }
                "--help" | "-h" => Mode::Help,
                _ => {
                    return Err(format!("unknown mp-tune option: {argument}; use --help"));
                }
            }
        };
        if selected != Mode::All && selected != candidate {
            return Err("mp-tune modes are mutually exclusive".to_owned());
        }
        selected = candidate;
    }
    Ok(selected)
}

fn print_help() {
    println!(
        "Usage: mp-tune [--tiers-only | --compiled-only | --division-only | --toom-only | --profile-for <arch>]\n\
         \n\
         With no mode, tunes the complete profile in 6 phases:\n\
         1. Toom-8.5 kernels\n\
         2. Multiplication and squaring\n\
         3. Division geometry\n\
         4. Division dispatch\n\
         5. SSA and transforms\n\
         6. Radix formatting and validation\n\
         --tiers-only     Tune multiplication, square, division, transform, and formatting tiers.\n\
         --compiled-only  Tune kernel and geometry constants that require rebuilding.\n\
         --division-only  Tune only division recursion geometry and dispatch thresholds.\n\
         --toom-only      Tune only compiled Toom-8.5 kernel crossovers.\n\
         --formatting-only Tune only formatting recursion thresholds.\n\
         --profile-for <arch>\n\
                         Render the architecture profile for a target that cannot\n\
                         run the tuner (embedded, wasm, a different ISA) and install\n\
                         it as the local build override. No measurements happen.\n\
                         Known archs: x86_64, aarch64, powerpc64le, s390x, riscv64,\n\
                         x86, arm, wasm32, riscv32, avr, msp430, and the other sets\n\
                         listed in build_support/tuning.rs.\n\
         --score-ssa      Hidden worker: score one compiled profile on forced-SSA cells.\n\
         --score-ssa-coarse\n\
                         Hidden worker: reduced-sample SSA candidate screening.\n\
         --score-toom85   Hidden worker: score one profile with forced Toom-8.5.\n\
         --score-burnikel Hidden worker: score one compiled profile with forced Burnikel-Ziegler.\n\
         --score-newton   Hidden worker: score one compiled profile with forced Newton-Raphson.\n\
         --score-production-division\n\
                         Hidden worker: score production division for coupled thresholds.\n\
         --score-production\n\
                         Hidden worker: score one profile on production-dispatch cells."
    );
}
