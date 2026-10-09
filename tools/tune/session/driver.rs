//! Session initialization, validation, and durable result publication.

use std::{
    env::{consts::ARCH, var, var_os},
    fs::read_to_string,
    path::{Path, PathBuf},
    process::id as process_id,
    time::{SystemTime, UNIX_EPOCH},
};

use mp_anafis::tune_api::Limb;

use crate::{
    arguments::Mode,
    crossovers::Crossovers,
    platform::Platform,
    store::{ProfileWriter, ScoreStore},
    validation::Validation,
};

use super::{TuneSession, TuningProfile};

impl TuneSession {
    /// Execute the selected search and preserve its verdict before publication.
    ///
    /// # Errors
    /// Returns an error for invalid session inputs, failed validation, partial
    /// modes, or failure to persist the report or profile. Installation requires
    /// a complete validated profile and a successfully written report.
    pub fn run(mode: &Mode) -> Result<(), String> {
        if matches!(mode, Mode::ValidateOnly) && var_os("MP_TUNING_START").is_none() {
            return Err(
                "--validate-only requires MP_TUNING_START pointing to a complete candidate profile"
                    .to_owned(),
            );
        }
        if matches!(mode, Mode::ParallelOnly(_))
            && !cfg!(all(feature = "rayon", not(target_pointer_width = "16")))
        {
            return Err("parallel tuning requires rayon and at least 32-bit pointers".to_owned());
        }
        let mut session = Self::start(mode)?;
        session.run_phases(mode);
        let mut rejection_reason = None;
        let context_valid = session.harness.check_context();
        if !context_valid {
            session.record(
                "MEASUREMENT_CONTEXT_VALIDATION",
                "rejected: source/build context or earlier worker failure",
            );
            rejection_reason = Some("measurement context or worker validation failed".to_owned());
        }
        let semantically_valid = context_valid
            && match session.profile.validate() {
                Ok(()) => true,
                Err(reason) => {
                    println!("Validation failed: candidate profile is invalid: {reason}");
                    session.record("PROFILE_SEMANTIC_VALIDATION", format!("rejected: {reason}"));
                    rejection_reason = Some(format!("candidate profile is invalid: {reason}"));
                    false
                }
            };
        let formatting_valid = session.formatting_validated;
        if !formatting_valid {
            rejection_reason = Some("formatting boundary validation did not complete".to_owned());
        }
        let complete_mode = matches!(mode, Mode::All | Mode::ValidateOnly);
        if !complete_mode {
            println!(
                "Partial tuner mode: preserving the candidate as rejected output; \
                 no local profile will be installed"
            );
            session.record(
                "PARTIAL_MODE_INSTALL",
                "rejected: complete end-to-end validation was not requested",
            );
            rejection_reason = Some("partial tuner modes cannot install a profile".to_owned());
        }
        let production_valid = if semantically_valid && formatting_valid && complete_mode {
            let passed = Validation::end_to_end(&mut session);
            if !passed {
                rejection_reason = Some("end-to-end production validation failed".to_owned());
            }
            passed
        } else {
            false
        };
        let mut validated = semantically_valid && formatting_valid && production_valid;
        let round_trip_failure = match TuningProfile::from_source(
            &session.profile.render("// tuner candidate round-trip"),
        ) {
            Ok(parsed) if parsed == session.profile => None,
            Ok(_) => Some("rendered tuner profile changed during parser round-trip".to_owned()),
            Err(reason) => Some(format!("rendered tuner profile did not parse: {reason}")),
        };
        if let Some(reason) = round_trip_failure {
            validated = false;
            session.record("PROFILE_ROUND_TRIP", format!("rejected: {reason}"));
            rejection_reason = Some(reason);
        }
        session.harness.store.save();
        let run_directory = session.write_report()?;
        println!("\nTuning finished");
        if validated {
            ProfileWriter::write_profile(&session.profile, &session.cpu, &session.date)
                .map_err(|error| format!("cannot install tuning profile: {error}"))
        } else {
            ProfileWriter::write_rejected_profile(
                &session.profile,
                &run_directory,
                &session.cpu,
                &session.date,
            )
            .map_err(|error| format!("cannot preserve rejected tuning profile: {error}"))?;
            Err(rejection_reason.unwrap_or_else(|| "tuning profile was rejected".to_owned()))
        }
    }

    /// Persist the parallel verdict and candidate without installing a profile.
    ///
    /// # Errors
    /// Returns an error for invalid context, failed or inconclusive parallel
    /// validation, or failure to preserve the report and candidate.
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    pub fn run_parallel_validation(specification: &str) -> Result<(), String> {
        if var_os("MP_TUNING_START").is_none() {
            return Err("--validate-parallel-only requires MP_TUNING_START pointing to a complete candidate profile".to_owned());
        }
        let mut session = Self::start(&Mode::ValidateParallelOnly(specification.to_owned()))?;
        session.harness.parallel_cpus = Platform::parse_cpu_list(specification)?
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let passed =
            Validation::parallel_production(&mut session) && session.harness.check_context();
        session.harness.store.save();
        let directory = session.write_report()?;
        ProfileWriter::write_source(
            &session.profile.render("// Parallel validation candidate"),
            &directory.join("candidate.rs"),
        )
        .map_err(|error| format!("cannot preserve parallel candidate: {error}"))?;
        if passed {
            println!(
                "Parallel production validation passed; report: {}",
                directory.display()
            );
            Ok(())
        } else {
            Err(format!(
                "parallel production validation failed or was inconclusive; report: {}",
                directory.display()
            ))
        }
    }

    /// Establish affinity, calibration, defaults, and the optional start profile.
    fn start(mode: &Mode) -> Result<Self, String> {
        println!("mp-anafis hardware autotuner");
        println!("Target limb width: {} bits", Limb::BITS);
        let affinity = if let Mode::ParallelOnly(specification)
        | Mode::ValidateParallelOnly(specification) = mode
        {
            Platform::parallel_cpu_affinity(specification)
        } else {
            if cfg!(feature = "rayon") && var("RAYON_NUM_THREADS").is_ok_and(|value| value != "1") {
                return Err(
                    "single-CPU tuning requires RAYON_NUM_THREADS=1 or the default pool budget"
                        .to_owned(),
                );
            }
            Platform::single_cpu_affinity()
        }
        .map_err(|reason| format!("measurement affinity validation failed: {reason}"))?;
        println!("Pinned measurement context: {}", affinity.description);
        let frequency_warning = Platform::frequency_stability_warning();
        if let Some(warning) = &frequency_warning {
            println!("WARNING: {warning}");
        }
        let calibration = Crossovers::calibrate_noise();
        let pointer_width = Limb::BITS.to_string();
        let defaults = TuningProfile::for_target(ARCH, &pointer_width);
        let cpu_model = Platform::cpu_model();
        let cpu = format!("{cpu_model} [{}]", affinity.description);
        let date = Platform::today();
        let machine_dir = ScoreStore::machine_dir(&cpu);
        let mut session = Self::new(&defaults, calibration, machine_dir, cpu, date)?;
        if let Some(path) = var_os("MP_TUNING_START") {
            let source = read_to_string(&path)
                .map_err(|error| format!("cannot read MP_TUNING_START: {error}"))?;
            session.profile = TuningProfile::from_source(&source)
                .map_err(|error| format!("invalid MP_TUNING_START profile: {error}"))?;
            session
                .profile
                .validate()
                .map_err(|error| format!("invalid MP_TUNING_START profile: {error}"))?;
            session.record("START_PROFILE", format!("{}", Path::new(&path).display()));
        }
        session.record(
            "MEASUREMENT_CALIBRATION",
            format!(
                "noise_cv_ppm={}; timing_bucket_ms={}",
                session.calibration.noise_cv_ppm, session.calibration.timing_bucket_ms
            ),
        );
        session.record(
            "FREQUENCY_HYGIENE",
            frequency_warning.map_or_else(
                || "frequency controls not flagged".to_owned(),
                |warning| format!("warning: {warning}"),
            ),
        );
        Ok(session)
    }

    /// The complete report must reach disk before a profile can be installed.
    fn write_report(&self) -> Result<PathBuf, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("cannot timestamp tuning report: {error}"))?
            .as_nanos();
        let directory =
            self.machine_dir
                .join("runs")
                .join(format!("{}-{}-{stamp}", self.date, process_id()));
        let path = directory.join("report.json");
        ScoreStore::write_report(&path, &self.cpu, &self.date, &self.profile, &self.decisions)
            .map_err(|error| {
                format!(
                    "cannot preserve tuning report at {}: {error}",
                    path.display()
                )
            })?;
        Ok(directory)
    }
}
