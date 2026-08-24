//! Subprocess candidate scoring with a persistent cache.
//!
//! Compile-time constants must be timed through a real rebuild: a runtime
//! branch would measure a different program. Each candidate profile is written
//! to a temporary file, injected with `MP_TUNING_PROFILE`, and scored by the
//! forced worker domain that it changes. Compatible results are cached below
//! `target/tune/<cpu-and-core>/`; the key includes worker mode, profile, source,
//! toolchain, flags, and timing calibration.

use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    measure::ProbeQuality,
    store::{ScoreStore, fnv1a, measurement_context_hash, profile_hash},
    tuning_profile::TuningProfile,
};

/// Weighted-mean scale: scores are parts per million of the baseline.
pub const SCORE_SCALE: u128 = 1_000_000;

/// One rebuild-worker tier comparison.
#[derive(Clone, Copy)]
pub struct TierPairSpec<'name> {
    pub family: &'name str,
    pub baseline: &'name str,
    pub candidate: &'name str,
    pub len: usize,
    pub quality: ProbeQuality,
    pub iterations: u32,
}

/// One rebuild-worker formatting tier comparison, requiring a radix.
#[derive(Clone, Copy)]
pub struct FormattingPairSpec<'name> {
    pub baseline: &'name str,
    pub candidate: &'name str,
    pub radix: u32,
    pub len: usize,
    pub quality: ProbeQuality,
    pub iterations: u32,
}

const SSA_FLAG: &str = "--score-ssa";
const SSA_PREFIX: &str = "MP_ANAFIS_SSA_SCORE=";
const SSA_COARSE_FLAG: &str = "--score-ssa-coarse";
const SSA_COARSE_PREFIX: &str = "MP_ANAFIS_SSA_COARSE_SCORE=";
const TOOM85_FLAG: &str = "--score-toom85";
const TOOM85_PREFIX: &str = "MP_ANAFIS_TOOM85_SCORE=";
const TOOM85_MUL_FLAG: &str = "--score-toom85-mul";
const TOOM85_MUL_PREFIX: &str = "MP_ANAFIS_TOOM85_MUL_SCORE=";
const BURNIKEL_FLAG: &str = "--score-burnikel";
const BURNIKEL_PREFIX: &str = "MP_ANAFIS_BURNIKEL_SCORE=";
const NEWTON_FLAG: &str = "--score-newton";
const NEWTON_PREFIX: &str = "MP_ANAFIS_NEWTON_SCORE=";
const PRODUCTION_DIVISION_FLAG: &str = "--score-production-division";
const PRODUCTION_DIVISION_PREFIX: &str = "MP_ANAFIS_PRODUCTION_DIVISION_SCORE=";
const PRODUCTION_FLAG: &str = "--score-production";
const PRODUCTION_PREFIX: &str = "MP_ANAFIS_PRODUCTION_SCORE=";
const INTERNAL_TUNE_FEATURES: &str = "_internal-tune";

/// Rebuild-worker score domains. Keeping the worker flag and output prefix
/// together prevents phase code from duplicating the subprocess protocol.
#[derive(Clone, Copy)]
pub enum ScoreDomain {
    Ssa,
    SsaCoarse,
    Toom85,
    Toom85Mul,
    Burnikel,
    Newton,
    ProductionDivision,
    Production,
}

/// Reusable subprocess scoring with one cache file.
pub struct CandidateHarness {
    store: ScoreStore,
    file: CandidateFile,
    context_hash: u64,
}

impl CandidateHarness {
    /// `cache_path` is the machine-stable score cache file.
    #[must_use]
    pub fn new(cache_path: &Path, timing_bucket_ms: u128) -> Self {
        let mut context_bytes = Vec::new();
        context_bytes.extend_from_slice(&measurement_context_hash().to_le_bytes());
        context_bytes.extend_from_slice(&timing_bucket_ms.to_le_bytes());
        Self {
            store: ScoreStore::load(cache_path),
            file: CandidateFile::new(),
            context_hash: fnv1a(&context_bytes),
        }
    }

    /// Score a candidate in one typed rebuild-worker domain, using the
    /// persistent machine/context cache.
    pub fn score(&mut self, profile: &TuningProfile, domain: ScoreDomain) -> Option<Vec<u128>> {
        let (flag, prefix, features) = match domain {
            ScoreDomain::Ssa => (SSA_FLAG, SSA_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::SsaCoarse => (SSA_COARSE_FLAG, SSA_COARSE_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::Toom85 => (TOOM85_FLAG, TOOM85_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::Toom85Mul => (TOOM85_MUL_FLAG, TOOM85_MUL_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::Burnikel => (BURNIKEL_FLAG, BURNIKEL_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::Newton => (NEWTON_FLAG, NEWTON_PREFIX, INTERNAL_TUNE_FEATURES),
            ScoreDomain::ProductionDivision => (
                PRODUCTION_DIVISION_FLAG,
                PRODUCTION_DIVISION_PREFIX,
                INTERNAL_TUNE_FEATURES,
            ),
            ScoreDomain::Production => (PRODUCTION_FLAG, PRODUCTION_PREFIX, INTERNAL_TUNE_FEATURES),
        };
        self.score_worker(profile, flag, prefix, features)
    }

    /// Interleaved forced-tier timings compiled with `profile` active for all
    /// recursive children.
    pub fn score_tier_pair(
        &mut self,
        profile: &TuningProfile,
        specification: TierPairSpec<'_>,
    ) -> Option<(u128, u128)> {
        let quality_name = match specification.quality {
            ProbeQuality::Coarse => "coarse",
            ProbeQuality::Precise => "precise",
        };
        let flag = format!(
            "--score-{}-pair={},{},{},{},{}",
            specification.family,
            specification.baseline,
            specification.candidate,
            specification.len,
            quality_name,
            specification.iterations,
        );
        let values = self.score_worker(
            profile,
            &flag,
            "MP_ANAFIS_TIER_PAIR=",
            INTERNAL_TUNE_FEATURES,
        )?;
        let [baseline_time, candidate_time] = values.as_slice() else {
            return None;
        };
        Some((*baseline_time, *candidate_time))
    }

    /// Interleaved forced-tier timings for formatting, explicitly including
    /// the radix in the cache key and arguments.
    pub fn score_formatting_pair(
        &mut self,
        profile: &TuningProfile,
        specification: FormattingPairSpec<'_>,
    ) -> Option<(u128, u128)> {
        let quality_name = match specification.quality {
            ProbeQuality::Coarse => "coarse",
            ProbeQuality::Precise => "precise",
        };
        let flag = format!(
            "--score-fmt-pair={},{},{},{},{},{}",
            specification.baseline,
            specification.candidate,
            specification.radix,
            specification.len,
            quality_name,
            specification.iterations,
        );
        let values = self.score_worker(
            profile,
            &flag,
            "MP_ANAFIS_FMT_PAIR=",
            INTERNAL_TUNE_FEATURES,
        )?;
        let [baseline_time, candidate_time] = values.as_slice() else {
            return None;
        };
        Some((*baseline_time, *candidate_time))
    }

    /// Persist the cache. Call once after the search finishes.
    pub fn save_cache(&self) {
        self.store.save();
    }

    fn score_worker(
        &mut self,
        profile: &TuningProfile,
        flag: &str,
        prefix: &str,
        features: &str,
    ) -> Option<Vec<u128>> {
        let mut key_bytes = Vec::new();
        key_bytes.extend_from_slice(flag.as_bytes());
        key_bytes.extend_from_slice(features.as_bytes());
        key_bytes.extend_from_slice(&self.context_hash.to_le_bytes());
        key_bytes.extend_from_slice(&profile_hash(profile).to_le_bytes());
        let key = fnv1a(&key_bytes);
        if let Some(cached) = self.store.get(key) {
            println!("  (scored by an earlier run of this profile)");
            return Some(cached.to_vec());
        }
        let measurements = run_worker(profile, flag, prefix, features, &self.file.path)?;
        self.store.insert(key, measurements.clone());
        Some(measurements)
    }
}

/// Weighted relative score of `measurements` against `baseline`, in ppm.
///
/// The per-cell ratio is scaled by `SCORE_SCALE` and averaged with the given
/// weights, so a cell contributes in proportion to its width's log rather than
/// equally. A score below `SCORE_SCALE` is an improvement; a score at
/// `SCORE_SCALE` ties the baseline.
#[must_use]
pub fn relative_score(measurements: &[u128], baseline: &[u128], weights: &[u32]) -> u128 {
    if measurements.len() != baseline.len() || measurements.len() != weights.len() {
        return u128::MAX;
    }
    let mut weighted = 0_u128;
    let mut total_weight = 0_u128;
    for ((sample, reference), &weight) in measurements.iter().zip(baseline).zip(weights) {
        if weight == 0 {
            continue;
        }
        let ratio = sample
            .saturating_mul(SCORE_SCALE)
            .div_euclid((*reference).max(1));
        weighted = weighted.saturating_add(ratio.saturating_mul(u128::from(weight)));
        total_weight = total_weight.saturating_add(u128::from(weight));
    }
    weighted.div_euclid(total_weight.max(1))
}

fn run_worker(
    profile: &TuningProfile,
    flag: &str,
    prefix: &str,
    features: &str,
    path: &Path,
) -> Option<Vec<u128>> {
    let source = profile.render("// Temporary profile generated by mp-tune");
    let mut file = File::create(path).ok()?;
    file.write_all(source.as_bytes()).ok()?;
    drop(file);
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let output = Command::new(cargo)
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "mp-tune",
            "--features",
            features,
            "--",
            flag,
        ])
        .env("MP_TUNING_PROFILE", path)
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!(
            "candidate worker failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let encoded = stdout.lines().find_map(|line| line.strip_prefix(prefix))?;
    let measurements = encoded
        .split(',')
        .map(str::parse::<u128>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!measurements.is_empty()).then_some(measurements)
}

struct CandidateFile {
    path: PathBuf,
}

impl CandidateFile {
    fn new() -> Self {
        let path = env::temp_dir().join(format!("mp-anafis-tuning-{}.rs", std::process::id()));
        Self { path }
    }
}

impl Drop for CandidateFile {
    fn drop(&mut self) {
        drop(fs::remove_file(&self.path));
    }
}
