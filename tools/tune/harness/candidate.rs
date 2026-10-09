//! Subprocess candidate scoring with a persistent cache.
//!
//! Each profile is compiled through `MP_TUNING_PROFILE` and scored by its worker
//! domain. Cache keys include the profile, worker protocol, source, toolchain,
//! flags, and timing calibration.

use std::path::Path;

use crate::{
    measure::PairedStatistics,
    worker::{DivisionGrid, PARSING_CHUNK_SIZES, ProductGrid},
};

use super::{
    Executables, INTERNAL_TUNE_FEATURES, ProbeQuality, ScoreDomain, ScoreStore, TuningProfile,
};

/// Weighted-mean scale: scores are parts per million of the baseline.
pub const SCORE_SCALE: u128 = 1_000_000;

/// One rebuild-worker tier comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TierPairSpec<'name> {
    /// Algorithm family (`mul`, `sqr`, `pow`, or `gcd`).
    pub family: &'name str,
    /// Forced baseline algorithm name in the worker protocol.
    pub baseline: &'name str,
    /// Forced candidate algorithm name in the worker protocol.
    pub candidate: &'name str,
    /// Balanced operand width in limbs.
    pub len: usize,
    /// Sampling depth for this probe.
    pub quality: ProbeQuality,
    /// Inner repetitions requested of the worker.
    pub iterations: u32,
}

/// One rebuild-worker formatting tier comparison, requiring a radix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormattingPairSpec<'name> {
    /// Forced baseline formatting algorithm name.
    pub baseline: &'name str,
    /// Forced candidate formatting algorithm name.
    pub candidate: &'name str,
    /// Conversion radix in `3..=36`.
    pub radix: u32,
    /// Operand width in limbs.
    pub len: usize,
    /// Sampling depth for this probe.
    pub quality: ProbeQuality,
    /// Inner repetitions requested of the worker.
    pub iterations: u32,
}

/// Reusable subprocess scoring with one cache file.
#[derive(Debug)]
pub struct CandidateHarness {
    /// Persistent candidate-score cache for this machine directory.
    pub store: ScoreStore,
    /// Frozen candidate executables reused across worker probes.
    pub executables: Executables,
    /// Hash of the source and toolchain context captured at construction.
    pub source_context_hash: u64,
    /// Hash of the source context plus the timing-bucket identity.
    pub context_hash: u64,
    /// A worker or source-context failure prevents installation of this search.
    pub failed: bool,
    /// Ordinal of fresh acceptance comparisons, including rejected candidates.
    pub confirmations: u64,
    /// Frozen operand dimensions shared by every binary in a comparison.
    pub division_grid: DivisionGrid,
    /// Fixed radix-chunk widths shared by every parsing comparison binary.
    pub parsing_chunks: Vec<usize>,
    /// Frozen direct-product and consumer-guard dimensions.
    pub product_grid: ProductGrid,
    /// Indices in the frozen domain catalog. Empty selects every cell.
    pub selected_cells: Vec<usize>,
    /// Required upper ratio for precise forced-tier confirmations.
    pub acceptance_limit: u128,
    /// Canonical CPU set passed unchanged to every parallel worker.
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    pub parallel_cpus: String,
    /// Shared production product boundary widths for parallel validation.
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    pub parallel_widths: Vec<usize>,
}

impl CandidateHarness {
    /// Validate the fixed source context and preserve the session failure state.
    pub fn check_context(&mut self) -> bool {
        if ScoreStore::measurement_context_hash() != Some(self.source_context_hash) {
            self.failed = true;
            eprintln!("measurement context changed; restart tuning");
        }
        !self.failed
    }

    /// Stop selection after a failed phase and persist available screening work.
    pub fn reject(&mut self, reason: &str) {
        self.failed = true;
        eprintln!("tuning aborted: {reason}");
        self.store.save();
    }

    /// `cache_path` is the machine-stable score cache file.
    ///
    /// # Errors
    /// Returns an error if the source context cannot be read or the candidate
    /// source and artifact directory cannot be created.
    pub fn new(cache_path: &Path, timing_bucket_ms: u128) -> Result<Self, String> {
        let source_context_hash = ScoreStore::measurement_context_hash()
            .ok_or("cannot read the complete measurement source context")?;
        let directory = cache_path
            .parent()
            .ok_or("cache path has no parent directory")?;
        let executables = Executables::new(directory)
            .map_err(|error| format!("cannot create candidate artifact directory: {error}"))?;
        let mut context_bytes = Vec::new();
        context_bytes.extend_from_slice(&source_context_hash.to_le_bytes());
        context_bytes.extend_from_slice(&timing_bucket_ms.to_le_bytes());
        Ok(Self {
            store: ScoreStore::load(cache_path),
            executables,
            source_context_hash,
            context_hash: ScoreStore::fnv1a(&context_bytes),
            failed: false,
            confirmations: 0,
            division_grid: DivisionGrid::default(),
            parsing_chunks: PARSING_CHUNK_SIZES.to_vec(),
            product_grid: ProductGrid::default(),
            selected_cells: Vec::new(),
            acceptance_limit: SCORE_SCALE - 7_501,
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            parallel_cpus: String::new(),
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            parallel_widths: Vec::new(),
        })
    }

    /// Score a candidate in one typed rebuild-worker domain.
    ///
    /// `fresh` bypasses the screening cache and executes the frozen worker.
    pub fn score(
        &mut self,
        profile: &TuningProfile,
        domain: ScoreDomain,
        fresh: bool,
    ) -> Option<Vec<u128>> {
        if domain == ScoreDomain::Holdout && !fresh {
            self.reject("held-out fixtures are reserved for fresh installation validation");
            return None;
        }
        let (flag, prefix, features) = domain.protocol(self);
        self.score_worker(profile, &flag, prefix, features, fresh)
    }

    /// Interleaved forced-tier timings compiled with `profile` active for all
    /// recursive children.
    pub fn score_tier_pair(
        &mut self,
        profile: &TuningProfile,
        specification: TierPairSpec<'_>,
    ) -> Option<(u128, u128, u128)> {
        let quality_name = match specification.quality {
            ProbeQuality::Coarse => "coarse",
            ProbeQuality::Precise => "precise",
        };
        let confidence_bits = if specification.quality == ProbeQuality::Precise {
            self.confirmations = self.confirmations.checked_add(1)?;
            PairedStatistics::confidence_bits(self.confirmations, 1)
        } else {
            0
        };
        let flag = format!(
            "--score-{}-pair={},{},{},{},{},{},{}",
            specification.family,
            specification.baseline,
            specification.candidate,
            specification.len,
            quality_name,
            specification.iterations,
            confidence_bits,
            self.acceptance_limit,
        );
        let values = self.score_worker(
            profile,
            &flag,
            "MP_ANAFIS_TIER_PAIR=",
            INTERNAL_TUNE_FEATURES,
            matches!(specification.quality, ProbeQuality::Precise),
        )?;
        let [baseline_time, candidate_time, upper_ratio] = values.as_slice() else {
            return None;
        };
        Some((*baseline_time, *candidate_time, *upper_ratio))
    }

    /// Interleaved forced-tier timings for formatting, explicitly including
    /// the radix in the cache key and arguments.
    pub fn score_formatting_pair(
        &mut self,
        profile: &TuningProfile,
        specification: FormattingPairSpec<'_>,
    ) -> Option<(u128, u128, u128)> {
        let quality_name = match specification.quality {
            ProbeQuality::Coarse => "coarse",
            ProbeQuality::Precise => "precise",
        };
        let confidence_bits = if specification.quality == ProbeQuality::Precise {
            self.confirmations = self.confirmations.checked_add(1)?;
            PairedStatistics::confidence_bits(self.confirmations, 1)
        } else {
            0
        };
        let flag = format!(
            "--score-fmt-pair={},{},{},{},{},{},{},{}",
            specification.baseline,
            specification.candidate,
            specification.radix,
            specification.len,
            quality_name,
            specification.iterations,
            confidence_bits,
            self.acceptance_limit,
        );
        let values = self.score_worker(
            profile,
            &flag,
            "MP_ANAFIS_FMT_PAIR=",
            INTERNAL_TUNE_FEATURES,
            matches!(specification.quality, ProbeQuality::Precise),
        )?;
        let [baseline_time, candidate_time, upper_ratio] = values.as_slice() else {
            return None;
        };
        Some((*baseline_time, *candidate_time, *upper_ratio))
    }

    /// Weighted relative score of `measurements` against `baseline`, in ppm.
    /// Rounding upward preserves upper bounds supplied by confirmation.
    #[must_use]
    pub fn relative_score(measurements: &[u128], baseline: &[u128], weights: &[u32]) -> u128 {
        if measurements.is_empty()
            || measurements.len() != baseline.len()
            || measurements.len() != weights.len()
        {
            return u128::MAX;
        }
        let mut weighted = 0_u128;
        let mut total_weight = 0_u128;
        for ((sample, reference), &weight) in measurements.iter().zip(baseline).zip(weights) {
            if *sample == 0 || *reference == 0 {
                return u128::MAX;
            }
            if weight == 0 {
                continue;
            }
            let Some(contribution) = sample
                .checked_mul(SCORE_SCALE)
                .map(|scaled| scaled.div_ceil(*reference))
                .and_then(|ratio| ratio.checked_mul(u128::from(weight)))
            else {
                return u128::MAX;
            };
            let Some(next_weighted) = weighted.checked_add(contribution) else {
                return u128::MAX;
            };
            let Some(next_weight) = total_weight.checked_add(u128::from(weight)) else {
                return u128::MAX;
            };
            weighted = next_weighted;
            total_weight = next_weight;
        }
        if total_weight == 0 {
            u128::MAX
        } else {
            weighted.div_ceil(total_weight)
        }
    }

    fn score_worker(
        &mut self,
        profile: &TuningProfile,
        flag: &str,
        prefix: &str,
        features: &str,
        fresh: bool,
    ) -> Option<Vec<u128>> {
        if !self.check_context() {
            return None;
        }
        let mut key_bytes = Vec::new();
        key_bytes.extend_from_slice(flag.as_bytes());
        key_bytes.extend_from_slice(features.as_bytes());
        for index in &self.selected_cells {
            key_bytes.extend_from_slice(&index.to_le_bytes());
        }
        key_bytes.extend_from_slice(&self.context_hash.to_le_bytes());
        key_bytes.extend_from_slice(&ScoreStore::profile_hash(profile).to_le_bytes());
        let key = ScoreStore::fnv1a(&key_bytes);
        if let Some(cached) = (!fresh).then(|| self.store.get(key)).flatten()
            && !cached.is_empty()
            && cached.iter().all(|&value| value > 0)
        {
            println!("  (cached screening score; acceptance requires fresh paired execution)");
            return Some(cached.to_vec());
        }
        let result = self
            .executables
            .prepare(profile, features)
            .and_then(|path| {
                self.executables
                    .run(&path, flag, prefix, &self.selected_cells)
            });
        let measurements = match result {
            Ok(values) => values,
            Err(error) => {
                self.failed = true;
                eprintln!("candidate measurement failed: {error}");
                return None;
            }
        };
        if !self.check_context() {
            return None;
        }
        self.store.insert(key, measurements.clone());
        Some(measurements)
    }

    /// Compute paired score ratios of candidate against baseline.
    /// The ratio of the two sums avoids intermediate mean truncation;
    /// upward ppm rounding preserves a conservative confirmation bound.
    #[must_use]
    pub fn paired_ratios(baseline: &[Vec<u128>], candidate: &[Vec<u128>]) -> Option<Vec<u128>> {
        let ([first_a, second_a], [first_b, second_b]) = (baseline, candidate) else {
            return None;
        };
        if first_a.is_empty()
            || [second_a, first_b, second_b]
                .iter()
                .any(|sample| sample.len() != first_a.len())
        {
            return None;
        }
        first_a
            .iter()
            .zip(second_a)
            .zip(first_b.iter().zip(second_b))
            .map(|((&a1, &a2), (&b1, &b2))| {
                if [a1, a2, b1, b2].contains(&0) {
                    return None;
                }
                let denominator = a1.checked_add(a2)?;
                let numerator = b1.checked_add(b2)?.checked_mul(SCORE_SCALE)?;
                Some(numerator.div_ceil(denominator))
            })
            .collect()
    }
}
