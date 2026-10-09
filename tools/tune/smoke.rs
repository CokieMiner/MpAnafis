//! Bounded rebuild-worker verification without profile installation.

use std::{
    env::consts::ARCH,
    path::Path,
    process::id as process_id,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    CONSUMER_SCORE_COUNT, CandidateHarness, DivisionGrid, DivisionWorker, FormattingPairSpec,
    GCD_SCORE_CASES, GcdWorker, Platform, ProbeQuality, ProductGrid, SCORE_SCALE, ScoreDomain,
    TOOM85_MUL_SCORE_CELLS, TierPairSpec, TuningProfile, Validation,
};

/// Bounded rebuild-worker verification without profile installation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SmokeCheck;

impl SmokeCheck {
    /// Exercises candidate compilation, result checks, timing, and protocol parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if CPU affinity cannot be determined, temporary directories cannot be created,
    /// or if any candidate worker probe fails.
    pub fn check_workers() -> Result<(), String> {
        let affinity = Platform::single_cpu_affinity()?;
        println!("Checking rebuild workers on {}", affinity.description);
        let profile = TuningProfile::for_target(ARCH, &usize::BITS.to_string());
        // A unique directory separates smoke measurements from tuning records.
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/tune/checks")
            .join(format!(
                "{}-{}",
                process_id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            ));
        let mut harness = CandidateHarness::new(&directory.join("scores.json"), 0)?;
        harness.product_grid = ProductGrid {
            montgomery: vec![2, 31, 32, 33, 129],
            cyclic: vec![193, 3_073],
        };
        for (family, baseline, candidate, len) in [
            ("mul", "schoolbook", "karatsuba", 32),
            ("sqr", "schoolbook", "karatsuba", 32),
            ("low", "schoolbook", "mulders", 32),
            ("low", "mulders", "full", 32),
            ("pow", "montgomery", "barrett", 8),
            ("gcd", "lehmer", "half_gcd", 64),
            ("gcd", "narrow", "wide", 32),
            ("gcd", "fused", "separate", 32),
        ] {
            let specification = TierPairSpec {
                family,
                baseline,
                candidate,
                len,
                quality: ProbeQuality::Coarse,
                iterations: 12,
            };
            let _ = harness
                .score_tier_pair(&profile, specification)
                .ok_or_else(|| format!("{family}: {baseline}/{candidate} worker failed"))?;
            println!("  {family}: {baseline}/{candidate} at {len} limbs passed");
        }
        for radix in [3, 10, 36] {
            let _ = harness
                .score_formatting_pair(
                    &profile,
                    FormattingPairSpec {
                        baseline: "schoolbook",
                        candidate: "recursive",
                        radix,
                        len: 32,
                        quality: ProbeQuality::Coarse,
                        iterations: 12,
                    },
                )
                .ok_or_else(|| format!("radix {radix} formatting worker failed"))?;
            println!("  formatting: radix {radix} at 32 limbs passed");
        }
        Self::check_catalogs(&mut harness, &profile)?;
        Self::check_profile_pairing(&mut harness, &profile)?;
        println!(
            "Worker checks passed; no profile installed. Details: {}",
            directory.display()
        );
        Ok(())
    }

    /// Check complete worker catalogs and their advertised cell counts.
    fn check_catalogs(
        harness: &mut CandidateHarness,
        profile: &TuningProfile,
    ) -> Result<(), String> {
        for (name, domain, weights) in [
            (
                "direct products and consumer guards",
                ScoreDomain::Products,
                ScoreDomain::Products.cell_weights(harness),
            ),
            (
                "GCD consumers",
                ScoreDomain::Gcd,
                GcdWorker::cell_weights(&GCD_SCORE_CASES),
            ),
            (
                "division shapes",
                ScoreDomain::ProductionDivision,
                DivisionWorker::cell_weights(&DivisionWorker::cases(&DivisionGrid::default())),
            ),
            (
                "formatting and modular exponentiation",
                ScoreDomain::Consumers,
                vec![1; CONSUMER_SCORE_COUNT],
            ),
            (
                "radix parsing",
                ScoreDomain::Parsing,
                ScoreDomain::Parsing.cell_weights(harness),
            ),
            (
                "held-out validation",
                ScoreDomain::Holdout,
                ScoreDomain::Holdout.cell_weights(harness),
            ),
        ] {
            let scores = harness
                .score(profile, domain, true)
                .ok_or_else(|| format!("{name} worker failed"))?;
            if !Validation::cells_non_regress(&scores, &scores, weights.len()) {
                return Err(format!("{name} worker returned malformed measurements"));
            }
            println!("  {name}: all size/shape cells passed");
        }
        Ok(())
    }

    /// Checks fresh baseline/candidate execution through the compiled-profile protocol.
    fn check_profile_pairing(
        harness: &mut CandidateHarness,
        profile: &TuningProfile,
    ) -> Result<(), String> {
        let mut trial = *profile;
        trial.lehmer_branchless_threshold = 16_384;
        let paired = harness
            .compare_profiles(
                profile,
                &trial,
                ScoreDomain::Toom85Mul,
                &Validation::comparison_rule(
                    &vec![1; TOOM85_MUL_SCORE_CELLS.len()],
                    SCORE_SCALE,
                    &[],
                ),
            )
            .ok_or("fresh compiled-profile pairing failed")?;
        if paired.len() != TOOM85_MUL_SCORE_CELLS.len() || paired.contains(&0) {
            return Err("paired worker returned malformed measurements".to_owned());
        }
        println!("  frozen candidate binaries: fresh ABBA/BAAB protocol passed");
        Ok(())
    }
}
