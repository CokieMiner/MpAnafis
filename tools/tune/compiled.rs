//! Rebuild-based tuning for Toom, division, and SSA constants.

use crate::{
    harness::{CandidateHarness, relative_score},
    tuning_profile::{FOUR_STEP_DISABLED, TuningProfile},
    worker::{
        MUL_SCORE_CELLS, SQR_SCORE_CELLS, TOOM85_MUL_SCORE_CELLS, TOOM85_SQR_SCORE_CELLS,
        cell_weights, division_cell_weights, ring_cell_weights,
    },
};

/// Coordinate passes before the search is considered converged. Passes after
/// the first re-score every knob against the winners of the previous pass,
/// which resolves the documented coupling between base modulus, transform
/// layout, and geometry pins.
const MAX_COORDINATE_PASSES: usize = 3;

const BASE_MODULUS_BITS: &[usize] = &[8_192, 16_384, 24_576, 32_768, 49_152, 65_536];
const BNM1_BASECASE_LIMBS: &[usize] = &[16, 24, 32, 48, 64, 96];
const FACTOR3_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
const FACTOR5_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
/// The final entry exceeds every reachable transform log, which disables the
/// explicit four-step layout in favour of the recursive transform.
const FOUR_STEP_LOGS: &[usize] = &[8, 9, 10, 11, 12, 13, FOUR_STEP_DISABLED];
const TRANSPOSE_TILE_LIMBS: &[usize] = &[128, 256, 384, 512, 768, 1_024, 1_536, 2_048];
const DIRECT_SHIFT_LIMBS: &[usize] = &[0, 4, 6, 8, 9, 10, 12, 16];
const PAIRED_RECONSTRUCTION_LIMBS: &[usize] = &[64, 96, 128, 192, 256, 384, 512, 768, 1_024];
const FULL_GUARD_PRODUCT_SPLIT_LIMBS: &[usize] = &[128, 192, 256, 320, 384, 448, 512, 640];
const DIVISION_RECURSION_LIMBS: &[usize] = &[16, 24, 32, 40, 48, 64, 96];
const BURNIKEL_THRESHOLDS: &[usize] = &[8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256];
const NEWTON_THRESHOLDS: &[usize] = &[
    128, 192, 256, 384, 512, 640, 768, 1_024, 1_536, 2_048, 3_072, 4_096,
];

/// RAM-sized top-level rings at which the cost model may earn an exact override.
///
/// The table lookup is exact: a pin for one ring cannot affect its parent or
/// child ring. Zero remains a candidate at every width, so a ring whose model
/// choice is already competitive stays model-driven.
#[cfg(target_pointer_width = "64")]
const GEOMETRY_RINGS: [usize; 4] = [1 << 26, 1 << 27, 1 << 28, 1 << 29];
#[cfg(target_pointer_width = "32")]
const GEOMETRY_RINGS: [usize; 3] = [1 << 26, 1 << 27, 1 << 28];
/// Candidate exponents per ring; zero defers to the cost model.
const GEOMETRY_EXPONENTS: [u8; 7] = [0, 9, 10, 11, 12, 13, 14];

struct Knob {
    name: &'static str,
    candidates: &'static [usize],
    domain: ScoreDomain,
    requires_four_step: bool,
    get: fn(TuningProfile) -> usize,
    set: fn(&mut TuningProfile, usize),
}

#[derive(Clone, Copy)]
enum ScoreDomain {
    Ssa,
    Toom85,
    Toom85Mul,
    Burnikel,
    Newton,
}

const TOOM_KNOBS: [Knob; 2] = [
    Knob {
        name: "TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS",
        candidates: PAIRED_RECONSTRUCTION_LIMBS,
        domain: ScoreDomain::Toom85,
        requires_four_step: false,
        get: |profile| profile.toom85_paired_reconstruction_min_limbs,
        set: |profile, value| profile.toom85_paired_reconstruction_min_limbs = value,
    },
    Knob {
        name: "TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS",
        candidates: FULL_GUARD_PRODUCT_SPLIT_LIMBS,
        domain: ScoreDomain::Toom85Mul,
        requires_four_step: false,
        get: |profile| profile.toom8_full_guard_product_min_split_limbs,
        set: |profile, value| profile.toom8_full_guard_product_min_split_limbs = value,
    },
];

const SSA_KNOBS: [Knob; 7] = [
    Knob {
        name: "SSA_BASE_MODULUS_BITS",
        candidates: BASE_MODULUS_BITS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_base_modulus_bits,
        set: |profile, value| profile.ssa_base_modulus_bits = value,
    },
    Knob {
        name: "SSA_BNM1_BASECASE_LIMBS",
        candidates: BNM1_BASECASE_LIMBS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_bnm1_basecase_limbs,
        set: |profile, value| profile.ssa_bnm1_basecase_limbs = value,
    },
    Knob {
        name: "SSA_NEGACYCLIC_FACTOR3_THRESHOLD",
        candidates: FACTOR3_THRESHOLDS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_negacyclic_factor3,
        set: |profile, value| profile.ssa_negacyclic_factor3 = value,
    },
    Knob {
        name: "SSA_NEGACYCLIC_FACTOR5_THRESHOLD",
        candidates: FACTOR5_THRESHOLDS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_negacyclic_factor5,
        set: |profile, value| profile.ssa_negacyclic_factor5 = value,
    },
    Knob {
        name: "SSA_FOUR_STEP_MIN_LOG",
        candidates: FOUR_STEP_LOGS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_four_step_min_log,
        set: |profile, value| profile.ssa_four_step_min_log = value,
    },
    Knob {
        name: "SSA_TRANSPOSE_TILE_LIMBS",
        candidates: TRANSPOSE_TILE_LIMBS,
        domain: ScoreDomain::Ssa,
        requires_four_step: true,
        get: |profile| profile.ssa_transpose_tile_limbs,
        set: |profile, value| profile.ssa_transpose_tile_limbs = value,
    },
    Knob {
        name: "SSA_DIRECT_SHIFT_MAX_LIMBS",
        candidates: DIRECT_SHIFT_LIMBS,
        domain: ScoreDomain::Ssa,
        requires_four_step: false,
        get: |profile| profile.ssa_direct_shift_max_limbs,
        set: |profile, value| profile.ssa_direct_shift_max_limbs = value,
    },
];

const DIVISION_KNOBS: [Knob; 2] = [
    Knob {
        name: "BURNIKEL_ZIEGLER_BLOCK_LIMBS",
        candidates: DIVISION_RECURSION_LIMBS,
        domain: ScoreDomain::Burnikel,
        requires_four_step: false,
        get: |profile| profile.burnikel_ziegler_block,
        set: |profile, value| profile.burnikel_ziegler_block = value,
    },
    Knob {
        name: "NEWTON_RAPHSON_BASECASE_LIMBS",
        candidates: DIVISION_RECURSION_LIMBS,
        domain: ScoreDomain::Newton,
        requires_four_step: false,
        get: |profile| profile.newton_reciprocal_basecase,
        set: |profile, value| profile.newton_reciprocal_basecase = value,
    },
];

/// Tune the compiled Toom-8.5 reconstruction boundary before Toom dispatch.
pub fn tune_toom(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        for knob in &TOOM_KNOBS {
            let weights = if matches!(knob.domain, ScoreDomain::Toom85Mul) {
                cell_weights(&TOOM85_MUL_SCORE_CELLS, &[])
            } else {
                cell_weights(&TOOM85_MUL_SCORE_CELLS, &TOOM85_SQR_SCORE_CELLS)
            };
            let _accepted = tune_knob(profile, margin_ppm, harness, &weights, knob, decisions);
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = (profile, margin_ppm, harness, decisions);
}

/// Tune compiled division recursion geometry before division dispatch.
pub fn tune_division_geometry(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let weights = division_cell_weights();
        for knob in &DIVISION_KNOBS {
            let _accepted = tune_knob(profile, margin_ppm, harness, &weights, knob, decisions);
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = (profile, margin_ppm, harness, decisions);
}

/// Tune the coupled division thresholds through the production dispatcher.
///
/// `BURNIKEL_ZIEGLER_THRESHOLD` controls both outer entry into Burnikel and
/// its recursive handoff to Algorithm D. Rebuilding each candidate and timing
/// production division captures both effects. Newton is then tuned with the
/// selected Burnikel cutoff fixed. A bounded coordinate pass handles any
/// remaining interaction without placing candidate checks in the timed loop.
pub fn tune_division_dispatch(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        const MAX_DIVISION_PASSES: usize = 3;
        let weights = division_cell_weights();
        for pass in 1..=MAX_DIVISION_PASSES {
            println!("\nDivision dispatch coordinate pass {pass}/{MAX_DIVISION_PASSES}");
            let burnikel_changed = tune_division_threshold(
                profile,
                margin_ppm,
                harness,
                &weights,
                "BURNIKEL_ZIEGLER_THRESHOLD",
                BURNIKEL_THRESHOLDS,
                |candidate_profile| candidate_profile.burnikel_ziegler,
                |candidate_profile, value| candidate_profile.burnikel_ziegler = value,
                |candidate_profile, value| value < candidate_profile.newton_raphson,
                decisions,
            );
            let newton_changed = tune_division_threshold(
                profile,
                margin_ppm,
                harness,
                &weights,
                "NEWTON_RAPHSON_THRESHOLD",
                NEWTON_THRESHOLDS,
                |candidate_profile| candidate_profile.newton_raphson,
                |candidate_profile, value| candidate_profile.newton_raphson = value,
                |candidate_profile, value| value > candidate_profile.burnikel_ziegler,
                decisions,
            );
            if !burnikel_changed && !newton_changed {
                println!("Division dispatch thresholds converged on pass {pass}");
                break;
            }
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = (profile, margin_ppm, harness, decisions);
}

#[cfg(not(target_pointer_width = "16"))]
#[allow(
    clippy::too_many_arguments,
    reason = "the threshold coordinate carries its profile accessors, ordering invariant, measurement context, and report sink explicitly"
)]
fn tune_division_threshold(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    weights: &[u32],
    name: &str,
    candidates: &[usize],
    get: fn(TuningProfile) -> usize,
    set: fn(&mut TuningProfile, usize),
    valid: fn(TuningProfile, usize) -> bool,
    decisions: &mut Vec<(String, String)>,
) -> bool {
    println!("\nTuning coupled {name}");
    let Some(baseline) = harness.score_production_division(profile) else {
        println!("Could not score baseline {name}; retaining it");
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let current = get(*profile);
    let mut best_value = current;
    let mut best_score = baseline_total;
    for &candidate in candidates {
        if candidate == current || !valid(*profile, candidate) {
            continue;
        }
        let mut trial = *profile;
        set(&mut trial, candidate);
        let Some(measurements) = harness.score_production_division(&trial) else {
            println!("  {candidate}: rejected");
            continue;
        };
        let score = relative_score(&measurements, &baseline, weights);
        println!("  {candidate}: {score} ppm weighted production division");
        if score < best_score {
            best_score = score;
            best_value = candidate;
        }
    }
    if beats_by_margin(best_score, baseline_total, margin_ppm) {
        println!("Selected {name}={best_value}");
        set(profile, best_value);
        decisions.push((
            name.to_owned(),
            format!("{best_value} at {best_score} ppm vs {baseline_total}"),
        ));
        true
    } else {
        println!(
            "Retained {name}={current}; no sustained {:.2}% production-division win",
            f64::from(margin_ppm) / 10_000.0
        );
        false
    }
}

/// Tune compiled SSA kernels, layouts, and per-ring geometry pins.
///
/// Coordinate passes are confined to SSA here because these knobs genuinely
/// affect one another. Division dispatch performs its own earlier coordinate
/// pass over the two coupled thresholds.
pub fn tune_ssa(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let weights = cell_weights(&MUL_SCORE_CELLS, &SQR_SCORE_CELLS);
        for pass in 1..=MAX_COORDINATE_PASSES {
            println!("\nCompiled SSA coordinate pass {pass}/{MAX_COORDINATE_PASSES}");
            let mut changed = false;
            for knob in &SSA_KNOBS {
                if knob.requires_four_step && profile.ssa_four_step_min_log == FOUR_STEP_DISABLED {
                    println!(
                        "\nSkipping {} because the four-step layout is disabled",
                        knob.name
                    );
                    continue;
                }
                if tune_knob(profile, margin_ppm, harness, &weights, knob, decisions) {
                    changed = true;
                }
            }
            if tune_geometry_exponents(profile, margin_ppm, harness, decisions) {
                changed = true;
            }
            if !changed {
                println!("Compiled SSA parameters converged on pass {pass}");
                break;
            }
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = (profile, margin_ppm, harness, decisions);
}

#[cfg(not(target_pointer_width = "16"))]
fn tune_knob(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    weights: &[u32],
    knob: &Knob,
    decisions: &mut Vec<(String, String)>,
) -> bool {
    if matches!(knob.domain, ScoreDomain::Ssa) {
        return tune_ssa_knob(profile, knob, margin_ppm, harness, weights, decisions);
    }
    println!("\nTuning compiled {}", knob.name);
    let Some(baseline) = score_profile(harness, profile, knob.domain) else {
        println!("Could not score baseline {}; retaining it", knob.name);
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let current = (knob.get)(*profile);
    let mut best_value = current;
    let mut best_score = baseline_total;

    for &candidate in knob.candidates {
        if candidate == current {
            continue;
        }
        let mut trial = *profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = score_profile(harness, &trial, knob.domain) else {
            println!("  {candidate}: rejected");
            continue;
        };
        let score = relative_score(&measurements, &baseline, weights);
        println!("  {candidate}: {score} ppm weighted");
        if score < best_score {
            best_score = score;
            best_value = candidate;
        }
    }

    if beats_by_margin(best_score, baseline_total, margin_ppm) {
        println!("Selected {}={best_value}", knob.name);
        (knob.set)(profile, best_value);
        decisions.push((
            knob.name.to_owned(),
            format!("{best_value} at {best_score} ppm vs {baseline_total}"),
        ));
        true
    } else {
        println!(
            "Retained {}={current}; no sustained {:.2}% aggregate win",
            knob.name,
            f64::from(margin_ppm) / 10_000.0
        );
        false
    }
}

/// Screen every SSA value cheaply, then precisely score only plausible wins.
///
/// The coarse worker retains the complete size/shape ladder and reduces only
/// repeat samples. Keeping the three best values plus every value within two
/// measured noise margins of the best prevents a noisy screen from discarding
/// a close winner. The final selection still uses the original full score.
#[cfg(not(target_pointer_width = "16"))]
fn tune_ssa_knob(
    profile: &mut TuningProfile,
    knob: &Knob,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    weights: &[u32],
    decisions: &mut Vec<(String, String)>,
) -> bool {
    println!("\nTuning compiled {}", knob.name);
    let Some(coarse_baseline) = harness.score_ssa_coarse(profile) else {
        println!("Could not screen baseline {}; retaining it", knob.name);
        return false;
    };
    let current = (knob.get)(*profile);
    let mut screened = Vec::with_capacity(knob.candidates.len().saturating_sub(1));
    for &candidate in knob.candidates {
        if candidate == current {
            continue;
        }
        let mut trial = *profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = harness.score_ssa_coarse(&trial) else {
            println!("  {candidate}: screening failed");
            continue;
        };
        let score = relative_score(&measurements, &coarse_baseline, weights);
        println!("  {candidate}: {score} ppm coarse");
        screened.push((candidate, score));
    }
    let shortlist = shortlist_candidates(&mut screened, margin_ppm);
    if shortlist.is_empty() {
        return false;
    }
    println!(
        "  precise shortlist: {}",
        shortlist
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let Some(baseline) = harness.score_ssa(profile) else {
        println!(
            "Could not precisely score baseline {}; retaining it",
            knob.name
        );
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let mut best_value = current;
    let mut best_score = baseline_total;
    for candidate in shortlist {
        let mut trial = *profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = harness.score_ssa(&trial) else {
            println!("  {candidate}: precise score failed");
            continue;
        };
        let score = relative_score(&measurements, &baseline, weights);
        println!("  {candidate}: {score} ppm precise");
        if score < best_score {
            best_score = score;
            best_value = candidate;
        }
    }

    if beats_by_margin(best_score, baseline_total, margin_ppm) {
        println!("Selected {}={best_value}", knob.name);
        (knob.set)(profile, best_value);
        decisions.push((
            knob.name.to_owned(),
            format!("{best_value} at {best_score} ppm vs {baseline_total}"),
        ));
        true
    } else {
        println!(
            "Retained {}={current}; no sustained {:.2}% aggregate win",
            knob.name,
            f64::from(margin_ppm) / 10_000.0
        );
        false
    }
}

#[cfg(not(target_pointer_width = "16"))]
fn shortlist_candidates(screened: &mut [(usize, u128)], margin_ppm: u32) -> Vec<usize> {
    screened.sort_unstable_by_key(|&(_, score)| score);
    let Some(&(_, best_coarse)) = screened.first() else {
        return Vec::new();
    };
    let tolerance_ppm = margin_ppm.saturating_mul(2).min(300_000);
    let cutoff = best_coarse
        .saturating_mul(u128::from(1_000_000_u32.saturating_add(tolerance_ppm)))
        .div_euclid(1_000_000);
    screened
        .iter()
        .enumerate()
        .filter_map(|(rank, &(candidate, score))| {
            (rank < 3 || score <= cutoff).then_some(candidate)
        })
        .collect()
}

#[cfg(not(target_pointer_width = "16"))]
fn score_profile(
    harness: &mut CandidateHarness,
    profile: &TuningProfile,
    domain: ScoreDomain,
) -> Option<Vec<u128>> {
    match domain {
        ScoreDomain::Ssa => harness.score_ssa(profile),
        ScoreDomain::Toom85 => harness.score_toom85(profile),
        ScoreDomain::Toom85Mul => harness.score_toom85_mul(profile),
        ScoreDomain::Burnikel => harness.score_burnikel(profile),
        ScoreDomain::Newton => harness.score_newton(profile),
    }
}

/// Tune the per-ring geometry pins of `SSA_GEOMETRY_EXPONENTS`.
///
/// Unlike the scalar knobs this is one decision per ring, and a pin is scored
/// only on cells whose top-level ring exactly matches it. Nested plans consult
/// their own exact ring width, not the parent ring's entry. Zero defers to the
/// cost model and is always in the candidate set.
#[cfg(not(target_pointer_width = "16"))]
fn tune_geometry_exponents(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) -> bool {
    let mut changed = false;
    for ring in GEOMETRY_RINGS {
        let current = pinned_exponent(profile, ring);
        println!("\nTuning SSA_GEOMETRY_EXPONENTS at ring {ring} (current {current})");
        let local_weights = ring_cell_weights(ring);
        let Some(coarse_baseline) = harness.score_ssa_ring_coarse(profile, ring) else {
            println!("Could not screen the geometry baseline; retaining the pins");
            continue;
        };
        let mut screened = Vec::with_capacity(GEOMETRY_EXPONENTS.len().saturating_sub(1));
        for exponent in GEOMETRY_EXPONENTS {
            if exponent == current {
                continue;
            }
            let mut trial = *profile;
            set_pin(&mut trial, ring, exponent);
            let Some(measurements) = harness.score_ssa_ring_coarse(&trial, ring) else {
                continue;
            };
            let score = relative_score(&measurements, &coarse_baseline, &local_weights);
            println!("  ring {ring} exponent {exponent}: {score} ppm coarse local");
            screened.push((usize::from(exponent), score));
        }
        let shortlist = shortlist_candidates(&mut screened, margin_ppm);
        if shortlist.is_empty() {
            continue;
        }
        println!(
            "  precise shortlist: {}",
            shortlist
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let Some(baseline) = harness.score_ssa_ring(profile, ring) else {
            println!("Could not precisely score the geometry baseline; retaining the pins");
            continue;
        };
        let baseline_total = relative_score(&baseline, &baseline, &local_weights);
        let mut best_value = current;
        let mut best_score = baseline_total;
        for candidate in shortlist {
            let exponent = u8::try_from(candidate).expect("geometry exponents fit u8");
            let mut trial = *profile;
            set_pin(&mut trial, ring, exponent);
            let Some(measurements) = harness.score_ssa_ring(&trial, ring) else {
                continue;
            };
            let score = relative_score(&measurements, &baseline, &local_weights);
            println!("  ring {ring} exponent {exponent}: {score} ppm precise local");
            if score < best_score {
                best_score = score;
                best_value = exponent;
            }
        }

        if beats_by_margin(best_score, baseline_total, margin_ppm) {
            println!("Selected ring {ring} exponent {best_value}");
            set_pin(profile, ring, best_value);
            decisions.push((
                format!("SSA_GEOMETRY_EXPONENTS[{ring}]"),
                format!("{best_value} at {best_score} ppm local"),
            ));
            changed = true;
        } else {
            println!("Retained ring {ring} exponent {current}");
        }
    }
    changed
}

#[cfg(not(target_pointer_width = "16"))]
fn pinned_exponent(profile: &TuningProfile, ring: usize) -> u8 {
    profile
        .ssa_geometry_exponents
        .iter()
        .find(|(candidate_ring, _)| *candidate_ring == ring)
        .map_or(0, |(_, exponent)| *exponent)
}

#[cfg(not(target_pointer_width = "16"))]
fn set_pin(profile: &mut TuningProfile, ring: usize, exponent: u8) {
    if let Some(slot) = profile
        .ssa_geometry_exponents
        .iter_mut()
        .find(|(candidate_ring, _)| *candidate_ring == ring)
    {
        *slot = (ring, exponent);
    } else if let Some(empty) = profile
        .ssa_geometry_exponents
        .iter_mut()
        .find(|(candidate_ring, _)| *candidate_ring == 0)
    {
        *empty = (ring, exponent);
    }
}

#[cfg(not(target_pointer_width = "16"))]
fn beats_by_margin(best_score: u128, baseline_total: u128, margin_ppm: u32) -> bool {
    let factor = u128::from(1_000_000_u32.wrapping_sub(margin_ppm));
    best_score.saturating_mul(1_000_000) < baseline_total.saturating_mul(factor)
}
