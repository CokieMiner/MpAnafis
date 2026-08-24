//! Rebuild-based tuning for Toom, division, and SSA constants.

use crate::{
    harness::{ScoreDomain, relative_score},
    session::TuneSession,
    tuning_profile::TuningProfile,
    worker::{
        MUL_SCORE_CELLS, SQR_SCORE_CELLS, TOOM85_MUL_SCORE_CELLS, TOOM85_SQR_SCORE_CELLS,
        cell_weights, division_cell_weights,
    },
};

/// Coordinate passes before the search is considered converged. Passes after
/// the first re-score every knob against the winners of the previous pass,
/// which resolves the documented coupling between base modulus and transform
/// parameters.
const MAX_COORDINATE_PASSES: usize = 3;

const BASE_MODULUS_BITS: &[usize] = &[8_192, 16_384, 24_576, 32_768, 49_152, 65_536];
const BNM1_BASECASE_LIMBS: &[usize] = &[16, 24, 32, 48, 64, 96];
const FACTOR3_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
const FACTOR5_THRESHOLDS: &[usize] = &[16, 24, 32, 40, 48, 64, 96, 128, 1_048_576];
const DIRECT_SHIFT_LIMBS: &[usize] = &[0, 4, 6, 8, 9, 10, 12, 16];
const PAIRED_RECONSTRUCTION_LIMBS: &[usize] = &[64, 96, 128, 192, 256, 384, 512, 768, 1_024];
const FULL_GUARD_PRODUCT_SPLIT_LIMBS: &[usize] = &[128, 192, 256, 320, 384, 448, 512, 640];
const DIVISION_RECURSION_LIMBS: &[usize] = &[16, 24, 32, 40, 48, 64, 96];
const BURNIKEL_THRESHOLDS: &[usize] = &[8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256];
const NEWTON_THRESHOLDS: &[usize] = &[
    128, 192, 256, 384, 512, 640, 768, 1_024, 1_536, 2_048, 3_072, 4_096,
];

struct Knob {
    name: &'static str,
    candidates: &'static [usize],
    domain: ScoreDomain,
    get: fn(TuningProfile) -> usize,
    set: fn(&mut TuningProfile, usize),
}

const TOOM_KNOBS: [Knob; 2] = [
    Knob {
        name: "TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS",
        candidates: PAIRED_RECONSTRUCTION_LIMBS,
        domain: ScoreDomain::Toom85,
        get: |profile| profile.toom85_paired_reconstruction_min_limbs,
        set: |profile, value| profile.toom85_paired_reconstruction_min_limbs = value,
    },
    Knob {
        name: "TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS",
        candidates: FULL_GUARD_PRODUCT_SPLIT_LIMBS,
        domain: ScoreDomain::Toom85Mul,
        get: |profile| profile.toom8_full_guard_product_min_split_limbs,
        set: |profile, value| profile.toom8_full_guard_product_min_split_limbs = value,
    },
];

const SSA_KNOBS: [Knob; 5] = [
    Knob {
        name: "SSA_BASE_MODULUS_BITS",
        candidates: BASE_MODULUS_BITS,
        domain: ScoreDomain::Ssa,
        get: |profile| profile.ssa_base_modulus_bits,
        set: |profile, value| profile.ssa_base_modulus_bits = value,
    },
    Knob {
        name: "SSA_BNM1_BASECASE_LIMBS",
        candidates: BNM1_BASECASE_LIMBS,
        domain: ScoreDomain::Ssa,
        get: |profile| profile.ssa_bnm1_basecase_limbs,
        set: |profile, value| profile.ssa_bnm1_basecase_limbs = value,
    },
    Knob {
        name: "SSA_NEGACYCLIC_FACTOR3_THRESHOLD",
        candidates: FACTOR3_THRESHOLDS,
        domain: ScoreDomain::Ssa,
        get: |profile| profile.ssa_negacyclic_factor3,
        set: |profile, value| profile.ssa_negacyclic_factor3 = value,
    },
    Knob {
        name: "SSA_NEGACYCLIC_FACTOR5_THRESHOLD",
        candidates: FACTOR5_THRESHOLDS,
        domain: ScoreDomain::Ssa,
        get: |profile| profile.ssa_negacyclic_factor5,
        set: |profile, value| profile.ssa_negacyclic_factor5 = value,
    },
    Knob {
        name: "SSA_DIRECT_SHIFT_MAX_LIMBS",
        candidates: DIRECT_SHIFT_LIMBS,
        domain: ScoreDomain::Ssa,
        get: |profile| profile.ssa_direct_shift_max_limbs,
        set: |profile, value| profile.ssa_direct_shift_max_limbs = value,
    },
];

const DIVISION_KNOBS: [Knob; 2] = [
    Knob {
        name: "BURNIKEL_ZIEGLER_BLOCK_LIMBS",
        candidates: DIVISION_RECURSION_LIMBS,
        domain: ScoreDomain::Burnikel,
        get: |profile| profile.burnikel_ziegler_block,
        set: |profile, value| profile.burnikel_ziegler_block = value,
    },
    Knob {
        name: "NEWTON_RAPHSON_BASECASE_LIMBS",
        candidates: DIVISION_RECURSION_LIMBS,
        domain: ScoreDomain::Newton,
        get: |profile| profile.newton_reciprocal_basecase,
        set: |profile, value| profile.newton_reciprocal_basecase = value,
    },
];

/// Tune the compiled Toom-8.5 reconstruction boundary before Toom dispatch.
pub fn tune_toom(session: &mut TuneSession) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        for knob in &TOOM_KNOBS {
            let weights = if matches!(knob.domain, ScoreDomain::Toom85Mul) {
                cell_weights(&TOOM85_MUL_SCORE_CELLS, &[])
            } else {
                cell_weights(&TOOM85_MUL_SCORE_CELLS, &TOOM85_SQR_SCORE_CELLS)
            };
            let _accepted = tune_knob(session, &weights, knob);
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = session;
}

/// Tune compiled division recursion geometry before division dispatch.
pub fn tune_division_geometry(session: &mut TuneSession) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let weights = division_cell_weights();
        for knob in &DIVISION_KNOBS {
            let _accepted = tune_knob(session, &weights, knob);
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = session;
}

/// Tune the coupled division thresholds through the production dispatcher.
///
/// `BURNIKEL_ZIEGLER_THRESHOLD` controls both outer entry into Burnikel and
/// its recursive handoff to Algorithm D. Rebuilding each candidate and timing
/// production division captures both effects. Newton is then tuned with the
/// selected Burnikel cutoff fixed. A bounded coordinate pass handles any
/// remaining interaction without placing candidate checks in the timed loop.
pub fn tune_division_dispatch(session: &mut TuneSession) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        const MAX_DIVISION_PASSES: usize = 3;
        let weights = division_cell_weights();
        for pass in 1..=MAX_DIVISION_PASSES {
            println!("\nDivision dispatch coordinate pass {pass}/{MAX_DIVISION_PASSES}");
            let burnikel_changed = tune_division_threshold(
                session,
                &weights,
                "BURNIKEL_ZIEGLER_THRESHOLD",
                BURNIKEL_THRESHOLDS,
                |candidate_profile| candidate_profile.burnikel_ziegler,
                |candidate_profile, value| candidate_profile.burnikel_ziegler = value,
                |candidate_profile, value| value < candidate_profile.newton_raphson,
            );
            let newton_changed = tune_division_threshold(
                session,
                &weights,
                "NEWTON_RAPHSON_THRESHOLD",
                NEWTON_THRESHOLDS,
                |candidate_profile| candidate_profile.newton_raphson,
                |candidate_profile, value| candidate_profile.newton_raphson = value,
                |candidate_profile, value| value > candidate_profile.burnikel_ziegler,
            );
            if !burnikel_changed && !newton_changed {
                println!("Division dispatch thresholds converged on pass {pass}");
                break;
            }
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = session;
}

#[cfg(not(target_pointer_width = "16"))]
fn tune_division_threshold(
    session: &mut TuneSession,
    weights: &[u32],
    name: &str,
    candidates: &[usize],
    get: fn(TuningProfile) -> usize,
    set: fn(&mut TuningProfile, usize),
    valid: fn(TuningProfile, usize) -> bool,
) -> bool {
    println!("\nTuning coupled {name}");
    let Some(baseline) = session
        .harness
        .score(&session.profile, ScoreDomain::ProductionDivision)
    else {
        println!("Could not score baseline {name}; retaining it");
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let current = get(session.profile);
    let mut best_value = current;
    let mut best_score = baseline_total;
    for &candidate in candidates {
        if candidate == current || !valid(session.profile, candidate) {
            continue;
        }
        let mut trial = session.profile;
        set(&mut trial, candidate);
        let Some(measurements) = session
            .harness
            .score(&trial, ScoreDomain::ProductionDivision)
        else {
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
    if beats_by_margin(best_score, baseline_total, session.margin_ppm) {
        println!("Selected {name}={best_value}");
        set(&mut session.profile, best_value);
        session.record(
            name.to_owned(),
            format!("{best_value} at {best_score} ppm vs {baseline_total}"),
        );
        true
    } else {
        println!(
            "Retained {name}={current}; no sustained {:.2}% production-division win",
            f64::from(session.margin_ppm) / 10_000.0
        );
        false
    }
}

/// Tune compiled SSA kernels, layouts, and per-ring geometry pins.
///
/// Coordinate passes are confined to SSA here because these knobs genuinely
/// affect one another. Division dispatch performs its own earlier coordinate
/// pass over the two coupled thresholds.
pub fn tune_ssa(session: &mut TuneSession) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let weights = cell_weights(&MUL_SCORE_CELLS, &SQR_SCORE_CELLS);
        for pass in 1..=MAX_COORDINATE_PASSES {
            println!("\nCompiled SSA coordinate pass {pass}/{MAX_COORDINATE_PASSES}");
            let mut changed = false;
            for knob in &SSA_KNOBS {
                if tune_knob(session, &weights, knob) {
                    changed = true;
                }
            }
            if !changed {
                println!("Compiled SSA parameters converged on pass {pass}");
                break;
            }
        }
    }
    #[cfg(target_pointer_width = "16")]
    let _ = session;
}

#[cfg(not(target_pointer_width = "16"))]
fn tune_knob(session: &mut TuneSession, weights: &[u32], knob: &Knob) -> bool {
    if let Some((coarse_domain, precise_domain)) = screened_domains(knob.domain) {
        return tune_screened_knob(session, weights, knob, coarse_domain, precise_domain);
    }
    println!("\nTuning compiled {}", knob.name);
    let Some(baseline) = session.harness.score(&session.profile, knob.domain) else {
        println!("Could not score baseline {}; retaining it", knob.name);
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let current = (knob.get)(session.profile);
    let mut best_value = current;
    let mut best_score = baseline_total;

    for &candidate in knob.candidates {
        if candidate == current {
            continue;
        }
        let mut trial = session.profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = session.harness.score(&trial, knob.domain) else {
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

    if beats_by_margin(best_score, baseline_total, session.margin_ppm) {
        println!("Selected {}={best_value}", knob.name);
        (knob.set)(&mut session.profile, best_value);
        session.record(
            knob.name.to_owned(),
            format!("{best_value} at {best_score} ppm vs {baseline_total}"),
        );
        true
    } else {
        println!(
            "Retained {}={current}; no sustained {:.2}% aggregate win",
            knob.name,
            f64::from(session.margin_ppm) / 10_000.0
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
fn tune_screened_knob(
    session: &mut TuneSession,
    weights: &[u32],
    knob: &Knob,
    coarse_domain: ScoreDomain,
    precise_domain: ScoreDomain,
) -> bool {
    let current = (knob.get)(session.profile);
    println!("\nTuning {} (current {current})", knob.name);
    let mut screened = Vec::with_capacity(knob.candidates.len().saturating_sub(1));
    let Some(coarse_baseline) = session.harness.score(&session.profile, coarse_domain) else {
        println!("Could not screen {} baseline; retaining", knob.name);
        return false;
    };
    for &candidate in knob.candidates {
        if candidate == current {
            continue;
        }
        let mut trial = session.profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = session.harness.score(&trial, coarse_domain) else {
            continue;
        };
        let score = relative_score(&measurements, &coarse_baseline, weights);
        println!("  candidate {candidate}: {score} ppm coarse");
        screened.push((candidate, score));
    }
    let shortlist = shortlist_candidates(&mut screened, session.margin_ppm);
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
    let Some(baseline) = session.harness.score(&session.profile, precise_domain) else {
        println!(
            "Could not precisely score {} baseline; retaining",
            knob.name
        );
        return false;
    };
    let baseline_total = relative_score(&baseline, &baseline, weights);
    let mut best_value = current;
    let mut best_score = baseline_total;
    for candidate in shortlist {
        let mut trial = session.profile;
        (knob.set)(&mut trial, candidate);
        let Some(measurements) = session.harness.score(&trial, precise_domain) else {
            continue;
        };
        let score = relative_score(&measurements, &baseline, weights);
        println!("  candidate {candidate}: {score} ppm precise");
        if score < best_score {
            best_score = score;
            best_value = candidate;
        }
    }

    if beats_by_margin(best_score, baseline_total, session.margin_ppm) {
        println!("Selected {} = {best_value}", knob.name);
        (knob.set)(&mut session.profile, best_value);
        session.record(
            knob.name.to_owned(),
            format!("{best_value} at {best_score} ppm"),
        );
        true
    } else {
        println!("Retained {} = {current}", knob.name);
        false
    }
}

#[cfg(not(target_pointer_width = "16"))]
const fn screened_domains(domain: ScoreDomain) -> Option<(ScoreDomain, ScoreDomain)> {
    match domain {
        ScoreDomain::Ssa => Some((ScoreDomain::SsaCoarse, ScoreDomain::Ssa)),
        ScoreDomain::SsaCoarse
        | ScoreDomain::Toom85
        | ScoreDomain::Toom85Mul
        | ScoreDomain::Burnikel
        | ScoreDomain::Newton
        | ScoreDomain::ProductionDivision
        | ScoreDomain::Production => None,
    }
}

#[cfg(not(target_pointer_width = "16"))]
fn shortlist_candidates(screened: &mut [(usize, u128)], margin_ppm: u32) -> Vec<usize> {
    screened.sort_by_key(|&(_, score)| score);
    let Some(&(_, best_score)) = screened.first() else {
        return Vec::new();
    };
    let close_score = best_score.saturating_add(u128::from(margin_ppm).saturating_mul(2));
    screened
        .iter()
        .enumerate()
        .filter_map(|(rank, &(candidate, score))| {
            (rank < 3 || score <= close_score).then_some(candidate)
        })
        .collect()
}

#[cfg(not(target_pointer_width = "16"))]
fn beats_by_margin(best_score: u128, baseline_total: u128, margin_ppm: u32) -> bool {
    let factor = u128::from(1_000_000_u32.saturating_sub(margin_ppm));
    best_score.saturating_mul(1_000_000) < baseline_total.saturating_mul(factor)
}

#[cfg(test)]
mod tests {
    use super::shortlist_candidates;

    #[test]
    fn shortlist_keeps_three_candidates_when_the_coarse_screen_dislikes_all() {
        let mut screened = [
            (4, 1_300_000),
            (8, 1_100_000),
            (12, 1_200_000),
            (16, 1_400_000),
        ];

        assert_eq!(shortlist_candidates(&mut screened, 20_000), [8, 12, 4]);
    }

    #[test]
    fn shortlist_keeps_every_candidate_close_to_the_coarse_winner() {
        let mut screened = [
            (4, 1_000_000),
            (8, 1_010_000),
            (12, 1_020_000),
            (16, 1_035_000),
            (20, 1_050_000),
        ];

        assert_eq!(shortlist_candidates(&mut screened, 20_000), [4, 8, 12, 16]);
    }
}
