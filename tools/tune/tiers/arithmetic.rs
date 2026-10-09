//! Multiplication, low-product and squaring towers, modular exponentiation,
//! and profile reconciliation for threshold monotonicity.

use mp_anafis::tune_api::{
    LowProductAlgorithm, ModularPowAlgorithm, MultiplicationAlgorithm, SquaringAlgorithm,
};

use super::{
    Candidate, Crossovers, ITERATIONS, KARATSUBA_SIZES, Parameter, Request, TOOM3_SIZES,
    TOOM4_SIZES, TOOM6_SIZES, TOOM85_SIZES, TowerWalker, TuneSession, TuningProfile,
};

/// Operand widths for the schoolbook-to-Mulders low-product search.
const LOW_RECURSIVE_SIZES: [usize; 9] = [24, 32, 40, 48, 64, 80, 96, 128, 160];

/// Operand widths for the Mulders-to-full low-product search.
const LOW_FULL_SIZES: [usize; 8] = [128, 160, 192, 256, 320, 384, 448, 512];

const POW_MOD_SIZES: [usize; 16] = [
    32, 48, 64, 72, 80, 88, 96, 104, 112, 128, 144, 160, 192, 224, 256, 384,
];

/// Maximum refinement passes for each arithmetic tower.
const MAX_TOWER_PASSES: usize = 8;

/// Maximum absolute limb difference between successive passes counting as convergent.
const CONVERGENCE_TOLERANCE: usize = 8;

/// Tuning drivers for multiplication, squaring, transforms, and modular exponentiation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TierTuner;

impl TierTuner {
    /// Tune the conventional multiplication tower with bounded reconciliation passes.
    pub fn multiplication(session: &mut TuneSession) {
        println!("\nMultiplication tiers");
        let candidates = [
            Candidate {
                algo: MultiplicationAlgorithm::Karatsuba,
                sizes: &KARATSUBA_SIZES,
                min_next_start: 16,
            },
            Candidate {
                algo: MultiplicationAlgorithm::ToomCook3,
                sizes: &TOOM3_SIZES,
                min_next_start: 32,
            },
            Candidate {
                algo: MultiplicationAlgorithm::ToomCook4,
                sizes: &TOOM4_SIZES,
                min_next_start: 128,
            },
            Candidate {
                algo: MultiplicationAlgorithm::ToomCook6,
                sizes: &TOOM6_SIZES,
                min_next_start: 256,
            },
            Candidate {
                algo: MultiplicationAlgorithm::ToomCook85,
                sizes: &TOOM85_SIZES,
                min_next_start: 512,
            },
        ];

        let mul_fields = &[
            Parameter::KARATSUBA_THRESHOLD,
            Parameter::TOOM_COOK_THRESHOLD,
            Parameter::TOOM_COOK_4_THRESHOLD,
            Parameter::TOOM_COOK_6_THRESHOLD,
            Parameter::TOOM_COOK_85_THRESHOLD,
        ];

        let mut previous: Vec<usize> = Vec::new();

        for pass in 1..=MAX_TOWER_PASSES {
            if pass > 1 {
                println!("\n  Fixed-point pass {pass}");
            }
            let scoring_profile = session.profile;
            let res = TowerWalker::tune_tower(
                session,
                &scoring_profile,
                MultiplicationAlgorithm::Schoolbook,
                &candidates,
                |sess, profile, baseline, candidate, start, sizes, tag, iterations| {
                    Crossovers::multiplication(
                        sess,
                        profile,
                        Request {
                            baseline,
                            candidate,
                            start,
                            sizes,
                            tag,
                            iterations,
                        },
                    )
                },
                Self::apply_multiplication_threshold,
            );
            if res.is_empty() {
                return;
            }
            if Self::converged(&previous, &res) {
                println!("  Multiplication tower converged after {pass} pass(es)");
                session.record(
                    "MUL_TOWER_CONVERGENCE",
                    format!("converged after pass {pass}"),
                );
                TowerWalker::record_thresholds(session, mul_fields);
                return;
            }
            previous = res;
        }

        println!(
            "  Multiplication tower accepted after {MAX_TOWER_PASSES} passes (tolerance {CONVERGENCE_TOLERANCE})"
        );
        session.record(
            "MUL_TOWER_CONVERGENCE",
            format!("accepted after {MAX_TOWER_PASSES} passes"),
        );
        TowerWalker::record_thresholds(session, mul_fields);
    }

    /// Measures low-product crossovers after the multiplication tower.
    ///
    /// `LOW_PRODUCT_RECURSIVE_THRESHOLD` is the first width where Mulders
    /// short multiplication beats the triangular schoolbook; it is searched
    /// first because the Mulders recursion uses it as its basecase boundary.
    /// `LOW_PRODUCT_FULL_THRESHOLD` is the first width where a full product
    /// truncated to `len` limbs beats Mulders, scored with the installed
    /// recursive threshold. A failed search retains the incoming value.
    pub fn low_product(session: &mut TuneSession) {
        type Step<'step> = (
            LowProductAlgorithm,
            LowProductAlgorithm,
            &'step [usize],
            &'step str,
            &'step str,
            fn(&mut TuningProfile) -> &mut usize,
        );
        println!("\nLow product tiers");
        let steps: [Step<'_>; 2] = [
            (
                LowProductAlgorithm::Schoolbook,
                LowProductAlgorithm::Mulders,
                &LOW_RECURSIVE_SIZES,
                "Schoolbook -> Mulders low product",
                "LOW_PRODUCT_RECURSIVE_THRESHOLD",
                |profile| &mut profile.low_product_recursive,
            ),
            (
                LowProductAlgorithm::Mulders,
                LowProductAlgorithm::Full,
                &LOW_FULL_SIZES,
                "Mulders -> Full low product",
                "LOW_PRODUCT_FULL_THRESHOLD",
                |profile| &mut profile.low_product_full,
            ),
        ];
        for (baseline, candidate, sizes, tag, name, field) in steps {
            let scoring_profile = session.profile;
            let request = Request {
                baseline,
                candidate,
                start: sizes.first().copied().unwrap_or(0),
                sizes,
                tag,
                iterations: ITERATIONS,
            };
            match Crossovers::low_product(session, &scoring_profile, request) {
                Ok(Some(crossover)) => {
                    *field(&mut session.profile) = crossover;
                    session.record(name, crossover.to_string());
                }
                Ok(None) => {
                    let retained = *field(&mut session.profile);
                    session.record(name, format!("retained {retained}"));
                }
                Err(error) => {
                    session.harness.reject(&error);
                    println!("Low product tuning failed: {error}");
                    session.record(name, format!("failed: {error}"));
                    return;
                }
            }
        }
    }

    /// Tune the conventional squaring tower with bounded reconciliation passes.
    ///
    /// See [`Self::multiplication`] for the convergence protocol.
    pub fn squaring(session: &mut TuneSession) {
        println!("\nSquaring tiers");
        let candidates = [
            Candidate {
                algo: SquaringAlgorithm::Karatsuba,
                sizes: &KARATSUBA_SIZES,
                min_next_start: 16,
            },
            Candidate {
                algo: SquaringAlgorithm::ToomCook3,
                sizes: &TOOM3_SIZES,
                min_next_start: 32,
            },
            Candidate {
                algo: SquaringAlgorithm::ToomCook4,
                sizes: &TOOM4_SIZES,
                min_next_start: 128,
            },
            Candidate {
                algo: SquaringAlgorithm::ToomCook6,
                sizes: &TOOM6_SIZES,
                min_next_start: 256,
            },
            Candidate {
                algo: SquaringAlgorithm::ToomCook85,
                sizes: &TOOM85_SIZES,
                min_next_start: 512,
            },
        ];

        let sqr_fields = &[
            Parameter::SQR_KARATSUBA_THRESHOLD,
            Parameter::SQR_TOOM_COOK_THRESHOLD,
            Parameter::SQR_TOOM_COOK_4_THRESHOLD,
            Parameter::SQR_TOOM_COOK_6_THRESHOLD,
            Parameter::SQR_TOOM_COOK_85_THRESHOLD,
        ];

        let mut previous: Vec<usize> = Vec::new();

        for pass in 1..=MAX_TOWER_PASSES {
            if pass > 1 {
                println!("\n  Fixed-point pass {pass}");
            }
            let scoring_profile = session.profile;
            let res = TowerWalker::tune_tower(
                session,
                &scoring_profile,
                SquaringAlgorithm::Schoolbook,
                &candidates,
                |sess, profile, baseline, candidate, start, sizes, tag, iterations| {
                    Crossovers::squaring(
                        sess,
                        profile,
                        Request {
                            baseline,
                            candidate,
                            start,
                            sizes,
                            tag,
                            iterations,
                        },
                    )
                },
                Self::apply_squaring_threshold,
            );
            if res.is_empty() {
                return;
            }
            if Self::converged(&previous, &res) {
                println!("  Squaring tower converged after {pass} pass(es)");
                session.record(
                    "SQR_TOWER_CONVERGENCE",
                    format!("converged after pass {pass}"),
                );
                TowerWalker::record_thresholds(session, sqr_fields);
                return;
            }
            previous = res;
        }

        println!(
            "  Squaring tower accepted after {MAX_TOWER_PASSES} passes (tolerance {CONVERGENCE_TOLERANCE})"
        );
        session.record(
            "SQR_TOWER_CONVERGENCE",
            format!("accepted after {MAX_TOWER_PASSES} passes"),
        );
        TowerWalker::record_thresholds(session, sqr_fields);
    }

    /// Tune the Montgomery-to-Barrett modular exponentiation transition.
    pub fn modular_pow(session: &mut TuneSession) {
        println!("\nModular exponentiation transition (Montgomery -> Barrett)");
        let request = Request {
            baseline: ModularPowAlgorithm::Montgomery,
            candidate: ModularPowAlgorithm::Barrett,
            start: 32,
            sizes: &POW_MOD_SIZES,
            tag: "Montgomery -> Barrett pow_mod",
            iterations: 40,
        };
        match Crossovers::modular_pow(session, request) {
            Ok(Some(crossover)) => {
                session.profile.montgomery_pow_mod = crossover;
                session.record("MONTGOMERY_POW_MOD_THRESHOLD", crossover.to_string());
            }
            Ok(None) => {
                session.record(
                    "MONTGOMERY_POW_MOD_THRESHOLD",
                    format!("retained default {}", session.defaults.montgomery_pow_mod),
                );
            }
            Err(error) => {
                session.harness.reject(&error);
                println!("Modular pow tuning failed: {error}");
                session.record(
                    "MONTGOMERY_POW_MOD_THRESHOLD",
                    format!(
                        "failed: {error}; retained default {}",
                        session.defaults.montgomery_pow_mod
                    ),
                );
            }
        }
    }
    /// Two threshold vectors have converged when they have the same length and
    /// every element differs by at most [`CONVERGENCE_TOLERANCE`].
    fn converged(previous: &[usize], current: &[usize]) -> bool {
        previous.len() == current.len()
            && !previous.is_empty()
            && previous
                .iter()
                .zip(current)
                .all(|(&prev, &cur)| prev.abs_diff(cur) <= CONVERGENCE_TOLERANCE)
    }

    /// Install a discovered multiplication threshold and enforce monotonicity.
    pub fn apply_multiplication_threshold(session: &mut TuneSession, index: usize, value: usize) {
        match index {
            0 => session.profile.karatsuba = value,
            1 => session.profile.toom_cook_3 = value,
            2 => session.profile.toom_cook_4 = value,
            3 => session.profile.toom_cook_6 = value,
            4 => session.profile.toom_cook_85 = value,
            _ => {}
        }
        Self::reconcile_multiplication_gates(&mut session.profile);
    }

    /// Install a discovered squaring threshold and enforce monotonicity.
    pub fn apply_squaring_threshold(session: &mut TuneSession, index: usize, value: usize) {
        match index {
            0 => session.profile.sqr_karatsuba = value,
            1 => session.profile.sqr_toom_cook_3 = value,
            2 => session.profile.sqr_toom_cook_4 = value,
            3 => session.profile.sqr_toom_cook_6 = value,
            4 => session.profile.sqr_toom_cook_85 = value,
            _ => {}
        }
        Self::reconcile_squaring_gates(&mut session.profile);
    }

    /// Clamp multiplication tiers to non-decreasing order and shadow conventional
    /// tiers overtaken by SSA.
    pub fn reconcile_multiplication_gates(profile: &mut TuningProfile) {
        profile.toom_cook_3 = profile.toom_cook_3.max(profile.karatsuba);
        profile.toom_cook_4 = profile.toom_cook_4.max(profile.toom_cook_3);
        profile.toom_cook_6 = profile.toom_cook_6.max(profile.toom_cook_4);
        profile.toom_cook_85 = profile.toom_cook_85.max(profile.toom_cook_6);
        if profile.balanced_toom8 != 0 && profile.balanced_toom8 <= profile.karatsuba {
            profile.balanced_toom8 = 0;
        }
        if profile.ssa != 0 {
            for slot in [
                &mut profile.toom_cook_85,
                &mut profile.toom_cook_6,
                &mut profile.toom_cook_4,
                &mut profile.toom_cook_3,
            ] {
                if *slot >= profile.ssa {
                    *slot = usize::MAX - 1;
                }
            }
        }
    }

    /// Clamp squaring tiers to non-decreasing order and shadow conventional
    /// tiers overtaken by SSA.
    pub fn reconcile_squaring_gates(profile: &mut TuningProfile) {
        profile.sqr_toom_cook_3 = profile.sqr_toom_cook_3.max(profile.sqr_karatsuba);
        profile.sqr_toom_cook_4 = profile.sqr_toom_cook_4.max(profile.sqr_toom_cook_3);
        profile.sqr_toom_cook_6 = profile.sqr_toom_cook_6.max(profile.sqr_toom_cook_4);
        profile.sqr_toom_cook_85 = profile.sqr_toom_cook_85.max(profile.sqr_toom_cook_6);
        if profile.sqr_ssa != 0 {
            for slot in [
                &mut profile.sqr_toom_cook_85,
                &mut profile.sqr_toom_cook_6,
                &mut profile.sqr_toom_cook_4,
                &mut profile.sqr_toom_cook_3,
            ] {
                if *slot >= profile.sqr_ssa {
                    *slot = usize::MAX - 1;
                }
            }
        }
    }
}
