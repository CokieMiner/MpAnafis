//! Independent multiplication and squaring transitions into SSA.

#[cfg(not(target_pointer_width = "16"))]
use mp_anafis::tune_api::{MultiplicationAlgorithm, SquaringAlgorithm};

#[cfg(not(target_pointer_width = "16"))]
use super::{Crossovers, ITERATIONS, LARGE_SIZES, Request};
use super::{TierTuner, TuneSession};

impl TierTuner {
    /// Tune both SSA entry crossovers after the compiled constants are final.
    pub fn transforms(session: &mut TuneSession) {
        println!("\nTransform tiers");
        #[cfg(not(target_pointer_width = "16"))]
        {
            let original = session.profile;
            let multiplication_result = Self::transform_multiplication(session);
            let squaring_result = Self::transform_squaring(session);
            if !session.harness.check_context() {
                return;
            }
            let (Some(ssa_mul), Some(ssa_sqr)) = (multiplication_result, squaring_result) else {
                session.profile = original;
                session.record(
                    "TRANSFORM_CONFIRMATION",
                    "no confirmed crossover; input transform policies retained",
                );
                return;
            };
            session.profile.ssa = ssa_mul;
            session.profile.sqr_ssa = ssa_sqr;
            Self::reconcile_multiplication_gates(&mut session.profile);
            Self::reconcile_squaring_gates(&mut session.profile);
            session.record("SSA_THRESHOLD", session.profile.ssa.to_string());
            session.record("SQR_SSA_THRESHOLD", session.profile.sqr_ssa.to_string());
        }
        #[cfg(target_pointer_width = "16")]
        session.record(
            "TRANSFORM_CONFIRMATION",
            "SSA is unavailable on 16-bit targets",
        );
    }

    #[cfg(not(target_pointer_width = "16"))]
    fn transform_multiplication(session: &mut TuneSession) -> Option<usize> {
        let scoring_profile = session.profile;
        loop {
            let (baseline, threshold, prev_threshold, tier_name) =
                if session.profile.toom_cook_85 < usize::MAX - 1 {
                    (
                        MultiplicationAlgorithm::ToomCook85,
                        session.profile.toom_cook_85,
                        session.profile.toom_cook_6,
                        "toom_cook_85",
                    )
                } else if session.profile.toom_cook_6 < usize::MAX - 1 {
                    (
                        MultiplicationAlgorithm::ToomCook6,
                        session.profile.toom_cook_6,
                        session.profile.toom_cook_4,
                        "toom_cook_6",
                    )
                } else if session.profile.toom_cook_4 < usize::MAX - 1 {
                    (
                        MultiplicationAlgorithm::ToomCook4,
                        session.profile.toom_cook_4,
                        session.profile.toom_cook_3,
                        "toom_cook_4",
                    )
                } else if session.profile.toom_cook_3 < usize::MAX - 1 {
                    (
                        MultiplicationAlgorithm::ToomCook3,
                        session.profile.toom_cook_3,
                        session.profile.karatsuba,
                        "toom_cook_3",
                    )
                } else {
                    (
                        MultiplicationAlgorithm::Karatsuba,
                        session.profile.karatsuba,
                        16,
                        "karatsuba",
                    )
                };
            let tag = format!("{baseline:?} -> SSA multiplication");
            let request = Request {
                baseline,
                candidate: MultiplicationAlgorithm::SsaProduction,
                start: prev_threshold.max(512),
                sizes: &LARGE_SIZES,
                tag: &tag,
                iterations: ITERATIONS,
            };
            match Crossovers::multiplication(session, &scoring_profile, request) {
                Ok(Some(cross)) if cross <= threshold && tier_name != "karatsuba" => {
                    println!(
                        "SSA multiplication supersedes {tier_name} at {cross} limbs (tier entry {threshold})"
                    );
                    if tier_name == "toom_cook_85" {
                        session.profile.toom_cook_85 = usize::MAX - 1;
                    } else if tier_name == "toom_cook_6" {
                        session.profile.toom_cook_6 = usize::MAX - 1;
                    } else if tier_name == "toom_cook_4" {
                        session.profile.toom_cook_4 = usize::MAX - 1;
                    } else {
                        // The selector has five tiers; the guard excludes Karatsuba.
                        session.profile.toom_cook_3 = usize::MAX - 1;
                    }
                    session.record(format!("{tier_name}_shadowed_by_ssa"), cross.to_string());
                }
                Ok(cross) => return cross,
                Err(error) => {
                    session.harness.reject(&error);
                    eprintln!("Error tuning SSA multiplication: {error}");
                    return None;
                }
            }
        }
    }

    #[cfg(not(target_pointer_width = "16"))]
    fn transform_squaring(session: &mut TuneSession) -> Option<usize> {
        let scoring_profile = session.profile;
        loop {
            let (baseline, threshold, prev_threshold, tier_name) =
                if session.profile.sqr_toom_cook_85 < usize::MAX - 1 {
                    (
                        SquaringAlgorithm::ToomCook85,
                        session.profile.sqr_toom_cook_85,
                        session.profile.sqr_toom_cook_6,
                        "sqr_toom_cook_85",
                    )
                } else if session.profile.sqr_toom_cook_6 < usize::MAX - 1 {
                    (
                        SquaringAlgorithm::ToomCook6,
                        session.profile.sqr_toom_cook_6,
                        session.profile.sqr_toom_cook_4,
                        "sqr_toom_cook_6",
                    )
                } else if session.profile.sqr_toom_cook_4 < usize::MAX - 1 {
                    (
                        SquaringAlgorithm::ToomCook4,
                        session.profile.sqr_toom_cook_4,
                        session.profile.sqr_toom_cook_3,
                        "sqr_toom_cook_4",
                    )
                } else if session.profile.sqr_toom_cook_3 < usize::MAX - 1 {
                    (
                        SquaringAlgorithm::ToomCook3,
                        session.profile.sqr_toom_cook_3,
                        session.profile.sqr_karatsuba,
                        "sqr_toom_cook_3",
                    )
                } else {
                    (
                        SquaringAlgorithm::Karatsuba,
                        session.profile.sqr_karatsuba,
                        16,
                        "sqr_karatsuba",
                    )
                };
            let tag = format!("{baseline:?} -> SSA square");
            let request = Request {
                baseline,
                candidate: SquaringAlgorithm::SsaProduction,
                start: prev_threshold.max(512),
                sizes: &LARGE_SIZES,
                tag: &tag,
                iterations: ITERATIONS,
            };
            match Crossovers::squaring(session, &scoring_profile, request) {
                Ok(Some(cross)) if cross <= threshold && tier_name != "sqr_karatsuba" => {
                    println!(
                        "SSA squaring supersedes {tier_name} at {cross} limbs (tier entry {threshold})"
                    );
                    if tier_name == "sqr_toom_cook_85" {
                        session.profile.sqr_toom_cook_85 = usize::MAX - 1;
                    } else if tier_name == "sqr_toom_cook_6" {
                        session.profile.sqr_toom_cook_6 = usize::MAX - 1;
                    } else if tier_name == "sqr_toom_cook_4" {
                        session.profile.sqr_toom_cook_4 = usize::MAX - 1;
                    } else {
                        // The selector has five tiers; the guard excludes Karatsuba.
                        session.profile.sqr_toom_cook_3 = usize::MAX - 1;
                    }
                    session.record(format!("{tier_name}_shadowed_by_ssa"), cross.to_string());
                }
                Ok(cross) => return cross,
                Err(error) => {
                    session.harness.reject(&error);
                    eprintln!("Error tuning SSA squaring: {error}");
                    return None;
                }
            }
        }
    }
}
