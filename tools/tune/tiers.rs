//! Tier-crossover tuning policies for the conventional multiplication and
//! squaring towers and their transform crossovers.
//!
//! Each policy is a stack of [`Candidate`]s walked by [`tune_tower`]: the
//! reigning tier is compared against the next candidate, the first sustained
//! win becomes the new threshold, and a candidate that beats its reign before
//! the reign starts rolls the tower back and shadows that superseded tier. The
//! measured thresholds are written back to the profile and recorded in the
//! run report.
//!
//! Toom-8.5 reconstruction is settled before the multiplication and square
//! towers, and SSA geometry before the final transform crossovers. Division's
//! coupled thresholds are rebuild-tuned in [`crate::compiled`].

use core::cmp::max;

use mp_anafis::tune_api::{formatting, multiplication, squaring};

use crate::{crossovers, harness::CandidateHarness, tuning_profile::TuningProfile};

const ITERATIONS: u32 = 5_000;
const TIER_SIZES: [usize; 22] = [
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1_024,
    1_536,
];
const FORMAT_SIZES: [usize; 20] = [
    4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768,
];
// Late conventional tiers and transforms can cross well after the compact
// tower ladder ends. Keeping this extension off the early candidates bounds
// tuning time without mistaking an exhausted 1,536-limb ladder for proof that
// Toom-6, Toom-8.5, or SSA is permanently shadowed.
const LARGE_SIZES: [usize; 12] = [
    512, 768, 1_024, 1_536, 2_048, 3_072, 4_096, 6_144, 8_192, 12_288, 16_384, 32_768,
];

struct Candidate<A> {
    algo: A,
    sizes: &'static [usize],
    min_next_start: usize,
}

/// Walk one tier tower and return the measured thresholds.
///
/// A shadowed intermediate tier receives the threshold of the next measured
/// tier, so the higher tier wins the dispatch check at that exact width. A
/// shadowed tail remains `usize::MAX - 1` because it has no successor that can
/// encode the shadowing boundary.
fn tune_tower<A: Copy + core::fmt::Debug>(
    base_algo: A,
    candidates: &[Candidate<A>],
    mut crossover_fn: impl FnMut(A, A, usize, &[usize], &str, u32, u32) -> Option<usize>,
    margin_ppm: u32,
) -> Vec<usize> {
    let mut thresholds = vec![usize::MAX - 1; candidates.len()];
    // Stack of (candidate_index, algorithm, starting_threshold)
    let mut stack: Vec<(isize, A, usize)> = vec![(-1, base_algo, 8)];

    let mut i = 0;
    while let Some(cand) = candidates.get(i) {
        let (reign_idx, reign_algo, reign_start) =
            *stack.last().expect("stack must never be empty");

        let dynamic_tag = format!("{:?} -> {:?}", reign_algo, cand.algo);

        if let Some(c) = crossover_fn(
            reign_algo,
            cand.algo,
            reign_start,
            cand.sizes,
            &dynamic_tag,
            ITERATIONS,
            margin_ppm,
        ) {
            if c <= reign_start && reign_idx != -1 {
                println!(
                    "Tower rollback: {dynamic_tag} beat the reigning algorithm before it started; shadowing the previous tier."
                );
                if let Some(slot) = thresholds.get_mut(usize::try_from(reign_idx).unwrap_or(0)) {
                    *slot = usize::MAX - 1;
                }
                let _ = stack.pop();
                continue;
            }
            if let Some(slot) = thresholds.get_mut(i) {
                *slot = c;
            }
            stack.push((
                isize::try_from(i).unwrap_or(0),
                cand.algo,
                max(cand.min_next_start, c),
            ));
        } else {
            println!("No sustained crossover found. Shadowing {dynamic_tag}.");
        }
        i = i.saturating_add(1);
    }

    // Dispatch considers higher conventional tiers after lower ones and keeps
    // walking when their thresholds are equal. Therefore assigning a missing
    // intermediate tier the next real threshold makes it unreachable without
    // carrying a sentinel into an otherwise ordered tower.
    let mut next_measured = None;
    for threshold in thresholds.iter_mut().rev() {
        if *threshold == usize::MAX - 1 {
            if let Some(next) = next_measured {
                *threshold = next;
            }
        } else {
            next_measured = Some(*threshold);
        }
    }

    thresholds
}

/// Tune the conventional multiplication tower.
pub fn tune_multiplication(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    println!("\nMultiplication tiers");
    let candidates = [
        Candidate {
            algo: multiplication::Algorithm::Karatsuba,
            sizes: &TIER_SIZES,
            min_next_start: 16,
        },
        Candidate {
            algo: multiplication::Algorithm::ToomCook3,
            sizes: &TIER_SIZES,
            min_next_start: 64,
        },
        Candidate {
            algo: multiplication::Algorithm::ToomCook4,
            sizes: &TIER_SIZES,
            min_next_start: 128,
        },
        Candidate {
            algo: multiplication::Algorithm::ToomCook6,
            sizes: &LARGE_SIZES,
            min_next_start: 256,
        },
        Candidate {
            algo: multiplication::Algorithm::ToomCook85,
            sizes: &LARGE_SIZES,
            min_next_start: 512,
        },
    ];
    let res = tune_tower(
        multiplication::Algorithm::Schoolbook,
        &candidates,
        |baseline, candidate, start, sizes, tag, iterations, margin| {
            crossovers::multiplication(
                profile,
                harness,
                crossovers::Request {
                    baseline,
                    candidate,
                    start,
                    sizes,
                    tag,
                    iterations,
                },
                margin,
            )
        },
        margin_ppm,
    );
    record_thresholds(
        profile,
        &res,
        decisions,
        &[
            ("KARATSUBA_THRESHOLD", "karatsuba"),
            ("TOOM_COOK_THRESHOLD", "toom_cook_3"),
            ("TOOM_COOK_4_THRESHOLD", "toom_cook_4"),
            ("TOOM_COOK_6_THRESHOLD", "toom_cook_6"),
            ("TOOM_COOK_85_THRESHOLD", "toom_cook_85"),
        ],
    );
}

/// Tune the conventional squaring tower.
pub fn tune_squaring(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    println!("\nSquaring tiers");
    let candidates = [
        Candidate {
            algo: squaring::Algorithm::Karatsuba,
            sizes: &TIER_SIZES,
            min_next_start: 16,
        },
        Candidate {
            algo: squaring::Algorithm::ToomCook3,
            sizes: &TIER_SIZES,
            min_next_start: 64,
        },
        Candidate {
            algo: squaring::Algorithm::ToomCook4,
            sizes: &TIER_SIZES,
            min_next_start: 128,
        },
        Candidate {
            algo: squaring::Algorithm::ToomCook6,
            sizes: &LARGE_SIZES,
            min_next_start: 256,
        },
        Candidate {
            algo: squaring::Algorithm::ToomCook85,
            sizes: &LARGE_SIZES,
            min_next_start: 512,
        },
    ];
    let res = tune_tower(
        squaring::Algorithm::Schoolbook,
        &candidates,
        |baseline, candidate, start, sizes, tag, iterations, margin| {
            crossovers::squaring(
                profile,
                harness,
                crossovers::Request {
                    baseline,
                    candidate,
                    start,
                    sizes,
                    tag,
                    iterations,
                },
                margin,
            )
        },
        margin_ppm,
    );
    record_thresholds(
        profile,
        &res,
        decisions,
        &[
            ("SQR_KARATSUBA_THRESHOLD", "sqr_karatsuba"),
            ("SQR_TOOM_COOK_THRESHOLD", "sqr_toom_cook_3"),
            ("SQR_TOOM_COOK_4_THRESHOLD", "sqr_toom_cook_4"),
            ("SQR_TOOM_COOK_6_THRESHOLD", "sqr_toom_cook_6"),
            ("SQR_TOOM_COOK_85_THRESHOLD", "sqr_toom_cook_85"),
        ],
    );
}

/// Tune the Toom-8.5 to SSA crossovers, multiplication and squaring
/// separately, after the compiled constants are final.
pub fn tune_transforms(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    println!("\nTransform tiers");
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mul_start = max(512, profile.toom_cook_85.min(usize::MAX - 2));
        let ssa_mul = crossovers::multiplication(
            profile,
            harness,
            crossovers::Request {
                baseline: multiplication::Algorithm::ToomCook85,
                candidate: multiplication::Algorithm::Ssa,
                start: mul_start,
                sizes: &LARGE_SIZES,
                tag: "Toom-Cook 8.5 -> SSA multiplication",
                iterations: ITERATIONS,
            },
            margin_ppm,
        );
        let sqr_start = max(512, profile.sqr_toom_cook_85.min(usize::MAX - 2));
        let ssa_sqr = crossovers::squaring(
            profile,
            harness,
            crossovers::Request {
                baseline: squaring::Algorithm::ToomCook85,
                candidate: squaring::Algorithm::Ssa,
                start: sqr_start,
                sizes: &LARGE_SIZES,
                tag: "Toom-Cook 8.5 -> SSA square",
                iterations: ITERATIONS,
            },
            margin_ppm,
        );
        // The multiplication and squaring crossovers are separate tuning
        // questions: on the measured profile the squaring one sits 25% below
        // the multiplication one, and merging them with max() throws the
        // squaring measurement away. Each field takes its own answer.
        profile.ssa = ssa_mul.unwrap_or(usize::MAX - 1);
        profile.sqr_ssa = ssa_sqr.unwrap_or(usize::MAX - 1);
        decisions.push(("SSA_THRESHOLD".to_owned(), format!("{}", profile.ssa)));
        decisions.push((
            "SQR_SSA_THRESHOLD".to_owned(),
            format!("{}", profile.sqr_ssa),
        ));
    }
}

/// Tune the schoolbook-to-recursive formatting crossover.
pub fn tune_radix_formatting(
    profile: &mut TuningProfile,
    margin_ppm: u32,
    harness: &mut CandidateHarness,
    decisions: &mut Vec<(String, String)>,
) {
    println!("\nRadix formatting crossovers");

    let mut tune_one = |radix: u32,
                        fallback: usize,
                        inner_harness: &mut CandidateHarness|
     -> Result<usize, String> {
        let tag = format!("Schoolbook -> Recursive formatting (radix {radix})");
        let result = crossovers::formatting(
            &*profile,
            inner_harness,
            radix,
            crossovers::Request {
                baseline: formatting::Algorithm::Schoolbook,
                candidate: formatting::Algorithm::Recursive,
                start: 4,
                sizes: &FORMAT_SIZES,
                tag: &tag,
                iterations: ITERATIONS,
            },
            margin_ppm,
        )?;
        Ok(result.unwrap_or(fallback))
    };

    let mut cross_10 = tune_one(10, profile.radix_decimal_recursive, harness)
        .expect("failed to tune radix 10 formatting");
    cross_10 = validate_group(
        profile,
        harness,
        cross_10,
        margin_ppm,
        10..=10,
        profile.radix_decimal_recursive,
        &mut tune_one,
    )
    .expect("failed during decimal formatting validation");

    // Small group: radices 3..=9
    let mut small_threshold = max(
        tune_one(3, profile.radix_small_recursive, harness).expect("failed to tune radix 3"),
        tune_one(9, profile.radix_small_recursive, harness).expect("failed to tune radix 9"),
    );
    small_threshold = validate_group(
        profile,
        harness,
        small_threshold,
        margin_ppm,
        3..=9,
        profile.radix_small_recursive,
        &mut tune_one,
    )
    .expect("failed during small radix formatting validation");

    // Large group: radices 11..=36
    let mut large_threshold = max(
        tune_one(11, profile.radix_large_recursive, harness).expect("failed to tune radix 11"),
        tune_one(36, profile.radix_large_recursive, harness).expect("failed to tune radix 36"),
    );
    large_threshold = validate_group(
        profile,
        harness,
        large_threshold,
        margin_ppm,
        11..=36,
        profile.radix_large_recursive,
        &mut tune_one,
    )
    .expect("failed during large radix formatting validation");

    profile.radix_decimal_recursive = cross_10;
    profile.radix_small_recursive = small_threshold;
    profile.radix_large_recursive = large_threshold;

    decisions.push((
        "RADIX_DECIMAL_RECURSIVE_THRESHOLD".to_owned(),
        format!("{}", profile.radix_decimal_recursive),
    ));
    decisions.push((
        "RADIX_SMALL_RECURSIVE_THRESHOLD".to_owned(),
        format!("{}", profile.radix_small_recursive),
    ));
    decisions.push((
        "RADIX_LARGE_RECURSIVE_THRESHOLD".to_owned(),
        format!("{}", profile.radix_large_recursive),
    ));
}

fn validate_group(
    profile: &TuningProfile,
    harness: &mut CandidateHarness,
    mut threshold: usize,
    margin_ppm: u32,
    radices: core::ops::RangeInclusive<u32>,
    fallback: usize,
    tune_one: &mut impl FnMut(u32, usize, &mut CandidateHarness) -> Result<usize, String>,
) -> Result<usize, String> {
    loop {
        let mut failed_radix = None;
        for radix in radices.clone() {
            if radix.is_power_of_two() {
                continue;
            }
            for len in [
                threshold,
                threshold.saturating_add(4),
                threshold.saturating_add(8),
            ] {
                let spec = crate::harness::FormattingPairSpec {
                    baseline: "schoolbook",
                    candidate: "recursive",
                    radix,
                    len,
                    quality: crate::measure::ProbeQuality::Precise,
                    iterations: ITERATIONS,
                };
                let Some((baseline_time, candidate_time)) =
                    harness.score_formatting_pair(profile, spec)
                else {
                    return Err(format!("Could not validate formatting radix {radix}"));
                };
                if !crate::measure::confidently_faster_nanos(
                    candidate_time,
                    baseline_time,
                    margin_ppm,
                ) {
                    failed_radix = Some(radix);
                    break;
                }
            }
            if failed_radix.is_some() {
                break;
            }
        }
        if let Some(radix) = failed_radix {
            let new_cross = tune_one(radix, fallback, harness)?;
            if new_cross <= threshold {
                println!(
                    "Formatting validation for radix {radix} failed at {threshold}, \
                     but retuning produced no later crossover"
                );
                return Ok(usize::MAX - 1);
            }
            threshold = new_cross;
        } else {
            break;
        }
    }
    Ok(threshold)
}

/// Write tuned thresholds back to the profile and record the decision.
fn record_thresholds(
    profile: &mut TuningProfile,
    results: &[usize],
    decisions: &mut Vec<(String, String)>,
    fields: &[(&str, &str)],
) {
    for (slot, (name, field)) in results.iter().zip(fields) {
        match *field {
            "karatsuba" => profile.karatsuba = *slot,
            "toom_cook_3" => profile.toom_cook_3 = *slot,
            "toom_cook_4" => profile.toom_cook_4 = *slot,
            "toom_cook_6" => profile.toom_cook_6 = *slot,
            "toom_cook_85" => profile.toom_cook_85 = *slot,
            "sqr_karatsuba" => profile.sqr_karatsuba = *slot,
            "sqr_toom_cook_3" => profile.sqr_toom_cook_3 = *slot,
            "sqr_toom_cook_4" => profile.sqr_toom_cook_4 = *slot,
            "sqr_toom_cook_6" => profile.sqr_toom_cook_6 = *slot,
            "sqr_toom_cook_85" => profile.sqr_toom_cook_85 = *slot,
            _ => {}
        }
        decisions.push(((*name).to_owned(), slot.to_string()));
    }
}
