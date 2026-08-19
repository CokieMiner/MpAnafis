//! Whole-profile worker domains for compiled knobs and final validation.

use core::hint::black_box;

use mp_anafis::tune_api::{
    DivisionAlgorithm, MultiplicationAlgorithm, SquaringAlgorithm, Tuner,
    tier::state::{MultiplicationBenchState, SquaringBenchState},
};

use crate::measure::median_batch_samples;

use super::{
    DIVISION_SCORE_CELLS, HASH_A, HASH_B, MUL_SCORE_CELLS, PRODUCTION_DIV_CELLS,
    PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, SQR_SCORE_CELLS, ScoreCell, TOOM85_MUL_SCORE_CELLS,
    TOOM85_SQR_SCORE_CELLS, operand,
};

/// Sampling depth for the forced-SSA worker protocol.
#[derive(Clone, Copy)]
pub enum SsaScoreQuality {
    Precise,
    Coarse,
}

/// Division algorithm selected by a compiled-profile worker mode.
#[derive(Clone, Copy)]
pub enum DivisionScoreDomain {
    Burnikel,
    Newton,
    Production,
}

/// Print forced-SSA timings for precise scoring or candidate screening.
pub fn print_ssa_score(quality: SsaScoreQuality) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let coarse = matches!(quality, SsaScoreQuality::Coarse);
        let mut values =
            Vec::with_capacity(MUL_SCORE_CELLS.len().wrapping_add(SQR_SCORE_CELLS.len()));
        for cell in MUL_SCORE_CELLS {
            values.push(score_forced_mul_cell(
                if coarse { cell.coarse() } else { cell },
                MultiplicationAlgorithm::SsaForced,
            ));
        }
        for cell in SQR_SCORE_CELLS {
            values.push(score_forced_sqr_cell(
                if coarse { cell.coarse() } else { cell },
                SquaringAlgorithm::SsaForced,
            ));
        }
        print_encoded(
            match quality {
                SsaScoreQuality::Precise => "MP_ANAFIS_SSA_SCORE=",
                SsaScoreQuality::Coarse => "MP_ANAFIS_SSA_COARSE_SCORE=",
            },
            &values,
        );
    }
    #[cfg(target_pointer_width = "16")]
    match quality {
        SsaScoreQuality::Precise => println!("MP_ANAFIS_SSA_SCORE="),
        SsaScoreQuality::Coarse => println!("MP_ANAFIS_SSA_COARSE_SCORE="),
    }
}

/// Print direct forced-Toom-8.5 timings for its reconstruction knobs.
pub fn print_toom85_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values = Vec::with_capacity(
            TOOM85_MUL_SCORE_CELLS
                .len()
                .wrapping_add(TOOM85_SQR_SCORE_CELLS.len()),
        );
        for cell in TOOM85_MUL_SCORE_CELLS {
            values.push(score_forced_mul_cell(
                cell,
                MultiplicationAlgorithm::ToomCook85,
            ));
        }
        for cell in TOOM85_SQR_SCORE_CELLS {
            values.push(score_forced_sqr_cell(cell, SquaringAlgorithm::ToomCook85));
        }
        print_encoded("MP_ANAFIS_TOOM85_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_TOOM85_SCORE=");
}

/// Print forced Toom-8.5 multiplication timings for multiplication-only knobs.
pub fn print_toom85_mul_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let values = TOOM85_MUL_SCORE_CELLS
            .map(|cell| score_forced_mul_cell(cell, MultiplicationAlgorithm::ToomCook85));
        print_encoded("MP_ANAFIS_TOOM85_MUL_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_TOOM85_MUL_SCORE=");
}

/// Print one division worker domain using its stable protocol prefix.
pub fn print_division_score(domain: DivisionScoreDomain) {
    let (prefix, algorithm) = match domain {
        DivisionScoreDomain::Burnikel => (
            "MP_ANAFIS_BURNIKEL_SCORE=",
            DivisionAlgorithm::BurnikelZiegler,
        ),
        DivisionScoreDomain::Newton => {
            ("MP_ANAFIS_NEWTON_SCORE=", DivisionAlgorithm::NewtonRaphson)
        }
        DivisionScoreDomain::Production => (
            "MP_ANAFIS_PRODUCTION_DIVISION_SCORE=",
            DivisionAlgorithm::Production,
        ),
    };
    #[cfg(not(target_pointer_width = "16"))]
    {
        let values = DIVISION_SCORE_CELLS.map(|cell| score_division_cell(cell, algorithm));
        print_encoded(prefix, &values);
    }
    #[cfg(target_pointer_width = "16")]
    {
        let _ = algorithm;
        println!("{prefix}");
    }
}

/// Print production-dispatch timings for end-to-end validation.
pub fn print_production_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values = Vec::with_capacity(
            PRODUCTION_MUL_CELLS
                .len()
                .wrapping_add(PRODUCTION_SQR_CELLS.len())
                .wrapping_add(PRODUCTION_DIV_CELLS.len()),
        );
        for cell in PRODUCTION_MUL_CELLS {
            values.push(score_production_mul_cell(cell));
        }
        for cell in PRODUCTION_SQR_CELLS {
            values.push(score_production_sqr_cell(cell));
        }
        for cell in PRODUCTION_DIV_CELLS {
            values.push(score_division_cell(cell, DivisionAlgorithm::Production));
        }
        print_encoded("MP_ANAFIS_PRODUCTION_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_PRODUCTION_SCORE=");
}

#[cfg(not(target_pointer_width = "16"))]
fn print_encoded(prefix: &str, values: &[u128]) {
    let encoded = values
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!("{prefix}{encoded}");
}

#[cfg(not(target_pointer_width = "16"))]
fn score_forced_mul_cell(cell: ScoreCell, algorithm: MultiplicationAlgorithm) -> u128 {
    let left = operand(cell.len_a, HASH_A);
    let right = operand(cell.len_b, HASH_B);
    let mut destination = vec![0; cell.len_a.wrapping_add(cell.len_b)];
    let mut runner = Tuner::multiplication(algorithm, cell.len_a, cell.len_b);
    runner.run(&mut destination, &left, &right);
    let mut prepared = runner.prepare(&mut destination, &left, &right);
    median_batch_samples(
        || black_box(&mut prepared).run(),
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_forced_sqr_cell(cell: ScoreCell, algorithm: SquaringAlgorithm) -> u128 {
    let value = operand(cell.len_a, HASH_A);
    let mut destination = vec![0; cell.len_a.wrapping_mul(2)];
    let mut runner = Tuner::squaring(algorithm, cell.len_a);
    runner.run(&mut destination, &value);
    let mut prepared = runner.prepare(&mut destination, &value);
    median_batch_samples(
        || black_box(&mut prepared).run(),
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_production_mul_cell(cell: ScoreCell) -> u128 {
    let left = operand(cell.len_a, HASH_A);
    let right = operand(cell.len_b, HASH_B);
    let mut destination = vec![0; cell.len_a.wrapping_add(cell.len_b)];
    let mut runner = MultiplicationBenchState::default();
    runner.run(&mut destination, &left, &right);
    median_batch_samples(
        || {
            runner.run(
                black_box(&mut destination),
                black_box(&left),
                black_box(&right),
            );
        },
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_production_sqr_cell(cell: ScoreCell) -> u128 {
    let value = operand(cell.len_a, HASH_A);
    let mut destination = vec![0; cell.len_a.wrapping_mul(2)];
    let mut runner = SquaringBenchState::default();
    runner.run(&mut destination, &value);
    median_batch_samples(
        || runner.run(black_box(&mut destination), black_box(&value)),
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_division_cell(cell: ScoreCell, algorithm: DivisionAlgorithm) -> u128 {
    let numerator = operand(cell.len_a, HASH_A);
    let denominator = operand(cell.len_b, HASH_B);
    let mut reference = Tuner::division(&numerator, &denominator);
    let mut runner = Tuner::division(&numerator, &denominator);
    reference.run(DivisionAlgorithm::AlgorithmD);
    runner.run(algorithm);
    assert_eq!(
        runner.quotient_limbs(),
        reference.quotient_limbs(),
        "forced division candidate produced a different quotient"
    );
    assert_eq!(
        runner.remainder_limbs(),
        reference.remainder_limbs(),
        "forced division candidate produced a different remainder"
    );
    median_batch_samples(
        || black_box(&mut runner).run(algorithm),
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}
