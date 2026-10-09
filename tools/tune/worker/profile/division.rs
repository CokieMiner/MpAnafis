//! Division operand geometry and scalar-quotient policy measurements.

use core::{hint::black_box, iter::repeat_n, mem::size_of, ops::RangeInclusive};

use mp_anafis::{
    MpUint,
    tune_api::{DivisionAlgorithm, DivisionRunner},
};

use super::{
    CellSelection, DIVISION_SCORE_CELLS, DivisionGrid, HASH_A, HASH_B, InterleavedMeasure,
    ProfileWorkers, ScoreCell,
};

/// Division algorithm selected by a compiled-profile worker mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivisionScoreDomain {
    /// Forced Burnikel–Ziegler division.
    Burnikel,
    /// Forced Newton–Raphson division.
    Newton,
    /// Production-dispatch division.
    Production,
}

/// Exact quotient values straddling scalar-policy candidates. Each cell
/// includes two leading-limb shapes and remainders zero, one, and divisor-1.
pub const SMALL_QUOTIENT_LIMBS: [usize; 4] = [3, 4, 64, 512];
pub const SMALL_QUOTIENT_VALUES: RangeInclusive<usize> = 1..=65;

/// The upper guard exceeds every finite Newton candidate. The case generator
/// adds short, balanced and long quotient spans, block-ratio boundaries and
/// candidate neighbours. Each residue, normalization and output is a separate cell.
pub const PRODUCTION_DIVISOR_LIMBS: [usize; 28] = [
    1, 2, 3, 4, 5, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1_024, 1_536, 2_048,
    3_000, 4_096, 8_192, 16_384, 32_768, 65_536, 98_304,
];

/// One operand geometry. Each normalization, residue and output has its own cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivisionCase {
    pub divisor_limbs: usize,
    pub quotient_limbs: usize,
    pub scalar_quotient: Option<usize>,
}

/// Whole-profile division worker domains and fixture generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivisionWorker;

impl DivisionWorker {
    /// Print one division worker domain using its stable protocol prefix.
    pub fn print_score(
        domain: DivisionScoreDomain,
        grid: &DivisionGrid,
        selection: &str,
    ) -> Result<(), String> {
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
        let values = if matches!(domain, DivisionScoreDomain::Production) {
            let cases = Self::cases(grid);
            let indices = CellSelection::parse(selection, Self::cell_weights(&cases).len())?;
            Self::score_production_division(0, &cases, &indices)
        } else {
            DIVISION_SCORE_CELLS
                .map(|cell| Self::score_division_cell(cell, algorithm))
                .to_vec()
        };
        ProfileWorkers::print_encoded(prefix, &values);
        Ok(())
    }

    /// Complete public quotient, remainder, and combined calls. Constructed
    /// U=qD+r fixtures avoid a quadratic large-width division oracle.
    pub fn score_production_division(
        seed: usize,
        cases: &[DivisionCase],
        selection: &[usize],
    ) -> Vec<u128> {
        let mut values = Vec::new();
        let mut offset = 0_usize;
        for case in cases {
            let first = offset;
            offset = offset
                .checked_add(if case.scalar_quotient.is_some() {
                    24
                } else {
                    16
                })
                .expect("division catalog fits");
            if !selection.is_empty()
                && !selection
                    .iter()
                    .any(|&index| (first..offset).contains(&index))
            {
                continue;
            }
            let len = case.divisor_limbs;
            let quotient_len = case.quotient_limbs;
            let fixtures = case.scalar_quotient.map_or_else(
                || Self::division_fixtures(len, quotient_len, seed),
                |value| Self::small_quotient_fixtures(len, value),
            );
            let scalar = case
                .scalar_quotient
                .map_or_else(|| "none".to_owned(), |value| value.to_string());
            for (numerator, divisor, quotient, remainder) in &fixtures {
                assert_eq!(
                    numerator.div_rem(divisor),
                    Some((quotient.clone(), remainder.clone())),
                    "constructed div_rem identity"
                );
                assert_eq!(
                    numerator.checked_div(divisor),
                    Some(quotient.clone()),
                    "constructed quotient identity"
                );
                assert_eq!(
                    numerator.checked_rem(divisor),
                    Some(remainder.clone()),
                    "constructed remainder identity"
                );
                assert_eq!(
                    numerator.is_divisible_by(divisor),
                    remainder == &MpUint::zero(),
                    "constructed divisibility identity"
                );
            }
            for (fixture, (numerator, divisor, _, _)) in fixtures.iter().enumerate() {
                for output in 0..4 {
                    let index = first
                        .checked_add(fixture.checked_mul(4).expect("six fixtures"))
                        .and_then(|index| index.checked_add(output))
                        .expect("division catalog fits");
                    if !selection.is_empty() && selection.binary_search(&index).is_err() {
                        continue;
                    }
                    println!(
                        "MP_ANAFIS_CELL division index={index} divisor_limbs={len} quotient_limbs={quotient_len} scalar_quotient={scalar} fixture={fixture} output={output} seed={seed}"
                    );
                    let elapsed = InterleavedMeasure::median_batch_samples(
                        || match output {
                            0 => {
                                drop(black_box(
                                    black_box(numerator).checked_div(black_box(divisor)),
                                ));
                            }
                            1 => {
                                drop(black_box(
                                    black_box(numerator).checked_rem(black_box(divisor)),
                                ));
                            }
                            2 => {
                                drop(black_box(black_box(numerator).div_rem(black_box(divisor))));
                            }
                            _ => {
                                let _ = black_box(
                                    black_box(numerator).is_divisible_by(black_box(divisor)),
                                );
                            }
                        },
                        1,
                        3,
                    );
                    values.push(elapsed);
                }
            }
        }
        values
    }

    /// Derive weights from exactly the same geometry and fixture order as execution.
    #[must_use]
    pub fn cell_weights(cases: &[DivisionCase]) -> Vec<u32> {
        cases
            .iter()
            .flat_map(|case| {
                let fixtures: usize = if case.scalar_quotient.is_some() { 6 } else { 4 };
                repeat_n(
                    case.divisor_limbs.ilog2().max(1),
                    fixtures
                        .checked_mul(4)
                        .expect("six fixtures and four outputs"),
                )
            })
            .collect()
    }

    /// Score the outputs and operand classes controlled by one division policy.
    /// Complete phase validation guards the omitted cells. Fixture order is
    /// unnormalized exact/inexact followed by normalized exact/inexact; scalar
    /// quotient fixtures separately contain both leading-limb shapes.
    #[must_use]
    pub fn policy_weights(
        policy: &str,
        grid: &DivisionGrid,
        maximum: usize,
        newton_minimum: usize,
    ) -> Vec<u32> {
        Self::cases(grid)
            .into_iter()
            .flat_map(|case| {
                let fixtures = if case.scalar_quotient.is_some() { 6 } else { 4 };
                let weight = case.divisor_limbs.ilog2().max(1);
                (0..fixtures).flat_map(move |fixture| {
                    (0..4).map(move |output| {
                        let active = match policy {
                            "BURNIKEL_QUOTIENT_THRESHOLD"
                            | "BURNIKEL_LONG_QUOTIENT_THRESHOLD"
                            | "NEWTON_QUOTIENT_THRESHOLD"
                            | "APPROXIMATE_DIVISION_BLOCK_LIMBS"
                            | "DIVISION_TRUNCATION_RATIO" => output == 0,
                            "DIVISION_DIVISIBLE_THRESHOLD" => output == 3,
                            "DIVISION_SINGLE_NORMALIZED_PREINVERSE" => {
                                case.divisor_limbs == 1 && fixture >= 2 && output < 3
                            }
                            "DIVISION_SINGLE_UNNORMALIZED_PREINVERSE" => {
                                case.divisor_limbs == 1 && fixture < 2 && output < 3
                            }
                            "DIVISION_SMALL_QUOTIENT_MAX" => {
                                case.scalar_quotient
                                    .is_some_and(|value| value <= maximum.saturating_add(1))
                                    && output < 3
                            }
                            "DIVISION_BASECASE_QUOTIENT_MAX_LIMBS" => {
                                case.divisor_limbs > 2
                                    && case.quotient_limbs <= maximum.saturating_add(1)
                                    && output < 3
                            }
                            "DIVISION_STACK_LIMBS" => {
                                case.divisor_limbs > 1
                                    && case.quotient_limbs <= maximum.saturating_add(1)
                            }
                            "NEWTON_RAPHSON_BASECASE_LIMBS"
                            | "NEWTON_SMALL_QUOTIENT_BLOCK_RATIO" => {
                                case.divisor_limbs >= newton_minimum
                                    && case.scalar_quotient.is_none()
                            }
                            _ => true,
                        };
                        if active { weight } else { 0 }
                    })
                })
            })
            .collect()
    }

    /// Include native-width edges, quotient ratios and block-boundary neighbours.
    /// Extra widths are identical for all binaries in one compiled comparison.
    #[must_use]
    pub fn cases(grid: &DivisionGrid) -> Vec<DivisionCase> {
        let mut widths = PRODUCTION_DIVISOR_LIMBS.to_vec();
        widths.extend_from_slice(&grid.divisor_widths);
        widths.sort_unstable();
        widths.dedup();
        let mut cases = Vec::new();
        for len in widths {
            assert!(len > 0, "positive division fixture width");
            let mut quotients = vec![
                1,
                2,
                3,
                4,
                6,
                8,
                16,
                len,
                len.checked_add(1).expect("fixture width fits"),
                len.checked_mul(3)
                    .expect("fixture width fits")
                    .div_euclid(2),
                len.checked_mul(3).expect("fixture width fits"),
            ];
            quotients.extend_from_slice(&grid.quotient_widths);
            let mut ratios = vec![1, 2, 3, 4, 6, 8];
            ratios.extend_from_slice(&grid.block_ratios);
            ratios.sort_unstable();
            ratios.dedup();
            for ratio in ratios {
                let boundary = len.div_euclid(ratio);
                quotients.extend([
                    boundary.saturating_sub(2).max(1),
                    boundary.saturating_sub(1).max(1),
                    boundary.max(1),
                    boundary.checked_add(1).expect("fixture width fits"),
                ]);
            }
            if len == 1 {
                quotients.extend([5, 7, 9, 15, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]);
            }
            quotients.sort_unstable();
            quotients.dedup();
            for quotient_limbs in quotients {
                cases.push(DivisionCase {
                    divisor_limbs: len,
                    quotient_limbs,
                    scalar_quotient: None,
                });
            }
        }
        let mut scalars: Vec<_> = SMALL_QUOTIENT_VALUES.collect();
        scalars.extend_from_slice(&grid.scalar_quotients);
        scalars.sort_unstable();
        scalars.dedup();
        for len in SMALL_QUOTIENT_LIMBS {
            for &quotient in &scalars {
                cases.push(DivisionCase {
                    divisor_limbs: len,
                    quotient_limbs: 1,
                    scalar_quotient: Some(quotient),
                });
            }
        }
        cases
    }

    pub fn score_division_cell(cell: ScoreCell, algorithm: DivisionAlgorithm) -> u128 {
        let numerator = ScoreCell::operand(cell.len_a, HASH_A);
        let denominator = ScoreCell::operand(cell.len_b, HASH_B);
        let mut reference = DivisionRunner::new(&numerator, &denominator);
        let mut runner = DivisionRunner::new(&numerator, &denominator);
        reference.run::<true>(DivisionAlgorithm::AlgorithmD);
        runner.run::<true>(algorithm);
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
        InterleavedMeasure::median_batch_samples(
            || black_box(&mut runner).run::<true>(algorithm),
            cell.iterations,
            cell.samples,
        )
    }

    /// Normalized and one-bit leading divisor limbs, each with exact and maximal
    /// remainders. A dense top quotient limb also exercises native-width division.
    pub fn division_fixtures(
        len: usize,
        quotient_len: usize,
        seed: usize,
    ) -> Vec<(MpUint, MpUint, MpUint, MpUint)> {
        let mut quotient_limbs = ScoreCell::operand(quotient_len, HASH_A.wrapping_add(seed));
        *quotient_limbs.last_mut().expect("nonempty quotient") |= usize::MAX << 1;
        let quotient_bytes: Vec<_> = quotient_limbs
            .into_iter()
            .flat_map(usize::to_le_bytes)
            .collect();
        let quotient = MpUint::from_le_bytes(&quotient_bytes);
        let mut fixtures = Vec::with_capacity(4);
        for top in [if len == 1 { (usize::MAX >> 1) | 1 } else { 1 }, usize::MAX] {
            let mut divisor_limbs = ScoreCell::operand(len, HASH_B.wrapping_add(seed));
            *divisor_limbs.last_mut().expect("nonempty divisor") = top;
            let bytes: Vec<_> = divisor_limbs
                .into_iter()
                .flat_map(usize::to_le_bytes)
                .collect();
            let divisor = MpUint::from_le_bytes(&bytes);
            let product = divisor.checked_mul(&quotient).expect("unlimited product");
            let residual = divisor
                .checked_sub(&MpUint::one())
                .expect("positive divisor");
            let inexact = product.checked_add(&residual).expect("unlimited sum");
            fixtures.push((product, divisor.clone(), quotient.clone(), MpUint::zero()));
            fixtures.push((inexact, divisor, quotient.clone(), residual));
        }
        fixtures
    }

    /// Form U=q*D+r. For q <= 65 both leading-limb shapes retain the divisor
    /// width on supported native targets. Top limb 1 and dense low limbs
    /// increase the leading-ratio estimate; the larger top limb bounds it
    /// tightly. Additional larger quotients may extend the numerator width.
    pub fn small_quotient_fixtures(
        len: usize,
        quotient: usize,
    ) -> Vec<(MpUint, MpUint, MpUint, MpUint)> {
        let byte_len = len
            .checked_mul(size_of::<usize>())
            .expect("fixture byte width fits");
        let top_offset = byte_len
            .checked_sub(size_of::<usize>())
            .expect("nonzero fixture width");
        let mut fixtures = Vec::with_capacity(6);
        for top in [1_usize, usize::MAX >> 7] {
            let mut bytes = vec![255; byte_len];
            bytes
                .get_mut(top_offset..)
                .expect("one top limb")
                .copy_from_slice(&top.to_le_bytes());
            let denominator = MpUint::from_le_bytes(&bytes);
            let exact_quotient = MpUint::from(quotient);
            let last_remainder = denominator
                .checked_sub(&MpUint::one())
                .expect("nonzero denominator");
            let product = denominator
                .checked_mul(&exact_quotient)
                .expect("quotient product");
            for remainder in [MpUint::zero(), MpUint::one(), last_remainder] {
                let numerator = product.checked_add(&remainder).expect("U=qD+r");
                fixtures.push((
                    numerator,
                    denominator.clone(),
                    exact_quotient.clone(),
                    remainder,
                ));
            }
        }
        fixtures
    }
}
