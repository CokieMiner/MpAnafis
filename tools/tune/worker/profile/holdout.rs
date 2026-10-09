//! Fixed production fixtures reserved for the installation gate.
//!
//! These widths, seeds and digit/exponent patterns never score search updates.
//! Selection only limits diagnostic worker execution; installation uses all cells.

use core::{
    fmt::{Display, Formatter, Result as FmtResult},
    hint::black_box,
    ops::Range,
};

use mp_anafis::{
    MpUint,
    tune_api::{Limb, ModularPowAlgorithm, ModularPowRunner},
};

use super::{
    CellSelection, DivisionCase, DivisionWorker, GCD_OPERATIONS, GcdShape, GcdWorker, HASH_A,
    HASH_B, InterleavedMeasure, ProfileWorkers, ScoreCell,
};

/// Reproducible fixture seed distinct from screening and production catalogs.
pub const HOLDOUT_SEED: usize = 0x51ED_27B3;

/// Carry and sparsity distributions omitted from the arithmetic search operands.
#[derive(Clone, Copy, Debug)]
pub enum ProductPattern {
    Mixed,
    Maximal,
    Sparse,
    Alternating,
}

/// Widths between arithmetic search ladder points, including uneven shapes.
pub const PRODUCT_CASES: [(ScoreCell, ProductPattern); 6] = [
    (ScoreCell::new_balanced(59, 16, 3), ProductPattern::Mixed),
    (ScoreCell::new_balanced(241, 8, 3), ProductPattern::Maximal),
    (ScoreCell::new_balanced(1_009, 4, 3), ProductPattern::Sparse),
    (
        ScoreCell::new_balanced(4_093, 1, 3),
        ProductPattern::Alternating,
    ),
    (ScoreCell::new(2_053, 127, 4, 3), ProductPattern::Maximal),
    (ScoreCell::new(8_191, 509, 1, 3), ProductPattern::Mixed),
];

/// Independent divisor and quotient widths; every output and normalization is checked.
pub const DIVISION_CASES: [DivisionCase; 2] = [
    DivisionCase {
        divisor_limbs: 53,
        quotient_limbs: 17,
        scalar_quotient: None,
    },
    DivisionCase {
        divisor_limbs: 2_039,
        quotient_limbs: 3_059,
        scalar_quotient: None,
    },
];

/// Different widths and seeds for all four public Euclidean families.
pub const GCD_CASES: [(usize, GcdShape); 4] = [
    (37, GcdShape::Random),
    (211, GcdShape::NearEqual),
    (73, GcdShape::Fibonacci),
    (509, GcdShape::Uneven),
];

/// Radix/digit counts use a periodic digit pattern independent of parsing runners.
pub const CONVERSION_CASES: [(u32, usize); 6] = [
    (7, 113),
    (7, 12_289),
    (10, 113),
    (10, 12_289),
    (23, 113),
    (23, 12_289),
];

/// Exponent structure reserved for modular-power validation.
#[derive(Clone, Copy, Debug)]
enum ExponentPattern {
    Dense,
    Sparse,
    Irregular,
}

/// Odd modular widths and dense, sparse, or irregular exponent patterns.
const POW_CASES: [(usize, ExponentPattern); 6] = [
    (7, ExponentPattern::Dense),
    (7, ExponentPattern::Sparse),
    (7, ExponentPattern::Irregular),
    (211, ExponentPattern::Dense),
    (211, ExponentPattern::Sparse),
    (211, ExponentPattern::Irregular),
];

/// Reserved production catalog and its exact family-major protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HoldoutWorker;

impl HoldoutWorker {
    /// Derive weights and family ranges once from the catalogs execution uses.
    /// All four Euclidean operations, parsing and formatting have separate guards.
    #[must_use]
    pub fn layout() -> (Vec<u32>, Vec<Range<usize>>) {
        let cells: Vec<_> = PRODUCT_CASES.iter().map(|&(cell, _)| cell).collect();
        let mut groups = vec![
            ScoreCell::cell_weights(&cells, &[]),
            ScoreCell::cell_weights(&[], &cells),
            DivisionWorker::cell_weights(&DIVISION_CASES),
        ];
        groups.extend(GCD_OPERATIONS.iter().map(|_| vec![1; GCD_CASES.len()]));
        groups.extend([
            vec![1; CONVERSION_CASES.len()],
            vec![1; CONVERSION_CASES.len()],
            vec![1; POW_CASES.len()],
        ]);
        let mut weights = Vec::new();
        let mut families = Vec::new();
        for group in groups {
            let start = weights.len();
            weights.extend(group);
            families.push(start..weights.len());
        }
        (weights, families)
    }

    /// Execute fixed held-out calls with result checks outside the clock.
    pub fn print_score(selection: &str) -> Result<(), String> {
        let (weights, _) = Self::layout();
        let selected = CellSelection::parse(selection, weights.len())?;
        let mut values = Vec::with_capacity(selected.len());
        let mut index = 0_usize;
        for square in [false, true] {
            for (cell, pattern) in PRODUCT_CASES {
                if selected.binary_search(&index).is_ok() {
                    let left = pattern.operand(cell.len_a, HASH_A);
                    println!(
                        "MP_ANAFIS_CELL holdout_product index={index} square={square} limbs_a={} limbs_b={} pattern={pattern} seed={HOLDOUT_SEED}",
                        cell.len_a, cell.len_b
                    );
                    values.push(if square {
                        ProfileWorkers::score_production_sqr_cell(cell, &left)
                    } else {
                        let right = pattern.operand(cell.len_b, HASH_B);
                        ProfileWorkers::score_production_mul_cell(cell, &left, &right)
                    });
                }
                index = index.checked_add(1).expect("finite holdout catalog");
            }
        }
        Self::append_group(
            &selected,
            &mut index,
            DivisionWorker::cell_weights(&DIVISION_CASES).len(),
            &mut values,
            |indices| {
                DivisionWorker::score_production_division(HOLDOUT_SEED, &DIVISION_CASES, indices)
            },
        );
        Self::append_group(
            &selected,
            &mut index,
            GcdWorker::cell_weights(&GCD_CASES).len(),
            &mut values,
            |indices| {
                GcdWorker::score_cells(
                    u64::try_from(HOLDOUT_SEED).expect("seed fits u64"),
                    &GCD_CASES,
                    indices,
                )
            },
        );
        for formatting in [false, true] {
            for (radix, digits) in CONVERSION_CASES {
                if selected.binary_search(&index).is_ok() {
                    println!(
                        "MP_ANAFIS_CELL holdout_conversion index={index} formatting={formatting} radix={radix} digits={digits}"
                    );
                    values.push(Self::score_conversion(radix, digits, formatting));
                }
                index = index.checked_add(1).expect("finite holdout catalog");
            }
        }
        for (len, pattern) in POW_CASES {
            if selected.binary_search(&index).is_ok() {
                println!("MP_ANAFIS_CELL holdout_pow index={index} limbs={len} pattern={pattern}");
                values.push(Self::score_pow(len, pattern));
            }
            index = index.checked_add(1).expect("finite holdout catalog");
        }
        if index != weights.len() || values.len() != selected.len() {
            return Err("held-out layout or selection disagrees with execution".to_owned());
        }
        println!(
            "MP_ANAFIS_HOLDOUT_SCORE={}",
            values
                .iter()
                .map(u128::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(())
    }

    /// Translate global selected indices to a retained family's local protocol.
    fn append_group<F>(
        selected: &[usize],
        offset: &mut usize,
        count: usize,
        values: &mut Vec<u128>,
        score: F,
    ) where
        F: FnOnce(&[usize]) -> Vec<u128>,
    {
        let end = offset.checked_add(count).expect("finite holdout catalog");
        let local: Vec<_> = selected
            .iter()
            .copied()
            .filter(|&index| (*offset..end).contains(&index))
            .map(|index| index.checked_sub(*offset).expect("family index"))
            .collect();
        println!(
            "MP_ANAFIS_HOLDOUT_GROUP start={offset} end={end} selected={}",
            CellSelection::encode(&local)
        );
        if !local.is_empty() {
            values.extend(score(&local));
        }
        *offset = end;
    }

    /// Construct digits and their value by Horner accumulation, independently
    /// of parsing dispatch. The first digit is nonzero and every digit is valid.
    #[must_use]
    pub fn conversion_fixture(radix: u32, digits: usize) -> (String, MpUint) {
        let mut input = String::with_capacity(digits);
        let mut expected = MpUint::zero();
        let base = MpUint::from(radix);
        for index in 0..digits {
            let digit = if index.is_multiple_of(3) {
                radix.checked_sub(1).expect("radix at least three")
            } else {
                1
            };
            input.push(char::from_digit(digit, radix).expect("valid radix digit"));
            expected = expected
                .checked_mul(&base)
                .and_then(|value| value.checked_add(&MpUint::from(digit)))
                .expect("unbounded Horner value");
        }
        (input, expected)
    }

    fn score_conversion(radix: u32, digits: usize, formatting: bool) -> u128 {
        let (input, expected) = Self::conversion_fixture(radix, digits);
        assert_eq!(
            MpUint::from_str_radix(&input, radix),
            Ok(expected.clone()),
            "held-out parsing matches Horner"
        );
        assert_eq!(
            expected.to_string_radix(radix),
            input,
            "held-out formatting matches prepared digits"
        );
        if formatting {
            InterleavedMeasure::median_batch_samples(
                || {
                    drop(black_box(
                        black_box(&expected).to_string_radix(black_box(radix)),
                    ));
                },
                1,
                3,
            )
        } else {
            InterleavedMeasure::median_batch_samples(
                || {
                    drop(black_box(MpUint::from_str_radix(
                        black_box(&input),
                        black_box(radix),
                    )));
                },
                1,
                3,
            )
        }
    }

    fn score_pow(len: usize, pattern: ExponentPattern) -> u128 {
        let base = ProductPattern::Mixed.operand(len, HASH_A);
        let modulus = ProductPattern::Mixed.operand(len, HASH_B);
        let exponent = match pattern {
            ExponentPattern::Dense => vec![Limb::MAX; 3],
            ExponentPattern::Sparse => vec![1, 0, 1],
            ExponentPattern::Irregular => vec![HOLDOUT_SEED, 1, HOLDOUT_SEED],
        };
        let mut runner = ModularPowRunner::new(&base, &exponent, &modulus);
        assert_eq!(
            runner.run(ModularPowAlgorithm::Production),
            runner.run(ModularPowAlgorithm::Barrett),
            "held-out modular power matches Barrett"
        );
        InterleavedMeasure::median_batch_samples(
            || {
                drop(black_box(
                    black_box(&mut runner).run(ModularPowAlgorithm::Production),
                ));
            },
            1,
            3,
        )
    }
}

impl ProductPattern {
    /// Generate canonical full-width operands. Wrapping defines the mixed
    /// fixture's deterministic native-limb sequence rather than sizing arithmetic.
    #[must_use]
    pub fn operand(self, len: usize, hash: Limb) -> Vec<Limb> {
        let mut limbs: Vec<_> = (0..len)
            .map(|index| match self {
                Self::Mixed => index.wrapping_add(HOLDOUT_SEED).wrapping_mul(hash) | 1,
                Self::Maximal => Limb::MAX,
                Self::Sparse => Limb::from(index == 0),
                Self::Alternating => {
                    if index.is_multiple_of(2) {
                        Limb::MAX
                    } else {
                        0
                    }
                }
            })
            .collect();
        if let Some(top) = limbs.last_mut() {
            *top |= 1;
        }
        limbs
    }
}

impl Display for ProductPattern {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str(match self {
            Self::Mixed => "mixed",
            Self::Maximal => "maximal",
            Self::Sparse => "sparse",
            Self::Alternating => "alternating",
        })
    }
}

impl Display for ExponentPattern {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str(match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Irregular => "irregular",
        })
    }
}
