//! Direct product objectives and separate consumer regression guards.

use core::{hint::black_box, mem::size_of};

use mp_anafis::{
    MpUint,
    tune_api::{
        CyclicProductAlgorithm, CyclicProductRunner, DivisionAlgorithm, DivisionRunner,
        ModularPowAlgorithm, ModularPowRunner, MontgomeryProductRunner,
    },
};

use super::{CellSelection, HASH_A, HASH_B, InterleavedMeasure, ProfileWorkers, ScoreCell};

/// Frozen product dimensions, identical across all compared executables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductGrid {
    pub montgomery: Vec<usize>,
    pub cyclic: Vec<usize>,
}

impl Default for ProductGrid {
    fn default() -> Self {
        Self {
            montgomery: vec![1, 2, 4, 8, 16, 31, 32, 33, 64, 128, 129],
            cyclic: vec![
                128, 192, 256, 512, 1_024, 2_048, 3_072, 3_073, 3_074, 4_096, 8_192,
            ],
        }
    }
}

impl ProductGrid {
    /// Parses `montgomery_widths;cyclic_widths` before allocating fixtures.
    pub fn parse(specification: &str) -> Result<Self, String> {
        if specification.is_empty() {
            return Ok(Self::default());
        }
        let (montgomery, cyclic) = specification
            .split_once(';')
            .ok_or("product geometry requires Montgomery and cyclic dimensions")?;
        let parse = |text: &str| -> Result<Vec<usize>, String> {
            let values: Vec<usize> = text
                .split(',')
                .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
                .collect::<Result<_, _>>()?;
            if values.iter().any(|&value| {
                value == 0
                    || value
                        .checked_mul(4)
                        .and_then(|width| width.checked_add(2))
                        .and_then(|width| width.checked_mul(size_of::<usize>()))
                        .is_none_or(|bytes| isize::try_from(bytes).is_err())
            }) {
                return Err("product widths must be positive and addressable".to_owned());
            }
            Ok(values)
        };
        Ok(Self {
            montgomery: parse(montgomery)?,
            cyclic: parse(cyclic)?,
        })
    }

    /// Encodes both dimensions in the worker protocol and score-cache identity.
    #[must_use]
    pub fn render(&self) -> String {
        [&self.montgomery, &self.cyclic]
            .into_iter()
            .map(|widths| {
                widths
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join(";")
    }
}

#[derive(Clone, Copy, Debug)]
enum ProductOperation {
    Montgomery,
    CyclicBalanced,
    CyclicUneven,
    MontgomerySetupGuard,
    NewtonGuard,
}

/// Product measurement catalog and worker entry point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductWorker;

impl ProductWorker {
    /// Direct objectives exclude setup and caller work. Complete phase weights
    /// include every guard, so acceptance cannot hide a consumer regression.
    #[must_use]
    pub fn cell_weights(grid: &ProductGrid, policy: Option<&str>) -> Vec<u32> {
        Self::cells(grid)
            .iter()
            .map(|&(operation, _)| {
                u32::from(match policy {
                    None => true,
                    Some("MONTGOMERY_CIOS_MAX_LIMBS") => {
                        matches!(operation, ProductOperation::Montgomery)
                    }
                    Some("MUL_MOD_BNM1_THRESHOLD") => matches!(
                        operation,
                        ProductOperation::CyclicBalanced | ProductOperation::CyclicUneven
                    ),
                    Some(_) => false,
                })
            })
            .collect()
    }

    /// Executes direct calculations after checking independent results.
    pub fn print_score(specification: &str, selection: &str) -> Result<(), String> {
        let grid = ProductGrid::parse(specification)?;
        let cells = Self::cells(&grid);
        let indices = CellSelection::parse(selection, cells.len())?;
        let mut values = Vec::with_capacity(indices.len());
        for index in indices {
            let &(operation, width) = cells.get(index).expect("validated product cell");
            let operation_name = match operation {
                ProductOperation::Montgomery => "montgomery",
                ProductOperation::CyclicBalanced => "cyclic_balanced",
                ProductOperation::CyclicUneven => "cyclic_uneven",
                ProductOperation::MontgomerySetupGuard => "montgomery_setup_guard",
                ProductOperation::NewtonGuard => "newton_guard",
            };
            println!(
                "MP_ANAFIS_CELL products index={index} operation={operation_name} limbs={width}"
            );
            let value = match operation {
                ProductOperation::Montgomery => Self::montgomery(width),
                ProductOperation::CyclicBalanced | ProductOperation::CyclicUneven => {
                    Self::cyclic(width, matches!(operation, ProductOperation::CyclicUneven))
                }
                ProductOperation::MontgomerySetupGuard => Self::montgomery_setup(width),
                ProductOperation::NewtonGuard => Self::newton(width),
            };
            values.push(value);
        }
        ProfileWorkers::print_encoded("MP_ANAFIS_PRODUCTS_SCORE=", &values);
        Ok(())
    }

    fn cells(grid: &ProductGrid) -> Vec<(ProductOperation, usize)> {
        let mut cells = Vec::new();
        for (operation, widths) in [
            (ProductOperation::Montgomery, &grid.montgomery),
            (ProductOperation::CyclicBalanced, &grid.cyclic),
            (ProductOperation::CyclicUneven, &grid.cyclic),
            (ProductOperation::MontgomerySetupGuard, &grid.montgomery),
            (ProductOperation::MontgomerySetupGuard, &grid.cyclic),
            (ProductOperation::NewtonGuard, &grid.cyclic),
        ] {
            cells.extend(widths.iter().map(|&width| (operation, width)));
        }
        cells
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "fixture integers have unbounded precision, the positive modulus is checked by construction, and the radix bit span is validated before the independent congruence check"
    )]
    fn montgomery(width: usize) -> u128 {
        let mut modulus = ScoreCell::operand(width, HASH_A);
        let mut left = ScoreCell::operand(width, HASH_B);
        let mut right = ScoreCell::operand(width, HASH_B.wrapping_add(1));
        // M >= B^width/2 while a,b < B^width/2. The seed's low bit
        // keeps M odd, including width=1, where the original word is one.
        *modulus.last_mut().expect("positive width") |= !(usize::MAX >> 1);
        *left.last_mut().expect("positive width") &= usize::MAX >> 1;
        *right.last_mut().expect("positive width") &= usize::MAX >> 1;
        let mut runner = MontgomeryProductRunner::new(&left, &right, &modulus);
        let [a, b, m, result] = [&left[..], &right[..], &modulus[..], runner.run()].map(|limbs| {
            MpUint::from_le_bytes(
                &limbs
                    .iter()
                    .flat_map(|limb| limb.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        });
        let bits = width
            .checked_mul(usize::try_from(usize::BITS).expect("native width fits"))
            .expect("fixture bits fit");
        assert_eq!(
            (result << bits).checked_rem(&m).expect("positive modulus"),
            (a * b).checked_rem(&m).expect("positive modulus"),
            "raw Montgomery identity"
        );
        InterleavedMeasure::median_batch_samples(
            || {
                let _ = black_box(runner.run());
            },
            8,
            9,
        )
    }

    fn cyclic(width: usize, uneven: bool) -> u128 {
        let right_width = if uneven { width.div_ceil(2) } else { width };
        let left = ScoreCell::operand(width, HASH_A);
        let right = ScoreCell::operand(right_width, HASH_B);
        let mut runner = CyclicProductRunner::new(&left, &right, width);
        let expected = runner.run(CyclicProductAlgorithm::Full).to_vec();
        for algorithm in [
            CyclicProductAlgorithm::Cyclic,
            CyclicProductAlgorithm::Production,
        ] {
            let actual = runner.run(algorithm);
            let zero = |limbs: &[usize]| {
                limbs.iter().all(|&limb| limb == 0) || limbs.iter().all(|&limb| limb == usize::MAX)
            };
            assert!(
                actual == expected || (zero(actual) && zero(&expected)),
                "cyclic and folded products agree in the same ring"
            );
        }
        InterleavedMeasure::median_batch_samples(
            || {
                let _ = black_box(runner.run(CyclicProductAlgorithm::Production));
            },
            8,
            9,
        )
    }

    fn montgomery_setup(width: usize) -> u128 {
        let mut modulus = ScoreCell::operand(width, HASH_A);
        // Match the direct product's positive odd, full-width modulus.
        *modulus.last_mut().expect("positive width") |= !(usize::MAX >> 1);
        let base = ScoreCell::operand(width, HASH_B);
        let mut runner = ModularPowRunner::new(&base, &[0b10_1101], &modulus);
        let expected = runner.run(ModularPowAlgorithm::Barrett);
        assert_eq!(
            runner.run(ModularPowAlgorithm::Montgomery),
            expected,
            "Montgomery power agrees with Barrett"
        );
        InterleavedMeasure::median_batch_samples(
            || {
                drop(black_box(runner.run(ModularPowAlgorithm::Montgomery)));
            },
            2,
            9,
        )
    }

    fn newton(width: usize) -> u128 {
        let numerator_width = width
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .expect("validated Newton fixture span");
        let numerator = ScoreCell::operand(numerator_width, HASH_A);
        let divisor = ScoreCell::operand(width, HASH_B);
        let mut runner = DivisionRunner::new(&numerator, &divisor);
        runner.run::<true>(DivisionAlgorithm::AlgorithmD);
        let quotient = runner.quotient_limbs().to_vec();
        let remainder = runner.remainder_limbs().to_vec();
        runner.run::<true>(DivisionAlgorithm::NewtonRaphson);
        assert_eq!(
            runner.quotient_limbs(),
            quotient,
            "Newton quotient agrees with Algorithm D"
        );
        assert_eq!(
            runner.remainder_limbs(),
            remainder,
            "Newton remainder agrees with Algorithm D"
        );
        InterleavedMeasure::median_batch_samples(
            || {
                runner.run::<true>(DivisionAlgorithm::NewtonRaphson);
                let _ = black_box(runner.quotient_limbs());
            },
            2,
            9,
        )
    }
}
