//! Forced multiplication and squaring catalogs with product verification.

use core::hint::black_box;

use mp_anafis::tune_api::{
    Limb, MultiplicationAlgorithm, MultiplicationRunner, SquaringAlgorithm, SquaringRunner,
};

use super::{
    CellSelection, HASH_A, HASH_B, InterleavedMeasure, MUL_SCORE_CELLS, SQR_SCORE_CELLS, ScoreCell,
    TOOM85_MUL_SCORE_CELLS, TOOM85_SQR_SCORE_CELLS,
};

#[cfg(target_pointer_width = "64")]
const ORACLE_MODULI: [u128; 3] = [
    18_446_744_073_709_551_557,
    18_446_744_073_709_551_533,
    18_446_744_073_709_551_521,
];

#[cfg(target_pointer_width = "32")]
const ORACLE_MODULI: [u128; 3] = [4_294_967_291, 4_294_967_279, 4_294_967_231];

/// Sampling depth for the forced-SSA worker protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SsaScoreQuality {
    /// Full inner-batch sample depth for forced SSA workers.
    Precise,
    /// Reduced sample depth for candidate screening.
    Coarse,
}

/// Whole-profile worker domains and output emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileWorkers;

impl ProfileWorkers {
    /// Print forced-SSA timings for precise scoring or candidate screening.
    pub fn print_ssa_score(quality: SsaScoreQuality, selection: &str) -> Result<(), String> {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let coarse = matches!(quality, SsaScoreQuality::Coarse);
            let count = MUL_SCORE_CELLS
                .len()
                .checked_add(SQR_SCORE_CELLS.len())
                .expect("SSA catalog fits");
            let indices = CellSelection::parse(selection, count)?;
            let mut values = Vec::with_capacity(indices.len());
            for index in indices {
                let (square, cell) = if let Some(&cell) = MUL_SCORE_CELLS.get(index) {
                    (false, cell)
                } else {
                    (
                        true,
                        *SQR_SCORE_CELLS
                            .get(
                                index
                                    .checked_sub(MUL_SCORE_CELLS.len())
                                    .expect("square offset"),
                            )
                            .expect("validated SSA index"),
                    )
                };
                println!(
                    "MP_ANAFIS_CELL ssa index={index} square={square} limbs_a={} limbs_b={}",
                    cell.len_a, cell.len_b
                );
                let measured_cell = if coarse { cell.coarse() } else { cell };
                values.push(if square {
                    Self::score_forced_sqr_cell(measured_cell, SquaringAlgorithm::SsaForced)
                } else {
                    Self::score_forced_mul_cell(measured_cell, MultiplicationAlgorithm::SsaForced)
                });
            }
            Self::print_encoded(
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
        Ok(())
    }

    /// Print multiplication-only forced-SSA timings for direct-Fermat crossover tuning.
    pub fn print_ssa_mul_score(quality: SsaScoreQuality) {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let coarse = matches!(quality, SsaScoreQuality::Coarse);
            let values = MUL_SCORE_CELLS.map(|cell| {
                Self::score_forced_mul_cell(
                    if coarse { cell.coarse() } else { cell },
                    MultiplicationAlgorithm::SsaForced,
                )
            });
            Self::print_encoded(
                match quality {
                    SsaScoreQuality::Precise => "MP_ANAFIS_SSA_MUL_SCORE=",
                    SsaScoreQuality::Coarse => "MP_ANAFIS_SSA_MUL_COARSE_SCORE=",
                },
                &values,
            );
        }
        #[cfg(target_pointer_width = "16")]
        match quality {
            SsaScoreQuality::Precise => println!("MP_ANAFIS_SSA_MUL_SCORE="),
            SsaScoreQuality::Coarse => println!("MP_ANAFIS_SSA_MUL_COARSE_SCORE="),
        }
    }

    /// Print direct forced-Toom-8.5 timings for its reconstruction knobs.
    pub fn print_toom85_score(selection: &str) -> Result<(), String> {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let count = TOOM85_MUL_SCORE_CELLS
                .len()
                .checked_add(TOOM85_SQR_SCORE_CELLS.len())
                .expect("Toom catalog fits");
            let indices = CellSelection::parse(selection, count)?;
            let mut values = Vec::with_capacity(indices.len());
            for index in indices {
                values.push(if let Some(&cell) = TOOM85_MUL_SCORE_CELLS.get(index) {
                    println!(
                        "MP_ANAFIS_CELL toom85_mul index={index} limbs={}",
                        cell.len_a
                    );
                    Self::score_forced_mul_cell(cell, MultiplicationAlgorithm::ToomCook85)
                } else {
                    let offset = index
                        .checked_sub(TOOM85_MUL_SCORE_CELLS.len())
                        .expect("square offset");
                    let cell = *TOOM85_SQR_SCORE_CELLS
                        .get(offset)
                        .expect("validated Toom index");
                    println!(
                        "MP_ANAFIS_CELL toom85_sqr index={index} limbs={}",
                        cell.len_a
                    );
                    Self::score_forced_sqr_cell(cell, SquaringAlgorithm::ToomCook85)
                });
            }
            Self::print_encoded("MP_ANAFIS_TOOM85_SCORE=", &values);
        }
        #[cfg(target_pointer_width = "16")]
        println!("MP_ANAFIS_TOOM85_SCORE=");
        Ok(())
    }

    /// Print forced Toom-8.5 multiplication timings for multiplication-only knobs.
    pub fn print_toom85_mul_score(selection: &str) -> Result<(), String> {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let indices = CellSelection::parse(selection, TOOM85_MUL_SCORE_CELLS.len())?;
            let values: Vec<_> = indices
                .into_iter()
                .map(|index| {
                    let cell = *TOOM85_MUL_SCORE_CELLS
                        .get(index)
                        .expect("validated Toom index");
                    println!(
                        "MP_ANAFIS_CELL toom85_mul index={index} limbs={}",
                        cell.len_a
                    );
                    Self::score_forced_mul_cell(cell, MultiplicationAlgorithm::ToomCook85)
                })
                .collect();
            Self::print_encoded("MP_ANAFIS_TOOM85_MUL_SCORE=", &values);
        }
        #[cfg(target_pointer_width = "16")]
        println!("MP_ANAFIS_TOOM85_MUL_SCORE=");
        Ok(())
    }

    #[cfg(not(target_pointer_width = "16"))]
    pub fn score_forced_mul_cell(cell: ScoreCell, algorithm: MultiplicationAlgorithm) -> u128 {
        let left = ScoreCell::operand(cell.len_a, HASH_A);
        let right = ScoreCell::operand(cell.len_b, HASH_B);
        let mut destination = vec![
            0;
            cell.len_a
                .checked_add(cell.len_b)
                .expect("validated product span")
        ];
        let mut runner = MultiplicationRunner::new(algorithm, cell.len_a, cell.len_b);
        runner.run(&mut destination, &left, &right);
        Self::verify_product_residues(&destination, &left, &right);
        let mut prepared = runner.prepare(&mut destination, &left, &right);
        InterleavedMeasure::median_batch_samples(
            || black_box(&mut prepared).run(),
            cell.iterations,
            cell.samples,
        )
    }

    #[cfg(not(target_pointer_width = "16"))]
    pub fn score_forced_sqr_cell(cell: ScoreCell, algorithm: SquaringAlgorithm) -> u128 {
        let value = ScoreCell::operand(cell.len_a, HASH_A);
        let mut destination = vec![0; cell.len_a.checked_mul(2).expect("validated square span")];
        let mut runner = SquaringRunner::new(algorithm, cell.len_a);
        runner.run(&mut destination, &value);
        Self::verify_product_residues(&destination, &value, &value);
        let mut prepared = runner.prepare(&mut destination, &value);
        InterleavedMeasure::median_batch_samples(
            || black_box(&mut prepared).run(),
            cell.iterations,
            cell.samples,
        )
    }

    /// Verify three product congruences independently of the selected tier.
    /// Linear modular projections avoid a quadratic reference product in large
    /// SSA cells. Verification precedes timing; congruences are not an exact
    /// reference for arbitrary corrupt products.
    #[cfg(not(target_pointer_width = "16"))]
    pub fn verify_product_residues(product: &[Limb], left: &[Limb], right: &[Limb]) {
        for modulus in ORACLE_MODULI {
            // Each projection is below a modulus smaller than 2^64, so the
            // exact product fits u128 on all supported pointer widths.
            let expected = Self::limbs_mod(left, modulus)
                .checked_mul(Self::limbs_mod(right, modulus))
                .expect("projection product fits u128")
                .rem_euclid(modulus);
            assert_eq!(
                Self::limbs_mod(product, modulus),
                expected,
                "compiled multiplication candidate failed the independent modular-product oracle"
            );
        }
    }

    #[cfg(not(target_pointer_width = "16"))]
    fn limbs_mod(limbs: &[Limb], modulus: u128) -> u128 {
        let max = u128::try_from(Limb::MAX).expect("native limb fits u128");
        let radix = max
            .checked_add(1)
            .expect("native radix is at most 2^64")
            .rem_euclid(modulus);
        // residue < modulus < 2^64, radix <= 2^64 and limb < 2^64.
        // Hence residue*radix + limb < 2^128, including 64-bit limbs.
        limbs.iter().rev().fold(0, |residue, &limb| {
            residue
                .checked_mul(radix)
                .and_then(|product| {
                    product.checked_add(u128::try_from(limb).expect("native limb fits u128"))
                })
                .expect("projection step fits u128")
                .rem_euclid(modulus)
        })
    }

    #[cfg(not(target_pointer_width = "16"))]
    pub fn print_encoded(prefix: &str, values: &[u128]) {
        let encoded = values
            .iter()
            .map(u128::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!("{prefix}{encoded}");
    }
}
