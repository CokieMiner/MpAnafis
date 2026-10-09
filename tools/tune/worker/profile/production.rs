//! Production arithmetic scoring with caller-owned, prepared operands.

use core::hint::black_box;

use mp_anafis::tune_api::{Limb, MultiplicationBenchState, SquaringBenchState};

use super::{
    CellSelection, DivisionGrid, DivisionWorker, GCD_SCORE_CASES, GcdWorker, HASH_A, HASH_B,
    InterleavedMeasure, PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, ProfileWorkers, ScoreCell,
    TRANSFORM_SHAPE_CELLS,
};

impl ProfileWorkers {
    /// Print production-dispatch timings for end-to-end validation.
    pub fn print_production_score(grid: &DivisionGrid) {
        let cases = DivisionWorker::cases(grid);
        let mut values = Vec::with_capacity(
            PRODUCTION_MUL_CELLS
                .len()
                .checked_add(PRODUCTION_SQR_CELLS.len())
                .and_then(|count| count.checked_add(DivisionWorker::cell_weights(&cases).len()))
                .and_then(|count| {
                    count.checked_add(GcdWorker::cell_weights(&GCD_SCORE_CASES).len())
                })
                .expect("production cell catalog fits"),
        );
        for cell in PRODUCTION_MUL_CELLS {
            let left = ScoreCell::operand(cell.len_a, HASH_A);
            let right = ScoreCell::operand(cell.len_b, HASH_B);
            values.push(Self::score_production_mul_cell(cell, &left, &right));
        }
        for cell in PRODUCTION_SQR_CELLS {
            let value = ScoreCell::operand(cell.len_a, HASH_A);
            values.push(Self::score_production_sqr_cell(cell, &value));
        }
        values.extend(DivisionWorker::score_production_division(
            1_979,
            &cases,
            &[],
        ));
        values.extend(GcdWorker::score_cells(1_979, &GCD_SCORE_CASES, &[]));
        Self::print_encoded("MP_ANAFIS_PRODUCTION_SCORE=", &values);
    }

    /// Print production-dispatch timings on transform admission shapes.
    pub fn print_production_shapes_score(selection: &str) -> Result<(), String> {
        let indices = CellSelection::parse(selection, TRANSFORM_SHAPE_CELLS.len())?;
        let values: Vec<_> = indices
            .into_iter()
            .map(|index| {
                let cell = *TRANSFORM_SHAPE_CELLS
                    .get(index)
                    .expect("validated shape index");
                println!(
                    "MP_ANAFIS_CELL production_shape index={index} limbs_a={} limbs_b={}",
                    cell.len_a, cell.len_b
                );
                let left = ScoreCell::operand(cell.len_a, HASH_A);
                let right = ScoreCell::operand(cell.len_b, HASH_B);
                Self::score_production_mul_cell(cell, &left, &right)
            })
            .collect();
        Self::print_encoded("MP_ANAFIS_PRODUCTION_SHAPES_SCORE=", &values);
        Ok(())
    }

    /// Verify and time production multiplication with retained buffers. Operand
    /// construction and shape validation precede every measured batch.
    pub fn score_production_mul_cell(cell: ScoreCell, left: &[Limb], right: &[Limb]) -> u128 {
        assert_eq!(left.len(), cell.len_a, "product fixture left width");
        assert_eq!(right.len(), cell.len_b, "product fixture right width");
        let mut destination = vec![
            0;
            cell.len_a
                .checked_add(cell.len_b)
                .expect("validated product span")
        ];
        let mut runner = MultiplicationBenchState::default();
        runner.prepare(&mut destination, left, right).run();
        Self::verify_product_residues(&destination, left, right);
        let mut prepared = runner.prepare(&mut destination, left, right);
        InterleavedMeasure::median_batch_samples(
            || black_box(&mut prepared).run(),
            cell.iterations,
            cell.samples,
        )
    }

    /// Verify and time production squaring on the supplied full-width operand,
    /// retaining the output and scratch through warmup and measured batches.
    pub fn score_production_sqr_cell(cell: ScoreCell, value: &[Limb]) -> u128 {
        assert_eq!(value.len(), cell.len_a, "square fixture width");
        let mut destination = vec![0; cell.len_a.checked_mul(2).expect("validated square span")];
        let mut runner = SquaringBenchState::default();
        runner.prepare(&mut destination, value).run();
        Self::verify_product_residues(&destination, value, value);
        let mut prepared = runner.prepare(&mut destination, value);
        InterleavedMeasure::median_batch_samples(
            || black_box(&mut prepared).run(),
            cell.iterations,
            cell.samples,
        )
    }
}
