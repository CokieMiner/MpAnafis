//! Parallel SSA policy cells with explicit, real Rayon pool widths.

use mp_anafis::tune_api::{MultiplicationAlgorithm, SquaringAlgorithm};
use rayon::{ThreadPoolBuilder, current_num_threads};

use crate::platform::Platform;

use super::{
    CellSelection, HASH_A, HASH_B, MUL_SCORE_CELLS, PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS,
    ProfileWorkers, SQR_SCORE_CELLS, ScoreCell,
};

/// Compiled parallel-product measurements across the selected worker budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelWorker;

impl ParallelWorker {
    /// Validate affinity, construct each pool, and time the same complete products.
    ///
    /// Every pool width from one to the selected CPU count is measured. Pool
    /// construction and correctness verification precede each timed operation;
    /// runners retain their output buffers and scratch within a cell.
    pub fn print_score(
        specification: &str,
        production: bool,
        extra_widths: &[usize],
        selection: &str,
    ) -> Result<(), String> {
        let affinity = Platform::parallel_cpu_affinity(specification)?;
        let maximum = Platform::parse_cpu_list(specification)?.len();
        println!("MP_ANAFIS_PARALLEL_CONTEXT {}", affinity.description);
        let (multiplication, squaring) = Self::cell_catalog(production, extra_widths);
        let per_pool = multiplication
            .len()
            .checked_add(squaring.len())
            .expect("parallel catalog fits");
        let indices = CellSelection::parse(
            selection,
            maximum
                .checked_mul(per_pool)
                .expect("parallel catalog fits"),
        )?;
        let mut values = Vec::new();
        for workers in 1..=maximum {
            let offset = workers
                .checked_sub(1)
                .and_then(|pool| pool.checked_mul(per_pool))
                .expect("pool offset fits");
            let selected: Vec<_> = indices
                .iter()
                .filter_map(|&index| index.checked_sub(offset).filter(|&local| local < per_pool))
                .collect();
            if selected.is_empty() {
                continue;
            }
            let pool = ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|error| error.to_string())?;
            let measured = pool.install(|| {
                if current_num_threads() != workers {
                    return Err(
                        "the active pool does not match the requested worker budget".to_owned()
                    );
                }
                let mut timings = Vec::with_capacity(selected.len());
                for index in &selected {
                    if let Some(&cell) = multiplication.get(*index) {
                        println!("MP_ANAFIS_CELL parallel_mul production={production} workers={workers} index={index} limbs_a={} limbs_b={}", cell.len_a, cell.len_b);
                        timings.push(if production {
                            let left = ScoreCell::operand(cell.len_a, HASH_A);
                            let right = ScoreCell::operand(cell.len_b, HASH_B);
                            ProfileWorkers::score_production_mul_cell(cell, &left, &right)
                        } else { ProfileWorkers::score_forced_mul_cell(cell, MultiplicationAlgorithm::SsaForced) });
                    } else {
                        let local = index.checked_sub(multiplication.len()).expect("square offset");
                        let cell = *squaring.get(local).expect("validated square cell");
                        println!("MP_ANAFIS_CELL parallel_sqr production={production} workers={workers} index={index} limbs={}", cell.len_a);
                        timings.push(if production {
                            let value = ScoreCell::operand(cell.len_a, HASH_A);
                            ProfileWorkers::score_production_sqr_cell(cell, &value)
                        } else { ProfileWorkers::score_forced_sqr_cell(cell, SquaringAlgorithm::SsaForced) });
                    }
                }
                Ok(timings)
            })?;
            values.extend(measured);
        }
        ProfileWorkers::print_encoded(
            if production {
                "MP_ANAFIS_PARALLEL_PRODUCTION_SCORE="
            } else {
                "MP_ANAFIS_PARALLEL_SSA_SCORE="
            },
            &values,
        );
        Ok(())
    }

    /// Match the exact pool-major multiplication-then-square execution order.
    pub fn cell_weights(maximum: usize, production: bool, extra_widths: &[usize]) -> Vec<u32> {
        assert!(
            maximum >= 2,
            "parallel policy grid requires multiple workers"
        );
        let (multiplication, squaring) = Self::cell_catalog(production, extra_widths);
        let weights = ScoreCell::cell_weights(&multiplication, &squaring);
        let mut result = Vec::with_capacity(
            maximum
                .checked_mul(weights.len())
                .expect("parallel grid fits"),
        );
        for _ in 1..=maximum {
            result.extend_from_slice(&weights);
        }
        result
    }

    /// Forced roots retain their fixed kernel grid. Production includes the
    /// complete arithmetic ladder, SSA shapes and both profiles' boundary limbs.
    /// Boundary cells include balanced and 2:1 products, with shared buffers.
    #[must_use]
    pub fn cell_catalog(
        production: bool,
        extra_widths: &[usize],
    ) -> (Vec<ScoreCell>, Vec<ScoreCell>) {
        let mut multiplication = MUL_SCORE_CELLS.to_vec();
        let mut squaring = SQR_SCORE_CELLS.to_vec();
        if production {
            multiplication.extend_from_slice(&PRODUCTION_MUL_CELLS);
            squaring.extend_from_slice(&PRODUCTION_SQR_CELLS);
            for width in [1, 2, 4, 8].into_iter().chain(extra_widths.iter().copied()) {
                multiplication.push(ScoreCell::new_balanced(width, 1, 3));
                multiplication.push(ScoreCell::new(
                    width.checked_mul(2).expect("validated product span"),
                    width,
                    1,
                    3,
                ));
                squaring.push(ScoreCell::new_balanced(width, 1, 3));
            }
            multiplication.sort_by_key(|cell| (cell.len_a, cell.len_b));
            squaring.sort_by_key(|cell| cell.len_a);
            multiplication.dedup_by_key(|cell| (cell.len_a, cell.len_b));
            squaring.dedup_by_key(|cell| cell.len_a);
        }
        for cell in multiplication.iter_mut().chain(&mut squaring) {
            cell.samples = 3;
        }
        (multiplication, squaring)
    }
}
