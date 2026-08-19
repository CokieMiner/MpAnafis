//! Hidden subprocess workers for compiled profiles and adjacent-tier probes.

mod cells;
mod pairs;
mod profile;

use cells::{HASH_A, HASH_B, operand};

pub use cells::{
    DIVISION_SCORE_CELLS, MUL_SCORE_CELLS, PRODUCTION_DIV_CELLS, PRODUCTION_MUL_CELLS,
    PRODUCTION_SQR_CELLS, SQR_SCORE_CELLS, ScoreCell, TOOM85_MUL_SCORE_CELLS,
    TOOM85_SQR_SCORE_CELLS, cell_weights, division_cell_weights,
};
pub use pairs::{print_fmt_pair_score, print_mul_pair_score, print_sqr_pair_score};
pub use profile::{
    DivisionScoreDomain, SsaScoreQuality, print_division_score, print_production_score,
    print_ssa_score, print_toom85_mul_score, print_toom85_score,
};
