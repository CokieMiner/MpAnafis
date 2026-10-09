//! Whole-profile worker domains and independent product verification.

use super::{
    CellSelection, DIVISION_SCORE_CELLS, HASH_A, HASH_B, InterleavedMeasure, MUL_SCORE_CELLS,
    PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, SQR_SCORE_CELLS, ScoreCell, TOOM85_MUL_SCORE_CELLS,
    TOOM85_SQR_SCORE_CELLS, TRANSFORM_SHAPE_CELLS,
};

mod arithmetic;
mod consumers;
#[cfg(not(target_pointer_width = "16"))]
mod division;
mod gcd;
#[cfg(not(target_pointer_width = "16"))]
mod geometry;
#[cfg(not(target_pointer_width = "16"))]
mod holdout;
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
mod parallel;
#[cfg(not(target_pointer_width = "16"))]
mod production;
#[cfg(not(target_pointer_width = "16"))]
mod products;

pub use arithmetic::{ProfileWorkers, SsaScoreQuality};
pub use consumers::{CONSUMER_FORMAT_RADICES, CONSUMER_SCORE_COUNT, ConsumerWorker};
#[cfg(not(target_pointer_width = "16"))]
pub use division::{DivisionCase, DivisionScoreDomain, DivisionWorker, PRODUCTION_DIVISOR_LIMBS};
pub use gcd::{GCD_OPERATIONS, GCD_SCORE_CASES, GcdShape, GcdWorker};
#[cfg(not(target_pointer_width = "16"))]
pub use geometry::DivisionGrid;
#[cfg(not(target_pointer_width = "16"))]
pub use holdout::HoldoutWorker;
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
pub use parallel::ParallelWorker;
#[cfg(not(target_pointer_width = "16"))]
pub use products::{ProductGrid, ProductWorker};

#[cfg(all(test, not(target_pointer_width = "16")))]
mod tests;
