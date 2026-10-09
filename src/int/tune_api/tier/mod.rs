//! Prepared raw-limb calls through the production multiplication dispatcher.

#![doc(hidden)]

use super::{Limb, MulScratch, Multiplication};

mod state;
mod validation;

pub use state::{
    MultiplicationBenchState, PreparedMultiplication, PreparedSquaring, SquaringBenchState,
};
pub use validation::BenchValidation;
