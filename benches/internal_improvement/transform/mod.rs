//! SSA and NTT geometry selection.
//!
//! The planner picks a ring width, a transform exponent and a chunk size for
//! every product above the transform crossover. These sweeps measure the cost
//! surface it is choosing over, including at the RAM-resident ring widths where
//! the memory system rather than the butterfly kernel sets the cost.

pub mod base_modulus;
pub mod geometry;
pub mod ram_optima;
