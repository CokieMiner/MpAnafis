//! Where each threshold in the tower should sit.
//!
//! These sweeps answer "at what width should X hand over to Y", and their
//! output feeds the constants in `build_support/tuning.rs`. They differ from
//! `tiers` in that both arms of a crossover measurement are production paths;
//! nothing is forced outside the range it would really run in.

pub mod blocked_vs_transform;
pub mod padding_bound;
pub mod square;
pub mod square_ladder;
pub mod tower;
pub mod transform;
pub mod wide_block_gate;
