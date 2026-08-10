//! Multiplication benchmarks for internal improvement.
//!
//! Grouped by the question each sweep answers rather than by algorithm:
//!
//! - `tiers` — how fast is one forced algorithm at a given width
//! - `crossovers` — where should a threshold between two tiers sit
//! - `shapes` — how the tower handles unbalanced operand ratios
//! - `transform` — which SSA/NTT geometry the planner should pick
//! - `compare` — where we stand against GMP and FLINT
//!
//! Public-API-level comparisons belong in the `public_api` target instead.

mod compare;
mod crossovers;
mod shapes;
mod shared;
mod tiers;
mod transform;

fn main() {
    divan::main();
}
