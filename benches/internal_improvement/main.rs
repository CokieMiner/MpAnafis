//! Multiplication benchmarks for internal improvement.
//!
//! Grouped by the question each sweep answers rather than by algorithm:
//!
//! - `tiers` — how fast is one forced algorithm at a given width
//! - `crossovers` — where should a threshold between two tiers sit
//! - `shapes` — how the tower handles unbalanced operand ratios
//! - `transform` — which SSA geometry and execution policy the planner should pick
//! - `kernels` — whether a leaf kernel used by multiplication earns its complexity
//! - `compare` — external references, led by complete production-tower comparisons
//!
//! Public-API-level comparisons belong in the `public_api` target instead.
//!
//! Divan does not pin this process or interleave separate benchmark rows. Run
//! comparison filters on an otherwise idle host under an external affinity
//! tool such as `taskset`. A close performance claim requires repeated filtered
//! invocations in A/B/B/A order; one invocation of this target is not that
//! protocol. Transform argument labels record resolved geometry, executor
//! parallelism, and workspace ownership so those runs remain distinguishable.

mod compare;
mod crossovers;
mod kernels;
mod shapes;
mod shared;
mod tiers;
mod transform;

fn main() {
    divan::main();
}
