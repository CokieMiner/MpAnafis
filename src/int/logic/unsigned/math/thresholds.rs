//! Algorithmic crossover thresholds for integer operations.
//!
//! These constants define the limb counts at which the library transitions
//! between standard quadratic algorithms (such as Schoolbook multiplication or
//! Algorithm D division) and subquadratic divide-and-conquer algorithms
//! (such as Karatsuba, Toom-Cook, Burnikel-Ziegler, or Newton-Raphson).
//!
//! # Profile resolution
//!
//! `build.rs` resolves one complete typed profile in this order:
//!
//! | Priority | Source | How |
//! |---|---|---|
//! | 1 | `MP_TUNING_PROFILE` | Explicit candidate path, used by the rebuild-and-score phase of `mp-tune`. |
//! | 2 | `src/int/tuned_thresholds.rs` | Local, ignored machine profile emitted by `mp-tune`. |
//! | 3 | `build_support/tuning.rs` | Conservative target-architecture defaults, with pointer-width fallback. |
//!
//! Candidate and local files must define the entire profile. The build rejects
//! partial overrides, preventing a stale tuning file from silently inheriting a
//! newly introduced hardware-sensitive constant. The resolved source is written
//! to `OUT_DIR` and included below; none of these values are public API.
//!
//! # Autotuning
//!
//! Run the complete tuner on an otherwise idle, pinned CPU:
//!
//! ```sh
//! taskset -c 2 cargo run --release --bin mp-tune \
//!   --features _internal-tune
//! ```
//!
//! Its ordered phases tune compiled Toom geometry, multiplication and squaring
//! tiers, division geometry and dispatch, compiled SSA geometry, transform
//! crossovers, and radix formatting. Rebuild-based candidates are scored over
//! multiplication and squaring cells ranging from the crossover region through
//! working sets larger than the last-level cache. A candidate must beat the
//! calibrated host-noise margin; otherwise the existing value is retained.
//!
//! Partial modes make individual phase families repeatable, but they deliberately
//! preserve only rejected candidates because they skip end-to-end validation.
//! A complete validated result is written to the ignored local profile and takes
//! effect on the next build.

include!(concat!(env!("OUT_DIR"), "/thresholds.rs"));
