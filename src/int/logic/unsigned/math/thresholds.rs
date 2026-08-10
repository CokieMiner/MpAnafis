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
//! | 1 | `ARBI_TUNING_PROFILE` | Explicit candidate path, used by the rebuild-and-score phase of `arbi-tune`. |
//! | 2 | `src/int/tuned_thresholds.rs` | Local, ignored machine profile emitted by `arbi-tune`. |
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
//! taskset -c 2 cargo run --release --bin arbi-tune \
//!   --features _internal-tune
//! ```
//!
//! Its first phase measures every adjacent multiplication and squaring tier
//! (Schoolbook through SSA/NTT) plus both division transitions. Its second phase
//! rebuilds candidate profiles for constants embedded in SSA planning and hot
//! loops. Those candidates are scored over multiplication and squaring cells
//! ranging from the crossover region through working sets larger than the last
//! level cache. A candidate needs a median-backed aggregate improvement of at
//! least one percent; otherwise the existing value is retained.
//!
//! `--tiers-only` and `--compiled-only` make the two phases independently
//! repeatable. The complete result is written to the ignored local profile and
//! takes effect on the next build.

include!(concat!(env!("OUT_DIR"), "/thresholds.rs"));
