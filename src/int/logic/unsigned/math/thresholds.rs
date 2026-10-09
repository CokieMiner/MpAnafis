//! Algorithmic crossover thresholds for integer operations.
//!
//! `build.rs` selects a complete typed profile from `MP_TUNING_PROFILE`, the
//! ignored local `src/int/tuned_thresholds.rs`, or `build_support/defaults.rs`,
//! in that order. Partial profiles are rejected. The resolved constants are
//! generated in `OUT_DIR` and remain internal to the crate.

include!(concat!(env!("OUT_DIR"), "/thresholds.rs"));
