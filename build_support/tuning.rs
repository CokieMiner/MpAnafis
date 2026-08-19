//! Shared tuning-profile facade for the build script and host-side autotuner.
//!
//! The implementation is divided by responsibility under `tuning/`: the
//! schema owns the typed record, `defaults` owns architecture policy, and
//! `source` owns generated-profile I/O. This file intentionally remains module
//! plumbing so both direct `#[path]` consumers use the same narrow surface.

#[path = "tuning/defaults.rs"]
mod defaults;
#[path = "tuning/schema.rs"]
mod schema;
#[path = "tuning/source.rs"]
mod source;
#[path = "tuning/validation.rs"]
mod validation;

#[cfg(test)]
#[path = "tuning/tests.rs"]
mod tests;

pub use defaults::profile_for_target;
pub use schema::TuningProfile;
pub use source::missing_definition;
use validation::{valid_finite, valid_optional_crossover, valid_threshold_chain};
