//! Shared tuning-profile facade for the build script and host-side autotuner.
//!
//! `schema` defines the record and semantic contracts, `defaults` selects built-in
//! values, and `source` parses and renders generated profiles. Both `#[path]`
//! consumers use this facade.

mod defaults;
mod parameters;
mod schema;
mod source;
mod validation;

pub use parameters::Parameter;
pub use schema::TuningProfile;
pub use validation::Validation;

#[cfg(test)]
mod tests;
