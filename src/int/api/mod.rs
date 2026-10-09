//! Public integer types and their API-facing trait implementations.

use super::{InternalMpInt, InternalMpUint, InternalPrecisionContext};

mod precision;
mod types;

pub use precision::{AmbientPrecision, BoundedPrecision, Precision, PrecisionContext};
pub use types::{DebugVerbose, MpInt, MpUint};
