//! Public integer types and their API-facing trait implementations.

use super::{InternalMpInt, InternalMpUint, InternalPrecisionContext};
mod cmp;
mod convert;
mod iter;
mod ops;
mod precision;
mod string;
mod types;
mod validation;

#[cfg(feature = "num-traits")]
mod num_traits;

mod int;
mod uint;

pub use precision::{AmbientPrecision, BoundedPrecision, Precision, PrecisionContext};
pub use types::{DebugVerbose, MpInt, MpUint};
