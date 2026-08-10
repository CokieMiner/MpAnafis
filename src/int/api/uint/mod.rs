//! Inherent `ArbiUint` API implementations by category.

use super::{
    ArbiUint, BoundedPrecision, DebugVerbose, InternalArbiUint, Precision, PrecisionContext,
};

mod bitwise;
mod capacity;
mod constructors;
mod convert;
mod math;
mod properties;
mod shift;
mod theory;
