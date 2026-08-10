//! Inherent `ArbiInt` API implementations by category.

use super::{
    ArbiInt, ArbiUint, BoundedPrecision, DebugVerbose, InternalArbiInt, InternalArbiUint,
    Precision, PrecisionContext,
};

mod bitwise;
mod capacity;
mod constructors;
mod convert;
mod math;
mod properties;
mod shift;
mod sign;
mod theory;
