//! Inherent `MpInt` API implementations by category.

use super::{
    BoundedPrecision, DebugVerbose, InternalMpInt, InternalMpUint, MpInt, MpUint, Precision,
    PrecisionContext,
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
