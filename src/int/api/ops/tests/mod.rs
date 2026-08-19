//! Operator contract properties grouped by behavior.

use support::{bounded_int, bounded_uint, signed_max, signed_min, unsigned_max};

use super::{super::BoundedPrecision, MpInt, MpUint};

mod assignment;
mod overflow;
mod support;
