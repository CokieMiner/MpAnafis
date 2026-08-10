//! Property-based integration tests for `ArbiUint` and `ArbiInt`.
//!
//! Each module owns one behavioral category. Shared constructors and strategies
//! remain here so individual property files stay focused on their contracts.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Test files: arithmetic operators are the API under test; casts use property-bounded values"
)]

extern crate std;

use alloc::{string::ToString, vec::Vec};
use core::cmp::Ordering;
#[cfg(feature = "std")]
use core::hash::{Hash, Hasher};
#[cfg(feature = "std")]
use std::collections::hash_map::DefaultHasher;

use proptest::prelude::*;
#[cfg(feature = "std")]
use support::hash_u64;
#[cfg(feature = "num-traits")]
use support::int_from_i64;
use support::{exact_limb_vec, nz, uint};

use super::{api::*, logic::InternalArbiUint};
use crate::{error::ArbiError, int::types::Limb};

mod add;
mod api_surface;
mod arithmetic;
mod bitwise;
mod bounded;
mod comparison;
mod conversions;
mod fused;
mod gcd;
mod memory;
mod modular;
mod mul;
mod num_traits;
mod ops;
mod powers;
mod precision;
mod primality;
mod scan;
mod strategies;
mod stress;
mod string;
mod support;
mod theory;
mod traits;
