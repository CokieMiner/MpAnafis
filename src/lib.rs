//! High-performance arbitrary-precision signed and unsigned integers.
//!
//! Import integer types, precision policies, and errors directly from the crate
//! root. The crate supports `no_std` environments through `alloc`.
#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    any(target_arch = "mips", target_arch = "mips64"),
    feature(asm_experimental_arch)
)]

extern crate alloc;

mod error;
mod int;

pub use error::{MpError, ParseMpIntError, ParseMpUintError};

pub use int::{
    AmbientPrecision, BoundedPrecision, DebugVerbose, MpInt, MpUint, Precision, PrecisionContext,
};

#[cfg(feature = "_internal-tune")]
#[doc(hidden)]
pub use int::tune_api;
