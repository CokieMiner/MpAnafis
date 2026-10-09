//! Arbitrary-precision signed and unsigned integer arithmetic.
//!
//! [`MpInt`] represents integers over $\mathbb{Z}$; [`MpUint`] represents
//! non-negative integers. Both use native pointer-width limbs, store up to four
//! limbs inline, and use heap storage for larger magnitudes. Values carry bounded
//! or unlimited precision in both `std` and `no_std` environments.
#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    any(target_arch = "mips", target_arch = "mips64"),
    feature(asm_experimental_arch)
)]

extern crate alloc;

mod error;
mod int;
mod parallel;

pub use error::{
    MpError, ParseMpIntError, ParseMpIntErrorKind, ParseMpUintError, ParseMpUintErrorKind,
};

#[cfg(feature = "_internal-tune")]
#[doc(hidden)]
pub use int::tune_api;
pub use int::{
    AmbientPrecision, BoundedPrecision, DebugVerbose, MpInt, MpUint, Precision, PrecisionContext,
};
