//! The two public integer types and their debug wrapper.
//!
//! Both are a pairing of an internal magnitude with [`Precision`] metadata. The
//! fields are crate-visible rather than public: every invariant that makes a
//! value valid — normalisation, the no-negative-zero rule, fitting the declared
//! precision — is enforced by the constructors and operations in this module
//! tree, and direct field access would bypass all of them.

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{InternalArbiInt, InternalArbiUint, Precision};

/// Arbitrary precision signed integer.
pub struct ArbiInt {
    /// The internal signed representation.
    pub(crate) value: InternalArbiInt,
    /// The precision metadata of the value.
    pub(crate) precision: Precision,
}

impl Clone for ArbiInt {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            precision: self.precision,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value.clone_from(&source.value);
        self.precision = source.precision;
    }
}

/// Arbitrary precision unsigned integer.
pub struct ArbiUint {
    /// The internal unsigned representation.
    pub(crate) value: InternalArbiUint,
    /// The precision metadata of the value.
    pub(crate) precision: Precision,
}

impl Clone for ArbiUint {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            precision: self.precision,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value.clone_from(&source.value);
        self.precision = source.precision;
    }
}

/// A wrapper for verbose debug formatting.
#[non_exhaustive]
pub struct DebugVerbose<'data, T>(pub &'data T);

impl Debug for DebugVerbose<'_, ArbiUint> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ArbiUint({}, precision: {:?})", self.0, self.0.precision)
    }
}
