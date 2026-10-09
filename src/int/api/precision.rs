//! Precision metadata and the ambient precision context.
//!
//! [`Precision`] describes an integer's bounded or unlimited width.
//! [`AmbientPrecision`] also represents an unset construction context.

use core::{
    fmt::{Debug, Formatter, Result as FmtResult},
    hash::Hash,
    num::NonZeroUsize,
};

use super::InternalPrecisionContext;

/// A validated non-zero bit width for bounded integer precision.
///
/// Valid widths are `1..usize::MAX`. The ambient encoding reserves zero for
/// [`AmbientPrecision::Unset`] and `usize::MAX` for [`AmbientPrecision::Unlimited`].
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct BoundedPrecision(NonZeroUsize);

/// Precision policy carried by an arbitrary precision integer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// Unlimited precision, growing automatically as needed.
    Unlimited,
    /// Bounded precision, acting strictly as an N-bit integer.
    Bounded(BoundedPrecision),
}

/// Precision policy applied during construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum AmbientPrecision {
    /// No ambient precision is set. Construction produces `Unlimited`.
    Unset,
    /// Ambient precision is explicitly set to unlimited.
    Unlimited,
    /// Ambient precision is set to a bounded bit width.
    Bounded(BoundedPrecision),
}

/// Access to global and scoped ambient precision.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PrecisionContext;

impl BoundedPrecision {
    /// Creates a bounded bit width.
    ///
    /// Returns `None` when `bits` is zero or `usize::MAX`.
    #[must_use]
    pub const fn new(bits: usize) -> Option<Self> {
        let Some(nonzero_bits) = NonZeroUsize::new(bits) else {
            return None;
        };
        if bits == usize::MAX {
            None
        } else {
            Some(Self(nonzero_bits))
        }
    }

    /// Returns the validated bit width as a `usize`.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl Debug for BoundedPrecision {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.get(), formatter)
    }
}

impl Precision {
    /// Returns `true` if the precision is unlimited.
    #[must_use]
    pub const fn is_unlimited(self) -> bool {
        matches!(self, Self::Unlimited)
    }

    /// Returns the explicit bit width if bounded, otherwise `None`.
    #[must_use]
    pub const fn significant_bits(self) -> Option<usize> {
        match self {
            Self::Unlimited => None,
            Self::Bounded(n) => Some(n.get()),
        }
    }

    /// Creates a `Bounded` precision, returning `None` if `bits` is 0 or `usize::MAX`.
    #[must_use]
    pub const fn new_bounded(bits: usize) -> Option<Self> {
        if let Some(width) = BoundedPrecision::new(bits) {
            Some(Self::Bounded(width))
        } else {
            None
        }
    }
}

impl From<AmbientPrecision> for Precision {
    fn from(ambient: AmbientPrecision) -> Self {
        match ambient {
            AmbientPrecision::Unset | AmbientPrecision::Unlimited => Self::Unlimited,
            AmbientPrecision::Bounded(n) => Self::Bounded(n),
        }
    }
}

impl AmbientPrecision {
    /// Creates a bounded ambient precision.
    ///
    /// Returns `None` when `bits` is zero or `usize::MAX`.
    #[must_use]
    pub const fn new_bounded(bits: usize) -> Option<Self> {
        match BoundedPrecision::new(bits) {
            Some(width) => Some(Self::Bounded(width)),
            None => None,
        }
    }
}

impl PrecisionContext {
    /// Returns the active ambient precision.
    ///
    /// Returns `Unset` on `no_std` targets without pointer-width atomics.
    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    #[must_use]
    pub const fn active() -> AmbientPrecision {
        InternalPrecisionContext::active()
    }

    /// Returns the active ambient precision.
    ///
    /// On targets without pointer-width atomics there is no global default, so
    /// this reports `Unset` unless a scoped context is active.
    #[cfg(not(all(not(feature = "std"), not(target_has_atomic = "ptr"))))]
    #[must_use]
    pub fn active() -> AmbientPrecision {
        InternalPrecisionContext::active()
    }

    /// Sets the global ambient precision and returns the previous value.
    ///
    /// Available on targets with pointer-width atomics. Scoped contexts override
    /// this default when the `std` feature is enabled.
    #[must_use]
    #[cfg(target_has_atomic = "ptr")]
    pub fn set_global(precision: AmbientPrecision) -> AmbientPrecision {
        InternalPrecisionContext::set_global(precision)
    }

    /// Executes the closure `f` with a scoped ambient bounded precision of `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is zero or `usize::MAX`.
    #[cfg(feature = "std")]
    pub fn with_bounded<F, R>(bits: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        InternalPrecisionContext::with_bounded(bits, f)
    }

    /// Executes the closure `f` with a scoped ambient unlimited precision.
    #[cfg(feature = "std")]
    pub fn with_unlimited<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        InternalPrecisionContext::with_unlimited(f)
    }
}
