//! Precision metadata and the ambient precision context.
//!
//! Precision is carried alongside every value rather than baked into its type:
//! [`Precision`] says what a value *is*, and [`AmbientPrecision`] says what
//! newly constructed values *default to*. The two are deliberately separate
//! enums because ambient precision has a third state, [`AmbientPrecision::Unset`],
//! that no value can ever hold.

use core::{
    fmt::{Debug, Formatter, Result as FmtResult},
    hash::Hash,
    num::NonZeroUsize,
};

use super::InternalPrecisionContext;

/// A validated non-zero bit width for bounded integer precision.
///
/// Valid widths are `1..usize::MAX`. The top `usize` value is reserved by the
/// ambient-precision encoding for [`AmbientPrecision::Unlimited`], so keeping
/// it out of this type makes every bounded precision representable without an
/// ambiguous sentinel value.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct BoundedPrecision(NonZeroUsize);

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

/// Defines the precision of an arbitrary precision integer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// Unlimited precision, growing automatically as needed.
    Unlimited,
    /// Bounded precision, acting strictly as an N-bit integer.
    Bounded(BoundedPrecision),
}

impl Precision {
    /// Check if the precision is unlimited.
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
        match BoundedPrecision::new(bits) {
            Some(width) => Some(Self::Bounded(width)),
            None => None,
        }
    }

    /// Derives the result precision for a non-assigning binary operation.
    ///
    /// - Bounded + Bounded → `Bounded(max(w_lhs, w_rhs))`
    /// - Bounded + Unlimited → `Unlimited`
    /// - Unlimited + Unlimited → `Unlimited`
    #[must_use]
    pub(crate) const fn combine_for_binary_op(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Bounded(a), Self::Bounded(b)) => {
                let max_bits = if a.get() >= b.get() { a } else { b };
                Self::Bounded(max_bits)
            }
            _ => Self::Unlimited,
        }
    }

    /// Resolves ambient precision for a `From<T>` constructor from the number
    /// of bits its value requires.
    ///
    /// Returns `max(ambient_width, required_bits)` for bounded ambient,
    /// or `Unlimited` otherwise.
    #[must_use]
    pub(crate) fn for_ambient_construction(required_bits: usize) -> Self {
        let ambient = Self::from(PrecisionContext::active());
        match ambient {
            Self::Bounded(n) => {
                let actual = if required_bits > n.get() {
                    required_bits
                } else {
                    n.get()
                };
                Self::new_bounded(actual).unwrap_or(Self::Unlimited)
            }
            Self::Unlimited => Self::Unlimited,
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

/// Represents the ambient precision context.
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

/// A context manager for precision.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PrecisionContext;

impl PrecisionContext {
    /// Returns the active ambient precision.
    ///
    /// On targets without pointer-width atomics there is no global default, so
    /// this reports `Unset` unless a scoped context is active.
    ///
    /// This `const` variant mirrors the const `InternalPrecisionContext::active`
    /// on targets without pointer-width atomics (and without `std`), where the
    /// ambient lookup is a constant.
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
    /// Only available on targets with pointer-width atomics: those without
    /// them cannot safely share mutable global state, so no global ambient
    /// default exists and [`PrecisionContext::active`] reports `Unset` unless
    /// a scoped [`PrecisionContext::with_bounded`] or
    /// [`PrecisionContext::with_unlimited`] context is active.
    #[must_use]
    #[cfg(target_has_atomic = "ptr")]
    pub fn set_global(precision: AmbientPrecision) -> AmbientPrecision {
        InternalPrecisionContext::set_global(precision)
    }

    /// Execute the closure `f` with a scoped ambient bounded precision of `bits`.
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

    /// Execute the closure `f` with a scoped ambient unlimited precision.
    #[cfg(feature = "std")]
    pub fn with_unlimited<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        InternalPrecisionContext::with_unlimited(f)
    }
}
