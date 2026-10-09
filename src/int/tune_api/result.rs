//! Owned results exposing canonical limbs without exposing the internal integer API.

use super::{InternalMpUint, Limb};

/// Owns a kernel result and exposes its canonical limb slice.
#[derive(Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuningResult(InternalMpUint);

impl TuningResult {
    /// Transfers an internal result into the opaque tuning result.
    pub(crate) const fn new(value: InternalMpUint) -> Self {
        Self(value)
    }
}

impl AsRef<[Limb]> for TuningResult {
    fn as_ref(&self) -> &[Limb] {
        self.0.limbs()
    }
}
