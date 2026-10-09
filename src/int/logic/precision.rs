//! Ambient precision resolution and scoped precision context internals.

#![cfg_attr(
    any(feature = "std", target_has_atomic = "ptr"),
    expect(
        unsafe_code,
        reason = "Sentinel decoding excludes both rejected bounded-precision encodings before construction"
    )
)]

#[cfg(feature = "std")]
use core::cell::Cell;
#[cfg(target_has_atomic = "ptr")]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::thread_local;

use super::AmbientPrecision;
#[cfg(any(feature = "std", target_has_atomic = "ptr"))]
use super::BoundedPrecision;

#[cfg(any(feature = "std", target_has_atomic = "ptr"))]
const UNSET_SENTINEL: usize = 0;
#[cfg(any(feature = "std", target_has_atomic = "ptr"))]
const UNLIMITED_SENTINEL: usize = usize::MAX;

// Encodings reserve zero for Unset and usize::MAX for Unlimited; every bounded
// width lies strictly between them. Each local state occupies one native word.
#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static THREAD_PRECISION: Cell<usize> = const { Cell::new(UNSET_SENTINEL) };
}

#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static THREAD_PRECISION: Cell<usize> = Cell::from(UNSET_SENTINEL);
}

#[cfg(target_has_atomic = "ptr")]
static GLOBAL_PRECISION: AtomicUsize = AtomicUsize::new(UNSET_SENTINEL);

#[cfg(feature = "std")]
struct PrecisionGuard<'scope> {
    cell: &'scope Cell<usize>,
    previous: usize,
}

/// Resolves thread-local and global ambient precision policies.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct InternalPrecisionContext;

impl InternalPrecisionContext {
    /// Returns `Unset` when neither thread-local nor atomic global state is available.
    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    #[must_use]
    pub const fn active() -> AmbientPrecision {
        AmbientPrecision::Unset
    }

    /// Resolves a set thread-local policy, then the global policy, then `Unset`.
    #[cfg(not(all(not(feature = "std"), not(target_has_atomic = "ptr"))))]
    #[must_use]
    pub fn active() -> AmbientPrecision {
        #[cfg(feature = "std")]
        {
            let local = THREAD_PRECISION.with(Cell::get);
            if local != UNSET_SENTINEL {
                return Self::decode_precision(local);
            }
        }

        #[cfg(target_has_atomic = "ptr")]
        return Self::decode_precision(GLOBAL_PRECISION.load(Ordering::Relaxed));
        #[cfg(not(target_has_atomic = "ptr"))]
        AmbientPrecision::Unset
    }

    /// Replaces the global policy and returns its previous value.
    ///
    /// The atomic word contains the complete policy and publishes no other data,
    /// so relaxed ordering is sufficient.
    #[cfg(target_has_atomic = "ptr")]
    pub fn set_global(precision: AmbientPrecision) -> AmbientPrecision {
        Self::decode_precision(
            GLOBAL_PRECISION.swap(Self::encode_precision(precision), Ordering::Relaxed),
        )
    }

    /// Executes `f` under a bounded thread-local policy and restores the prior policy.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is zero or `usize::MAX`.
    #[cfg(feature = "std")]
    pub fn with_bounded<F, R>(bits: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let width =
            BoundedPrecision::new(bits).expect("with_bounded requires bits in 1..usize::MAX");
        Self::with_precision(AmbientPrecision::Bounded(width), f)
    }

    /// Executes `f` under an unlimited thread-local policy and restores the prior policy.
    #[cfg(feature = "std")]
    pub fn with_unlimited<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        Self::with_precision(AmbientPrecision::Unlimited, f)
    }

    #[cfg(feature = "std")]
    fn with_precision<F, R>(precision: AmbientPrecision, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        THREAD_PRECISION.with(|cell| {
            // The guard retains the complete prior encoding inside the TLS
            // borrow and restores it on return or unwinding.
            let _guard = PrecisionGuard {
                cell,
                previous: cell.replace(Self::encode_precision(precision)),
            };
            f()
        })
    }

    #[inline]
    #[must_use]
    #[cfg(any(feature = "std", target_has_atomic = "ptr"))]
    const fn encode_precision(precision: AmbientPrecision) -> usize {
        match precision {
            AmbientPrecision::Unset => UNSET_SENTINEL,
            AmbientPrecision::Unlimited => UNLIMITED_SENTINEL,
            AmbientPrecision::Bounded(bits) => bits.get(),
        }
    }

    #[inline]
    #[must_use]
    #[cfg(any(feature = "std", target_has_atomic = "ptr"))]
    const fn decode_precision(value: usize) -> AmbientPrecision {
        if value == UNSET_SENTINEL {
            AmbientPrecision::Unset
        } else if value == UNLIMITED_SENTINEL {
            AmbientPrecision::Unlimited
        } else {
            // SAFETY: both sentinel branches returned, so 0 < value <
            // usize::MAX. These are exactly the bounded constructor's limits.
            let width = unsafe { BoundedPrecision::new(value).unwrap_unchecked() };
            AmbientPrecision::Bounded(width)
        }
    }
}

#[cfg(feature = "std")]
impl Drop for PrecisionGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}
