//! Ambient precision resolution and scoped precision context internals.

#[cfg(feature = "std")]
use core::cell::Cell;
#[cfg(target_has_atomic = "ptr")]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::thread_local;

use super::{AmbientPrecision, BoundedPrecision};

const UNSET_SENTINEL: usize = 0;
const UNLIMITED_SENTINEL: usize = usize::MAX;

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static THREAD_PRECISION: Cell<AmbientPrecision> = const { Cell::new(AmbientPrecision::Unset) };
}

// OS-key TLS cannot eagerly materialize a const value. `Cell::from` keeps that
// necessarily lazy initializer distinct from the const-capable branch and
// produces the identical initial value without suppressing Clippy.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static THREAD_PRECISION: Cell<AmbientPrecision> = Cell::from(AmbientPrecision::Unset);
}

#[cfg(target_has_atomic = "ptr")]
static GLOBAL_PRECISION: AtomicUsize = AtomicUsize::new(UNSET_SENTINEL);

#[cfg(target_has_atomic = "ptr")]
fn load_global() -> usize {
    GLOBAL_PRECISION.load(Ordering::Relaxed)
}

#[cfg(target_has_atomic = "ptr")]
fn swap_global(val: usize) -> usize {
    GLOBAL_PRECISION.swap(val, Ordering::Relaxed)
}

// Targets without pointer-width atomics cannot safely host a mutable global
// default: the absence of `target_has_atomic = "ptr"` does not imply the
// absence of concurrency (interrupt/preemption contexts on single-core MCUs
// race with the main context). No `Sync` claim on an `UnsafeCell` is sound
// there, so these targets simply have no global storage: `active()` reports
// `Unset` and `set_global` is not available (both are gated on
// `target_has_atomic = "ptr"`).
#[cfg(not(target_has_atomic = "ptr"))]
const fn load_global() -> usize {
    UNSET_SENTINEL
}

/// Scoped context for ambient precision.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct InternalPrecisionContext;

impl InternalPrecisionContext {
    /// Get the currently active ambient precision.
    ///
    /// Order of precedence:
    /// 1. Thread-local scoped context (if `std` is active).
    /// 2. Global precision default (targets with pointer-width atomics only).
    /// 3. `AmbientPrecision::Unset`.
    ///
    /// This `const` variant exists for targets without pointer-width atomics
    /// (and without `std`): they cannot share mutable global state, so the
    /// body reduces to a constant lookup through the const `load_global`
    /// fallback.
    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    #[must_use]
    pub const fn active() -> AmbientPrecision {
        decode_precision(load_global())
    }

    /// Get the currently active ambient precision.
    ///
    /// Order of precedence:
    /// 1. Thread-local scoped context (if `std` is active).
    /// 2. Global precision default (targets with pointer-width atomics only).
    /// 3. `AmbientPrecision::Unset`.
    #[cfg(not(all(not(feature = "std"), not(target_has_atomic = "ptr"))))]
    #[must_use]
    pub fn active() -> AmbientPrecision {
        #[cfg(feature = "std")]
        {
            let local = THREAD_PRECISION.with(Cell::get);
            if local != AmbientPrecision::Unset {
                return local;
            }
        }

        decode_precision(load_global())
    }

    /// Set the global default ambient precision for the application.
    ///
    /// Available only on targets with pointer-width atomics. Targets without
    /// them cannot safely share mutable global state, so they have no global
    /// default: [`InternalPrecisionContext::active`] reports `Unset` unless a
    /// scoped `std` context is active.
    #[cfg(target_has_atomic = "ptr")]
    pub fn set_global(precision: AmbientPrecision) -> AmbientPrecision {
        decode_precision(swap_global(encode_precision(precision)))
    }

    /// Execute the closure `f` with a scoped ambient bounded precision of `bits`.
    ///
    /// Available only when the `std` feature is enabled.
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
        with_precision(AmbientPrecision::Bounded(width), f)
    }

    /// Execute the closure `f` with a scoped ambient unlimited precision.
    ///
    /// Available only when the `std` feature is enabled.
    #[cfg(feature = "std")]
    pub fn with_unlimited<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        with_precision(AmbientPrecision::Unlimited, f)
    }
}

// --- Internal helpers ---

#[inline]
#[must_use]
#[cfg(target_has_atomic = "ptr")]
const fn encode_precision(precision: AmbientPrecision) -> usize {
    match precision {
        AmbientPrecision::Unset => UNSET_SENTINEL,
        AmbientPrecision::Unlimited => UNLIMITED_SENTINEL,
        AmbientPrecision::Bounded(bits) => bits.get(),
    }
}

#[inline]
#[must_use]
const fn decode_precision(value: usize) -> AmbientPrecision {
    if value == UNSET_SENTINEL {
        AmbientPrecision::Unset
    } else if value == UNLIMITED_SENTINEL {
        AmbientPrecision::Unlimited
    } else {
        let width = BoundedPrecision::new(value)
            .expect("sentinel checks prove the encoded bounded width is valid");
        AmbientPrecision::Bounded(width)
    }
}

#[cfg(feature = "std")]
fn with_precision<F, R>(p: AmbientPrecision, f: F) -> R
where
    F: FnOnce() -> R,
{
    struct Guard(AmbientPrecision);
    impl Drop for Guard {
        fn drop(&mut self) {
            THREAD_PRECISION.with(|c| c.set(self.0));
        }
    }

    let prev = THREAD_PRECISION.with(|c| {
        let old = c.get();
        c.set(p);
        old
    });

    let _guard = Guard(prev);

    f()
}

#[cfg(test)]
#[path = "tests/precision.rs"]
mod tests;
