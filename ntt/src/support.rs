//! Standalone limb types and synchronous execution adapters.
//!
//! These definitions follow `src/int/types.rs` and `src/parallel/api.rs` in
//! `MpAnafis`. Worker budgets describe available concurrency; they do not reserve
//! threads. A budget of one executes all work on the calling thread.

use core::num::NonZeroUsize;
#[cfg(feature = "rayon")]
use std::{env::var_os, sync::OnceLock, thread::available_parallelism};
#[cfg(all(feature = "rayon", target_os = "linux"))]
use std::{
    fs::{read_dir, read_to_string},
    path::Path,
};

#[cfg(feature = "rayon")]
use rayon::{ThreadPoolBuilder, current_num_threads, current_thread_index, join};

/// Native unsigned limb, identical to the `MpAnafis` representation.
pub type Limb = usize;
/// Number of bits in a native limb.
#[expect(
    clippy::as_conversions,
    reason = "Limb::BITS is at most 64 and fits usize on every supported pointer width"
)]
pub const LIMB_BITS: usize = Limb::BITS as usize;

/// Schedules independent synchronous arithmetic work.
pub trait ParallelExecutor: Sync {
    /// Returns the nonzero planning budget for this operation.
    fn parallelism(&self) -> NonZeroUsize;

    /// Executes two closures and returns after all started work has stopped.
    ///
    /// # Panics
    /// Propagates closure panics. Sequential execution does not start `right`
    /// when `left` panics.
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;
}

/// Resolves the default execution backend for one complete operation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct DefaultExecutor;

/// Executes closures sequentially on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SequentialExecutor;

/// Executes closures in the active Rayon pool.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct RayonExecutor;

/// Borrows an executor with a fixed nonzero planning budget.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug)]
pub struct FixedParallelismExecutor<'executor, E: ?Sized> {
    executor: &'executor E,
    parallelism: NonZeroUsize,
}

impl DefaultExecutor {
    /// Resolves the active pool, preserving explicit Rayon configuration.
    #[cfg(feature = "rayon")]
    pub fn with_resolved<R>(
        action: impl FnOnce(&FixedParallelismExecutor<'_, RayonExecutor>) -> R,
    ) -> R {
        static GLOBAL_PARALLELISM: OnceLock<NonZeroUsize> = OnceLock::new();
        let executor = RayonExecutor;
        let parallelism = if current_thread_index().is_some() {
            executor.parallelism()
        } else {
            *GLOBAL_PARALLELISM.get_or_init(|| {
                narrow_default_pool();
                executor.parallelism()
            })
        };
        let resolved = FixedParallelismExecutor::new(&executor, parallelism);
        action(&resolved)
    }

    /// Resolves sequential execution when Rayon is disabled.
    #[cfg(not(feature = "rayon"))]
    pub fn with_resolved<R>(action: impl FnOnce(&SequentialExecutor) -> R) -> R {
        action(&SequentialExecutor)
    }
}

#[cfg(feature = "rayon")]
impl<'executor, E: ParallelExecutor + ?Sized> FixedParallelismExecutor<'executor, E> {
    /// Borrows `executor` with the budget used by prepared arithmetic.
    #[must_use]
    pub const fn new(executor: &'executor E, parallelism: NonZeroUsize) -> Self {
        Self {
            executor,
            parallelism,
        }
    }
}

impl ParallelExecutor for SequentialExecutor {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (left(), right())
    }
}

#[cfg(feature = "rayon")]
impl ParallelExecutor for RayonExecutor {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(current_num_threads()).unwrap_or(NonZeroUsize::MIN)
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        join(left, right)
    }
}

#[cfg(feature = "rayon")]
impl<E: ParallelExecutor + ?Sized> ParallelExecutor for FixedParallelismExecutor<'_, E> {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        self.parallelism
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        if self.parallelism.get() == 1 {
            (left(), right())
        } else {
            self.executor.join(left, right)
        }
    }
}

#[cfg(feature = "rayon")]
fn narrow_default_pool() {
    static ATTEMPTED: OnceLock<()> = OnceLock::new();
    let _decision = ATTEMPTED.get_or_init(|| {
        if var_os("RAYON_NUM_THREADS").is_some() {
            return;
        }
        let Ok(available) = available_parallelism() else {
            return;
        };
        #[cfg(target_os = "linux")]
        let detected = detect_physical_parallelism(Path::new("/sys/devices/system/cpu"));
        #[cfg(not(target_os = "linux"))]
        let detected: Option<NonZeroUsize> = None;
        let Some(physical) = detected else {
            return;
        };
        if physical >= available {
            return;
        }
        let _initialized = ThreadPoolBuilder::new()
            .num_threads(physical.get())
            .build_global()
            .is_ok();
    });
}

#[cfg(all(feature = "rayon", target_os = "linux"))]
fn detect_physical_parallelism(root: &Path) -> Option<NonZeroUsize> {
    let mut cores = 0_usize;
    for entry_result in read_dir(root).ok()? {
        let entry = entry_result.ok()?;
        let name = entry.file_name();
        let Some(digits) = name.to_str().and_then(|text| text.strip_prefix("cpu")) else {
            continue;
        };
        if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
            continue;
        }
        let index = digits.parse::<usize>().ok()?;
        let list = read_to_string(entry.path().join("topology/thread_siblings_list")).ok()?;
        let primary = list
            .split(',')
            .next()
            .and_then(|first| first.split('-').next())
            .and_then(|first| first.trim().parse::<usize>().ok())?;
        if primary == index {
            cores = cores.checked_add(1)?;
        }
    }
    NonZeroUsize::new(cores)
}
