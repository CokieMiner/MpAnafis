//! Search-local memoization and bounded reuse of SSA geometries and plans.

#![expect(
    unsafe_code,
    reason = "Modulo-bounded replacement cursors and power-of-two ring exponents index fixed initialized cache arrays"
)]

#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use std::{
    sync::{OnceLock, RwLock},
    thread_local,
};

#[cfg(feature = "std")]
use super::LIMB_BITS;
use super::{Geometry, SsaOperation, SsaPlan};

/// Fixed storage bounds pricing stack use independently of operand width.
const COST_MEMO_SLOTS: usize = 64;
/// Retention budget per immutable plan type and operation.
#[cfg(feature = "std")]
const RETAINED_PLAN_SLOTS: usize = 4;

#[derive(Clone, Copy)]
struct CostEntry {
    bits: usize,
    depth: u32,
    operation: SsaOperation,
    cost: usize,
}

/// Search-local costs keyed by every state that affects the recurrence.
/// Eviction changes recomputation only; it never changes candidate costs.
pub struct CostMemo {
    entries: [Option<CostEntry>; COST_MEMO_SLOTS],
    next: usize,
}

/// Four recently constructed immutable plans, keyed by their exact ring width.
/// Each owner supplies one plan type and operation; recursive construction
/// happens outside the borrow so descendants may consult the same cache.
#[cfg(feature = "std")]
pub struct RetainedPlanCache<T> {
    entries: [Option<(usize, T)>; RETAINED_PLAN_SLOTS],
    next: usize,
}

/// Set-associative buckets for alignment-derived, non-power-of-two widths.
#[cfg(feature = "std")]
const IRREGULAR_GEOMETRY_CACHE_LEN: usize = 256;
#[cfg(feature = "std")]
const IRREGULAR_GEOMETRY_CACHE_WAYS: usize = 4;

/// Three operation slots for every representable power-of-two ring width.
#[cfg(feature = "std")]
const POWER_OF_TWO_GEOMETRY_CACHE_LEN: usize = LIMB_BITS * SsaOperation::ALL.len();

#[cfg(feature = "std")]
#[derive(Clone, Copy)]
struct CachedGeometry {
    modulus_bits: usize,
    operation: SsaOperation,
    geometry: Geometry,
}

#[cfg(feature = "std")]
static POWER_OF_TWO_GEOMETRY_CACHE: [OnceLock<CachedGeometry>; POWER_OF_TWO_GEOMETRY_CACHE_LEN] =
    [const { OnceLock::new() }; POWER_OF_TWO_GEOMETRY_CACHE_LEN];

#[cfg(feature = "std")]
struct IrregularGeometryCacheSet {
    state: RwLock<IrregularGeometryState>,
}

#[cfg(feature = "std")]
struct IrregularGeometryState {
    entries: [Option<CachedGeometry>; IRREGULAR_GEOMETRY_CACHE_WAYS],
    next_victim: usize,
}

#[cfg(feature = "std")]
impl IrregularGeometryCacheSet {
    const fn new() -> Self {
        Self {
            state: RwLock::new(IrregularGeometryState {
                entries: [None; IRREGULAR_GEOMETRY_CACHE_WAYS],
                next_victim: 0,
            }),
        }
    }

    fn get(&self, modulus_bits: usize, operation: SsaOperation) -> Option<Geometry> {
        let state = match self.state.read() {
            Ok(entries) => entries,
            Err(poisoned) => poisoned.into_inner(),
        };
        state
            .entries
            .iter()
            .flatten()
            .find(|entry| entry.modulus_bits == modulus_bits && entry.operation == operation)
            .map(|entry| entry.geometry)
    }

    fn insert(&self, entry: CachedGeometry) {
        let mut state = match self.state.write() {
            Ok(entries) => entries,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.entries.iter().flatten().any(|cached| {
            cached.modulus_bits == entry.modulus_bits && cached.operation == entry.operation
        }) {
            return;
        }
        if let Some(empty) = state.entries.iter_mut().find(|cached| cached.is_none()) {
            *empty = Some(entry);
            return;
        }
        let victim = state.next_victim;
        // SAFETY: victim is in 0..IRREGULAR_GEOMETRY_CACHE_WAYS (four slots),
        // so victim+1 fits; only the cache-way reduction is modular.
        state.next_victim = unsafe { victim.unchecked_add(1) } % IRREGULAR_GEOMETRY_CACHE_WAYS;
        debug_assert!(
            victim < state.entries.len(),
            "eviction cursor exceeds cache ways"
        );
        // SAFETY: next_victim starts at zero and is updated only modulo
        // IRREGULAR_GEOMETRY_CACHE_WAYS, the fixed nonzero length of entries.
        unsafe {
            *state.entries.get_unchecked_mut(victim) = Some(entry);
        }
    }
}

#[cfg(feature = "std")]
static IRREGULAR_GEOMETRY_CACHE: [IrregularGeometryCacheSet; IRREGULAR_GEOMETRY_CACHE_LEN] =
    [const { IrregularGeometryCacheSet::new() }; IRREGULAR_GEOMETRY_CACHE_LEN];

/// Per-thread result slots keyed by full required product bits and operation.
/// The complete key and width share one borrow; replacement cannot mix entries.
#[cfg(feature = "std")]
const CRT_WIDTH_CACHE_WAYS: usize = 64;

#[cfg(feature = "std")]
#[derive(Clone, Copy)]
struct CachedCrtWidth {
    required_bits: usize,
    operation: SsaOperation,
    width: usize,
}
#[cfg(feature = "std")]
struct CrtWidthCache {
    entries: [Option<CachedCrtWidth>; CRT_WIDTH_CACHE_WAYS],
    next_victim: usize,
}

#[cfg(feature = "std")]
impl CrtWidthCache {
    const fn new() -> Self {
        Self {
            entries: [None; CRT_WIDTH_CACHE_WAYS],
            next_victim: 0,
        }
    }

    fn get(&self, required_bits: usize, operation: SsaOperation) -> Option<usize> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| entry.required_bits == required_bits && entry.operation == operation)
            .map(|entry| entry.width)
    }

    fn insert(&mut self, entry: CachedCrtWidth) {
        if self.entries.iter().flatten().any(|cached| {
            cached.required_bits == entry.required_bits && cached.operation == entry.operation
        }) {
            return;
        }
        let victim = self.next_victim;
        // SAFETY: victim is below the fixed cache-way count, so victim+1
        // fits every target. The remainder maintains the bounded cursor.
        self.next_victim = unsafe { victim.unchecked_add(1) } % CRT_WIDTH_CACHE_WAYS;
        // SAFETY: next_victim starts at zero and advances only modulo the
        // nonzero array length. The exclusive borrow owns the initialized slot.
        unsafe {
            *self.entries.get_unchecked_mut(victim) = Some(entry);
        }
    }
}

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static CRT_WIDTH_CACHE: RefCell<CrtWidthCache> =
        const { RefCell::new(CrtWidthCache::new()) };
}

// OS-key TLS initializes the same cache lazily through RefCell::from.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static CRT_WIDTH_CACHE: RefCell<CrtWidthCache> =
        RefCell::from(CrtWidthCache::new());
}

#[cfg(feature = "std")]
const fn irregular_geometry_slot(modulus_bits: usize) -> usize {
    let mut remaining = modulus_bits.wrapping_div(LIMB_BITS);
    let mut folded = 0;
    while remaining != 0 {
        folded ^= remaining & 0xff;
        remaining = remaining.wrapping_shr(8);
    }
    folded % IRREGULAR_GEOMETRY_CACHE_LEN
}

#[cfg(feature = "std")]
fn power_of_two_geometry_slot(modulus_bits: usize, operation: SsaOperation) -> usize {
    debug_assert!(
        modulus_bits.is_power_of_two(),
        "the caller selects the power-of-two cache"
    );
    let operation_slot = match operation {
        SsaOperation::Multiply => 0,
        SsaOperation::Square => 1,
        SsaOperation::Pair => 2,
    };
    #[expect(
        clippy::as_conversions,
        reason = "a power-of-two usize has a trailing-zero count below usize::BITS, which fits every pointer width"
    )]
    let transform_log = modulus_bits.trailing_zeros() as usize;
    // SAFETY: a nonzero power of two has exponent below usize::BITS. That
    // count fits usize on every pointer width, and 3*exponent + operation_slot
    // is below 3*usize::BITS, the fixed power-of-two cache length.
    unsafe {
        transform_log
            .unchecked_mul(SsaOperation::ALL.len())
            .unchecked_add(operation_slot)
    }
}

impl Geometry {
    /// Reads the planner cache on `std` builds.
    #[cfg(feature = "std")]
    pub fn cached(modulus_bits: usize, operation: SsaOperation) -> Option<Self> {
        if modulus_bits.is_power_of_two() {
            let slot = power_of_two_geometry_slot(modulus_bits, operation);
            // SAFETY: slot < 3*usize::BITS by its construction above. The
            // static array has exactly that many initialized OnceLock entries.
            let entry = unsafe { POWER_OF_TWO_GEOMETRY_CACHE.get_unchecked(slot) };
            return entry.get().map(|cached| cached.geometry);
        }
        let slot = irregular_geometry_slot(modulus_bits);
        // SAFETY: the index is reduced modulo the nonzero static array length.
        unsafe { IRREGULAR_GEOMETRY_CACHE.get_unchecked(slot) }.get(modulus_bits, operation)
    }

    /// Recomputes plans on `no_std` builds, which have no synchronization cache.
    #[cfg(not(feature = "std"))]
    pub const fn cached(_modulus_bits: usize, _operation: SsaOperation) -> Option<Self> {
        None
    }

    /// Stores this deterministic geometry in the cache.
    #[cfg(feature = "std")]
    pub fn cache(&self, modulus_bits: usize, operation: SsaOperation) {
        let entry = CachedGeometry {
            modulus_bits,
            operation,
            geometry: *self,
        };
        if modulus_bits.is_power_of_two() {
            let index = power_of_two_geometry_slot(modulus_bits, operation);
            // SAFETY: the admitted exponent and operation index an initialized
            // entry strictly below the fixed 3*usize::BITS array length.
            let slot = unsafe { POWER_OF_TWO_GEOMETRY_CACHE.get_unchecked(index) };
            // Racing writers compute the same pure result, so the incumbent stays.
            let _published = slot.set(entry);
        } else {
            let index = irregular_geometry_slot(modulus_bits);
            // SAFETY: the modulo-reduced index lies in the initialized array.
            let set = unsafe { IRREGULAR_GEOMETRY_CACHE.get_unchecked(index) };
            set.insert(entry);
        }
    }
}

impl SsaPlan {
    /// Reads the CRT half-width result cache on `std` builds.
    ///
    /// The cached width is the complete search result for these exact required
    /// product bits under this operation, not a single candidate.
    #[cfg(feature = "std")]
    pub fn cached_crt_half_width(required_bits: usize, operation: SsaOperation) -> Option<usize> {
        CRT_WIDTH_CACHE.with(|cache| cache.borrow().get(required_bits, operation))
    }

    /// Recomputes half-width searches on `no_std` builds, which have no
    /// synchronization cache.
    #[cfg(not(feature = "std"))]
    pub const fn cached_crt_half_width(
        _required_bits: usize,
        _operation: SsaOperation,
    ) -> Option<usize> {
        None
    }

    /// Stores this deterministic search result in the cache.
    #[cfg(feature = "std")]
    pub fn cache_crt_half_width(required_bits: usize, operation: SsaOperation, width: usize) {
        CRT_WIDTH_CACHE.with(|cache| {
            cache.borrow_mut().insert(CachedCrtWidth {
                required_bits,
                operation,
                width,
            });
        });
    }

    /// No-op store on `no_std` builds.
    #[cfg(not(feature = "std"))]
    pub const fn cache_crt_half_width(
        _required_bits: usize,
        _operation: SsaOperation,
        _width: usize,
    ) {
    }
}

impl CostMemo {
    pub const fn new() -> Self {
        Self {
            entries: [None; COST_MEMO_SLOTS],
            next: 0,
        }
    }

    pub fn get(&self, bits: usize, depth: u32, operation: SsaOperation) -> Option<usize> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| {
                entry.bits == bits && entry.depth == depth && entry.operation == operation
            })
            .map(|entry| entry.cost)
    }

    pub fn insert(&mut self, bits: usize, depth: u32, operation: SsaOperation, cost: usize) {
        // SAFETY: next starts at zero and advances only modulo the nonzero
        // slot count. The exclusive borrow owns this initialized entry.
        unsafe {
            *self.entries.get_unchecked_mut(self.next) = Some(CostEntry {
                bits,
                depth,
                operation,
                cost,
            });
        }
        // SAFETY: next<COST_MEMO_SLOTS=64, so its successor fits even u16.
        self.next = unsafe { self.next.unchecked_add(1) } % COST_MEMO_SLOTS;
    }
}

#[cfg(feature = "std")]
impl<T: Clone> RetainedPlanCache<T> {
    /// Creates an empty cache with four fixed slots.
    pub const fn new() -> Self {
        Self {
            entries: [const { None }; RETAINED_PLAN_SLOTS],
            next: 0,
        }
    }

    /// Clones the shared handle without rebuilding its immutable descendants.
    pub fn get(&self, bits: usize) -> Option<T> {
        self.entries
            .iter()
            .flatten()
            .find(|(key, _)| *key == bits)
            .map(|(_, value)| value.clone())
    }

    /// Publishes one completed plan, replacing the oldest inserted slot.
    pub fn insert(&mut self, bits: usize, value: T) {
        let slot = self.next % RETAINED_PLAN_SLOTS;
        // SAFETY: slot is below the fixed retention budget, so slot+1 fits.
        self.next = unsafe { slot.unchecked_add(1) };
        // SAFETY: the remainder is below RETAINED_PLAN_SLOTS, exactly the
        // initialized array's bounds; the exclusive borrow owns this entry.
        unsafe {
            *self.entries.get_unchecked_mut(slot) = Some((bits, value));
        }
    }
}
