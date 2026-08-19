//! Raw bindings to FLINT's integer multiplication entry points.
//!
//! FLINT 3 does not route large products through GMP. `flint_mpn_mul` selects
//! between hand-written fixed-width kernels, its own Toom tier, and the
//! `fft_small` module — an FFT over word-size primes requiring AVX2 or NEON,
//! which is the same family as our own `ntt` path rather than a Schoenhage-
//! Strassen one. That makes FLINT, not GMP, the relevant adversary above the
//! transform crossover.
//!
//! The public `flint_mpn_mul` symbols are linked directly. FLINT's header also
//! exposes inline macros that consult a fixed-width function table before
//! falling through to the private `_flint_mpn_mul` implementation. The public
//! shared-library symbols preserve that production dispatch; binding the
//! private symbols would silently skip FLINT's small-operand kernels and make
//! the comparison look better below roughly 17 limbs.

#![allow(
    unsafe_code,
    reason = "the benchmark calls FLINT's raw mpn routines with disjoint, exactly sized vectors"
)]

use mp_anafis::tune_api::tier::Limb;

/// FLINT's `mp_limb_t`. Checked against `Limb` at every call site's benchmark.
pub type FlintLimb = u64;
/// FLINT's `mp_size_t`, which is `slong` and therefore pointer-width signed.
pub type FlintSize = isize;

#[link(name = "flint")]
unsafe extern "C" {
    /// FLINT's production `flint_mpn_mul(r, x, xn, y, yn)`, requiring
    /// `xn >= yn >= 1` and disjoint output/input spans.
    pub fn flint_mpn_mul(
        destination: *mut FlintLimb,
        larger: *const FlintLimb,
        larger_len: FlintSize,
        smaller: *const FlintLimb,
        smaller_len: FlintSize,
    ) -> FlintLimb;
    /// FLINT's production `flint_mpn_mul_n(r, x, y, n)` for equal widths and
    /// disjoint output/input spans.
    pub fn flint_mpn_mul_n(
        destination: *mut FlintLimb,
        left: *const FlintLimb,
        right: *const FlintLimb,
        len: FlintSize,
    );
    /// Returns FLINT's current global worker budget.
    pub fn flint_get_num_threads() -> i32;
    /// Replaces FLINT's current global worker budget.
    pub fn flint_set_num_threads(threads: i32);
}

/// Scoped FLINT worker budget for one benchmark case.
///
/// FLINT's setting is process-global. Keeping the prior value in this guard
/// prevents a serial or parallel comparison row from changing later rows.
#[derive(Debug)]
pub struct FlintThreadBudget {
    previous: i32,
    workers: usize,
}

impl FlintThreadBudget {
    /// Sets FLINT's worker budget until this guard is dropped.
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let workers_i32 = i32::try_from(workers).expect("FLINT worker budget fits in i32");
        assert!(workers_i32 > 0, "FLINT worker budget must be positive");
        // SAFETY: both functions are FLINT's public process-wide thread-budget
        // API, and `workers_i32` was proved positive above.
        let previous = unsafe {
            let previous = flint_get_num_threads();
            flint_set_num_threads(workers_i32);
            previous
        };
        Self { previous, workers }
    }

    /// Returns the worker budget recorded in benchmark labels.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }
}

impl Drop for FlintThreadBudget {
    fn drop(&mut self) {
        // SAFETY: `previous` came directly from FLINT's getter before the
        // scoped override, so restoring it satisfies FLINT's own invariant.
        unsafe {
            flint_set_num_threads(self.previous);
        }
    }
}

/// Asserts the one-limb-width precondition every FLINT arm depends on.
pub const fn assert_compatible_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<FlintLimb>(),
        "the FLINT comparison requires one limb width"
    );
}
