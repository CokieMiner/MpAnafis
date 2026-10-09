//! FLINT multiplication bindings, limb ABI checks, and scoped worker budgets.
//! Comparison rows call the linked production `flint_mpn_mul` symbols directly.

#![expect(
    unsafe_code,
    reason = "the benchmark calls FLINT's raw mpn routines with disjoint, exactly sized vectors"
)]

use mp_anafis::tune_api::Limb;

use crate::shared::assert_gmp_limb_width;

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
    fn flint_get_num_threads() -> i32;
    /// Replaces FLINT's current global worker budget.
    fn flint_set_num_threads(threads: i32);
}

/// Scoped FLINT worker budget for one benchmark case.
///
/// FLINT's setting is process-global. Keeping the prior value in this guard
/// prevents a serial or parallel comparison row from changing later rows.
#[derive(Debug)]
pub struct FlintThreadBudget {
    previous: i32,
}

impl FlintThreadBudget {
    /// Sets FLINT's worker budget until this guard is dropped.
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let workers_i32 = i32::try_from(workers).expect("FLINT worker budget must fit in i32");
        assert!(workers_i32 > 0, "FLINT worker budget must be positive");
        // SAFETY: both functions are FLINT's public process-wide thread-budget
        // API, and `workers_i32` is validated positive above.
        let previous = unsafe {
            let previous = flint_get_num_threads();
            flint_set_num_threads(workers_i32);
            previous
        };
        Self { previous }
    }
}

impl Drop for FlintThreadBudget {
    fn drop(&mut self) {
        // SAFETY: `previous` is obtained directly from FLINT's getter before the
        // scoped override, so restoring it satisfies FLINT's own invariant.
        unsafe {
            flint_set_num_threads(self.previous);
        }
    }
}

/// Proves matching Mp, GMP, and FLINT limb size and alignment.
pub const fn assert_one_limb_width() {
    assert_gmp_limb_width();
    assert!(
        size_of::<Limb>() == size_of::<FlintLimb>(),
        "the FLINT comparison requires one limb width"
    );
    assert!(
        align_of::<Limb>() == align_of::<FlintLimb>(),
        "the FLINT comparison requires one limb alignment"
    );
}
