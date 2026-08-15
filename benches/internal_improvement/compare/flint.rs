//! Raw bindings to FLINT's integer multiplication entry points.
//!
//! FLINT 3 does not route large products through GMP. `flint_mpn_mul` selects
//! between hand-written fixed-width kernels, its own Toom tier, and the
//! `fft_small` module — an FFT over word-size primes requiring AVX2 or NEON,
//! which is the same family as our own `ntt` path rather than a Schoenhage-
//! Strassen one. That makes FLINT, not GMP, the relevant adversary above the
//! transform crossover.
//!
//! The out-of-line `flint_mpn_mul` is linked rather than `_flint_mpn_mul`: the
//! header exposes the former as an inline that consults a fixed-width function
//! table before falling through to the latter, and only the out-of-line copy
//! reproduces that dispatch. Linking `_flint_mpn_mul` would silently skip
//! FLINT's small-operand kernels and flatter us below ~17 limbs.

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
    /// `flint_mpn_mul(r, x, xn, y, yn)`, requiring `xn >= yn >= 1`.
    #[link_name = "_flint_mpn_mul"]
    pub fn flint_mpn_mul(
        destination: *mut FlintLimb,
        larger: *const FlintLimb,
        larger_len: FlintSize,
        smaller: *const FlintLimb,
        smaller_len: FlintSize,
    ) -> FlintLimb;
    /// `flint_mpn_mul_n(r, x, y, n)` for equal widths.
    #[link_name = "_flint_mpn_mul_n"]
    pub fn flint_mpn_mul_n(
        destination: *mut FlintLimb,
        left: *const FlintLimb,
        right: *const FlintLimb,
        len: FlintSize,
    );
    /// Pins FLINT to a single worker so the comparison stays core-for-core.
    pub fn flint_set_num_threads(threads: i32);
}

/// Asserts the one-limb-width precondition every FLINT arm depends on.
pub const fn assert_compatible_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<FlintLimb>(),
        "the FLINT comparison requires one limb width"
    );
}

/// Forces single-threaded FLINT.
///
/// `fft_small` will otherwise fan a large product across every core it is
/// allowed, which would compare a parallel implementation against two serial
/// ones. Called once per benchmark rather than once per process because divan
/// gives no process-level hook, and the call is idempotent and untimed.
pub fn pin_to_one_thread() {
    // SAFETY: `flint_set_num_threads` takes a plain integer, has no
    // preconditions beyond a positive count, and is safe to call repeatedly.
    unsafe {
        flint_set_num_threads(1);
    }
}
