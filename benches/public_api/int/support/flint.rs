//! Minimal FLINT `fmpz` wrapper for opt-in public API benchmark references.
//!
//! This module is compiled only for 64-bit x86 Linux with `_internal-tune`.
//! On that target FLINT's `slong` and `ulong` are the LP64 C `long` types used
//! by the declarations below. Other targets must provide and verify their own
//! ABI before enabling these bindings.

#![allow(
    clippy::std_instead_of_alloc,
    reason = "FFI strings in the benchmark harness can use std"
)]
#![allow(unsafe_code, reason = "FLINT C FFI requires unsafe blocks")]

use core::ffi::{c_char, c_int, c_long, c_ulong};
use std::{ffi::CString, sync::Once};

// FLINT declares `fmpz` as `slong` and `fmpz_t` as `fmpz[1]`. This module's
// cfg restricts it to the verified LP64 Linux ABI, where `slong == c_long`.
type Fmpz = c_long;

static FLINT_SINGLE_THREAD: Once = Once::new();

#[link(name = "flint")]
unsafe extern "C" {
    fn flint_set_num_threads(num_threads: c_int);
    fn fmpz_init(value: *mut Fmpz);
    fn fmpz_clear(value: *mut Fmpz);
    fn fmpz_set_str(value: *mut Fmpz, text: *const c_char, radix: c_int) -> c_int;
    fn fmpz_equal(left: *const Fmpz, right: *const Fmpz) -> c_int;
    fn fmpz_sgn(value: *const Fmpz) -> c_int;
    fn fmpz_is_odd(value: *const Fmpz) -> c_int;
    fn fmpz_euler_phi(result: *mut Fmpz, value: *const Fmpz);
    fn fmpz_gcd(result: *mut Fmpz, left: *const Fmpz, right: *const Fmpz);
    fn fmpz_lcm(result: *mut Fmpz, left: *const Fmpz, right: *const Fmpz);
    fn fmpz_jacobi(value: *const Fmpz, modulus: *const Fmpz) -> c_int;
    fn fmpz_fac_ui(result: *mut Fmpz, value: c_ulong);
    fn fmpz_is_probabprime(value: *const Fmpz) -> c_int;
}

/// Forces FLINT to use one worker thread for serial benchmark comparisons.
///
/// Divan has no process-wide setup hook, so each FLINT benchmark calls this
/// once before entering its timed closure. The operation is idempotent.
pub fn pin_flint_to_one_thread() {
    FLINT_SINGLE_THREAD.call_once(|| {
        // SAFETY: FLINT accepts any positive `int` thread count. One selects
        // its serial execution policy. `Once` serializes concurrent benchmark
        // setup and publishes completion before any caller constructs inputs.
        unsafe {
            flint_set_num_threads(1);
        }
    });
}

/// An owning benchmark-only wrapper around one initialized FLINT `fmpz_t`.
pub struct FlintInt {
    inner: Fmpz,
}

impl FlintInt {
    /// Parses a FLINT integer in a radix from 2 through 62.
    ///
    /// # Panics
    ///
    /// Panics if `radix` is outside FLINT's supported range, `text` contains an
    /// interior NUL byte, or FLINT rejects the integer literal.
    #[must_use]
    pub fn from_str_radix(text: &str, radix: i32) -> Self {
        assert!(
            (2..=62).contains(&radix),
            "FLINT integer radix must be in 2..=62"
        );

        let mut value = Self::new();
        let c_text = CString::new(text).expect("integer literal must not contain NUL bytes");
        // SAFETY: `value.inner` is an initialized `fmpz_t` exclusively borrowed
        // for the call. `c_text` owns a live, NUL-terminated buffer, and `radix`
        // was validated above to be in FLINT's supported 2..=62 range.
        let status = unsafe {
            fmpz_set_str(
                &raw mut value.inner,
                c_text.as_ptr(),
                c_int::try_from(radix).expect("validated radix fits c_int"),
            )
        };
        assert_eq!(status, 0, "FLINT rejected the integer literal");
        value
    }

    /// Computes Euler's totient for a positive integer.
    ///
    /// # Panics
    ///
    /// Panics if the input is not positive.
    #[must_use]
    pub fn euler_phi(&self) -> Self {
        assert!(self.is_positive(), "Euler's totient input must be positive");
        let mut result = Self::new();
        // SAFETY: both pointers refer to live, initialized `fmpz_t` values for
        // the duration of the call. The result is exclusively borrowed and is
        // a distinct Rust owner from the immutable input.
        unsafe {
            fmpz_euler_phi(&raw mut result.inner, &raw const self.inner);
        }
        result
    }

    /// Computes the greatest common divisor.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        let mut result = Self::new();
        // SAFETY: all pointers refer to live, initialized `fmpz_t` values. The
        // output is exclusively borrowed and belongs to neither input, while
        // both input pointers are read-only for the complete call.
        unsafe {
            fmpz_gcd(
                &raw mut result.inner,
                &raw const self.inner,
                &raw const other.inner,
            );
        }
        result
    }

    /// Computes the least common multiple.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Self {
        let mut result = Self::new();
        // SAFETY: all pointers refer to live, initialized `fmpz_t` values. The
        // output is exclusively borrowed and belongs to neither input, while
        // both input pointers are read-only for the complete call.
        unsafe {
            fmpz_lcm(
                &raw mut result.inner,
                &raw const self.inner,
                &raw const other.inner,
            );
        }
        result
    }

    /// Computes the Jacobi symbol `(self / modulus)`.
    ///
    /// # Panics
    ///
    /// Panics unless `modulus` is positive and odd, as required by FLINT.
    #[must_use]
    pub fn jacobi(&self, modulus: &Self) -> i32 {
        assert!(
            modulus.is_positive_odd(),
            "Jacobi modulus must be positive and odd"
        );
        // SAFETY: both pointers refer to live, initialized `fmpz_t` values and
        // remain immutably borrowed for the call. The checked modulus satisfies
        // FLINT's positive-odd Jacobi domain.
        unsafe { fmpz_jacobi(&raw const self.inner, &raw const modulus.inner) }
    }

    /// Computes `n!` for an unsigned 32-bit argument.
    #[must_use]
    pub fn factorial(n: u32) -> Self {
        let mut result = Self::new();
        // SAFETY: `result.inner` is a live initialized `fmpz_t` exclusively
        // borrowed for the call. This module's cfg verifies the LP64 Linux ABI,
        // and every `u32` is exactly representable by FLINT's `ulong`/`c_ulong`.
        unsafe {
            fmpz_fac_ui(&raw mut result.inner, c_ulong::from(n));
        }
        result
    }

    /// Runs FLINT's built-in fixed-policy probable-prime test.
    ///
    /// This is only a cost reference. Unlike `MpUint::is_probably_prime`, the
    /// FLINT entry point does not accept a requested Miller-Rabin round count,
    /// so it is not a direct equivalent of the benchmark's 24-round cells.
    #[must_use]
    pub fn probable_prime_cost_reference(&self) -> bool {
        // SAFETY: the pointer refers to a live, initialized `fmpz_t` and remains
        // immutably borrowed for the duration of the read-only FLINT call.
        unsafe { fmpz_is_probabprime(&raw const self.inner) != 0 }
    }

    fn new() -> Self {
        let mut inner: Fmpz = 0;
        // SAFETY: `inner` is aligned, writable storage for FLINT's one-element
        // `fmpz_t`. It has no prior allocation. `fmpz_init` establishes the sole
        // initialized value, which is immediately placed under this RAII owner.
        unsafe {
            fmpz_init(&raw mut inner);
        }
        Self { inner }
    }

    fn is_positive(&self) -> bool {
        // SAFETY: the pointer refers to a live initialized `fmpz_t` and remains
        // immutably borrowed for the complete read-only sign query.
        unsafe { fmpz_sgn(&raw const self.inner) > 0 }
    }

    fn is_positive_odd(&self) -> bool {
        // SAFETY: the pointer refers to a live initialized `fmpz_t` and remains
        // immutably borrowed for both read-only property queries.
        unsafe { fmpz_sgn(&raw const self.inner) > 0 && fmpz_is_odd(&raw const self.inner) != 0 }
    }
}

impl PartialEq for FlintInt {
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: both pointers refer to live initialized `fmpz_t` values and
        // remain immutably borrowed for the complete read-only comparison.
        unsafe { fmpz_equal(&raw const self.inner, &raw const other.inner) != 0 }
    }
}

impl Eq for FlintInt {}

impl Drop for FlintInt {
    fn drop(&mut self) {
        // SAFETY: `new` initialized this exact `fmpz_t`, ownership was never
        // duplicated, and `drop` has exclusive access. Thus this call occurs
        // exactly once and releases every allocation owned by the value.
        unsafe {
            fmpz_clear(&raw mut self.inner);
        }
    }
}
