//! Owning FLINT integers for numeric benchmark references.
//! The module gate restricts these C declarations to the Linux LP64 ABI.

#![expect(
    clippy::std_instead_of_alloc,
    reason = "The benchmark harness uses std FFI strings"
)]
#![expect(unsafe_code, reason = "FLINT C FFI requires unsafe blocks")]

use core::ffi::{CStr, c_char, c_int, c_long};
use std::{ffi::CString, sync::Once};

use super::Outcome;

type Fmpz = c_long;
static FLINT_SINGLE_THREAD: Once = Once::new();

#[link(name = "flint")]
unsafe extern "C" {
    fn flint_set_num_threads(num_threads: c_int);
    fn fmpz_init(value: *mut Fmpz);
    fn fmpz_clear(value: *mut Fmpz);
    fn fmpz_set_str(value: *mut Fmpz, text: *const c_char, radix: c_int) -> c_int;
    fn fmpz_sizeinbase(value: *const Fmpz, radix: c_int) -> usize;
    fn fmpz_get_str(text: *mut c_char, radix: c_int, value: *const Fmpz) -> *mut c_char;
    fn fmpz_equal(left: *const Fmpz, right: *const Fmpz) -> c_int;
    fn fmpz_sgn(value: *const Fmpz) -> c_int;
    fn fmpz_euler_phi(result: *mut Fmpz, value: *const Fmpz);
    fn fmpz_jacobi(value: *const Fmpz, modulus: *const Fmpz) -> c_int;
}

/// An owning, initialized FLINT integer; ownership is never duplicated.
pub struct FlintInt {
    inner: Fmpz,
}

/// Sets the serial comparison policy before entering a timed closure.
pub fn pin_flint_to_one_thread() {
    FLINT_SINGLE_THREAD.call_once(|| {
        // SAFETY: One is a valid positive FLINT worker count. Once serializes
        // setup and publishes completion before concurrent callers proceed.
        unsafe {
            flint_set_num_threads(1);
        }
    });
}

impl FlintInt {
    /// Parses a radix-2 through radix-62 integer.
    /// # Panics
    /// Panics on an invalid radix, embedded NUL, or invalid integer literal.
    pub fn from_str_radix(text: &str, radix: i32) -> Self {
        assert!((2..=62).contains(&radix), "radix must be in 2..=62");
        let mut value = Self::new();
        let c_text = CString::new(text).expect("literal has no NUL");
        // SAFETY: value is initialized and exclusively borrowed. text owns a
        // live NUL-terminated buffer. The radix is validated in FLINT's domain.
        let status = unsafe { fmpz_set_str(&raw mut value.inner, c_text.as_ptr(), radix) };
        assert_eq!(status, 0, "FLINT rejected the literal");
        value
    }

    /// Computes the totient of a positive integer.
    /// # Panics
    /// Panics for nonpositive input.
    #[must_use]
    pub fn euler_phi(&self) -> Self {
        // SAFETY: self.inner is a live initialized fmpz, immutably borrowed for
        // the duration of this read-only query.
        let positive = unsafe { fmpz_sgn(&raw const self.inner) > 0 };
        assert!(positive, "Euler's totient requires positive input");
        let mut result = Self::new();
        // SAFETY: Both fmpz values are live and initialized. result has exclusive
        // access and is distinct from the immutable input. The input is positive.
        unsafe {
            fmpz_euler_phi(&raw mut result.inner, &raw const self.inner);
        }
        result
    }

    /// Computes `(self | modulus)` for a positive odd modulus.
    ///
    /// # Safety
    /// The caller must establish that `modulus` is positive and odd.
    /// The benchmark fixtures validate this domain outside timing.
    #[must_use]
    pub unsafe fn jacobi_symbol_odd(&self, modulus: &Self) -> i32 {
        // SAFETY: both owners retain live initialized fmpz values under shared
        // borrows. The caller establishes FLINT's positive odd modulus domain.
        unsafe { fmpz_jacobi(&raw const self.inner, &raw const modulus.inner) }
    }

    fn new() -> Self {
        let mut inner = 0;
        // SAFETY: inner is aligned writable storage for one LP64 fmpz with no
        // prior allocation. init establishes the initialized RAII-owned value.
        unsafe {
            fmpz_init(&raw mut inner);
        }
        Self { inner }
    }
}

impl Outcome for FlintInt {
    fn encode(&self) -> String {
        // SAFETY: the owner retains an initialized fmpz under a shared borrow,
        // and 16 is within FLINT's supported radix range.
        let digits = unsafe { fmpz_sizeinbase(&raw const self.inner, 16) };
        let capacity = digits
            .checked_add(2)
            .expect("hexadecimal string fits memory");
        let mut text: Vec<c_char> = vec![0; capacity];
        // SAFETY: digits bounds the magnitude's hexadecimal digits. The two
        // additional initialized bytes cover its possible sign and NUL. FLINT
        // writes within this exclusive buffer and terminates the ASCII string.
        let encoded = unsafe {
            let _ = fmpz_get_str(text.as_mut_ptr(), 16, &raw const self.inner);
            CStr::from_ptr(text.as_ptr())
        };
        encoded
            .to_str()
            .expect("hexadecimal digits are ASCII")
            .to_owned()
    }
}

impl PartialEq for FlintInt {
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: Both pointers refer to live initialized fmpz values and
        // remain immutably borrowed throughout the read-only comparison.
        unsafe { fmpz_equal(&raw const self.inner, &raw const other.inner) != 0 }
    }
}

impl Eq for FlintInt {}

impl Drop for FlintInt {
    fn drop(&mut self) {
        // SAFETY: new initialized this exact value, ownership is unique, and
        // drop has exclusive access. clear therefore releases allocations once.
        unsafe {
            fmpz_clear(&raw mut self.inner);
        }
    }
}
