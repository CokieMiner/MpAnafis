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
    fn fmpz_clear(value: *mut Fmpz);
    fn fmpz_set_str(value: *mut Fmpz, text: *const c_char, radix: c_int) -> c_int;
    fn fmpz_sizeinbase(value: *const Fmpz, radix: c_int) -> usize;
    fn fmpz_get_str(text: *mut c_char, radix: c_int, value: *const Fmpz) -> *mut c_char;
    fn fmpz_equal(left: *const Fmpz, right: *const Fmpz) -> c_int;
    fn fmpz_sgn(value: *const Fmpz) -> c_int;
    fn fmpz_euler_phi(result: *mut Fmpz, value: *const Fmpz);
    fn fmpz_gcd(result: *mut Fmpz, left: *const Fmpz, right: *const Fmpz);
    fn fmpz_lcm(result: *mut Fmpz, left: *const Fmpz, right: *const Fmpz);
    fn fmpz_divexact(result: *mut Fmpz, numerator: *const Fmpz, divisor: *const Fmpz);
    fn fmpz_mul(result: *mut Fmpz, left: *const Fmpz, right: *const Fmpz);
    fn fmpz_jacobi(value: *const Fmpz, modulus: *const Fmpz) -> c_int;
    fn fmpz_divisible(numerator: *const Fmpz, divisor: *const Fmpz) -> c_int;
    fn fmpz_tdiv_q(result: *mut Fmpz, numerator: *const Fmpz, divisor: *const Fmpz);
    fn fmpz_fdiv_r(result: *mut Fmpz, numerator: *const Fmpz, divisor: *const Fmpz);
    fn fmpz_tdiv_qr(
        quotient: *mut Fmpz,
        remainder: *mut Fmpz,
        numerator: *const Fmpz,
        divisor: *const Fmpz,
    );
}

/// An owning, initialized FLINT integer; ownership is never duplicated.
/// FLINT's `fmpz_init` initializes its LP64 coefficient to zero.
#[derive(Default)]
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
        let mut value = Self::default();
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
        let mut result = Self::default();
        // SAFETY: Both fmpz values are live and initialized. result has exclusive
        // access and is distinct from the immutable input. The input is positive.
        unsafe {
            fmpz_euler_phi(&raw mut result.inner, &raw const self.inner);
        }
        result
    }

    /// Computes a shared GCD and `lcm(a,b) = (a/gcd(a,b))*b`.
    /// The pair `(0,0)` produces `(0,0)` without exact division.
    ///
    /// # Safety
    /// Both inputs must be nonnegative. Fixtures establish this before timing.
    #[must_use]
    pub unsafe fn gcd_lcm_nonnegative(&self, other: &Self) -> (Self, Self) {
        let gcd = self.gcd(other);
        let mut lcm = Self::default();
        if gcd.inner != 0 {
            let mut quotient = Self::default();
            // SAFETY: every owner is initialized and each output is distinct
            // from its immutable inputs. A nonzero GCD divides self exactly.
            // The caller supplies nonnegative inputs, so the product is the LCM.
            unsafe {
                fmpz_divexact(
                    &raw mut quotient.inner,
                    &raw const self.inner,
                    &raw const gcd.inner,
                );
                fmpz_mul(
                    &raw mut lcm.inner,
                    &raw const quotient.inner,
                    &raw const other.inner,
                );
            }
        }
        (gcd, lcm)
    }

    /// Tests whether the GCD is one using FLINT's canonical unit encoding.
    #[must_use]
    pub fn is_coprime(&self, other: &Self) -> bool {
        let gcd = self.gcd(other);
        // fmpz_is_one compares the initialized canonical fmpz directly with 1.
        gcd.inner == 1
    }

    /// Computes the nonnegative greatest common divisor.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        let mut result = Self::default();
        // SAFETY: both immutable inputs remain initialized and live. The
        // initialized output is distinct and exclusively writable. GCD is total.
        unsafe {
            fmpz_gcd(
                &raw mut result.inner,
                &raw const self.inner,
                &raw const other.inner,
            );
        }
        result
    }

    /// Computes the nonnegative least common multiple.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Self {
        let mut result = Self::default();
        // SAFETY: both immutable inputs remain initialized and live. The
        // initialized output is distinct and exclusively writable. LCM is total.
        unsafe {
            fmpz_lcm(
                &raw mut result.inner,
                &raw const self.inner,
                &raw const other.inner,
            );
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

    /// Tests whether the divisor divides this value exactly.
    ///
    /// # Safety
    /// The divisor must be nonzero. Fixtures validate this domain before timing.
    #[must_use]
    pub unsafe fn is_divisible_by(&self, divisor: &Self) -> bool {
        // SAFETY: both owners retain live initialized fmpz values under shared
        // borrows. The caller establishes the nonzero divisor required by FLINT.
        unsafe { fmpz_divisible(&raw const self.inner, &raw const divisor.inner) != 0 }
    }

    /// Computes the quotient truncated toward zero.
    ///
    /// # Safety
    /// The divisor must be nonzero. Fixtures validate this domain before timing.
    #[must_use]
    pub unsafe fn div_trunc(&self, divisor: &Self) -> Self {
        let mut quotient = Self::default();
        // SAFETY: all owners retain initialized fmpz values. The output has
        // exclusive access and is distinct from both immutable inputs. The
        // caller establishes the nonzero divisor required by FLINT.
        unsafe {
            fmpz_tdiv_q(
                &raw mut quotient.inner,
                &raw const self.inner,
                &raw const divisor.inner,
            );
        }
        quotient
    }

    /// Computes the truncated remainder for nonnegative operands.
    /// Floor and truncated division coincide when the numerator is nonnegative
    /// and the divisor is positive.
    ///
    /// # Safety
    /// The numerator must be nonnegative and the divisor positive.
    /// Fixtures validate this domain before timing.
    #[must_use]
    pub unsafe fn rem_trunc_nonnegative(&self, divisor: &Self) -> Self {
        let mut remainder = Self::default();
        // SAFETY: the initialized output has exclusive access and is disjoint
        // from the live immutable inputs. The caller proves the numerator is
        // nonnegative and the divisor positive, so floor computes the desired
        // truncated remainder without a domain failure.
        unsafe {
            fmpz_fdiv_r(
                &raw mut remainder.inner,
                &raw const self.inner,
                &raw const divisor.inner,
            );
        }
        remainder
    }

    /// Computes the truncated quotient and remainder in one call.
    ///
    /// # Safety
    /// The divisor must be nonzero. Fixtures validate this domain before timing.
    #[must_use]
    pub unsafe fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let mut quotient = Self::default();
        let mut remainder = Self::default();
        // SAFETY: both outputs are initialized, distinct and exclusively
        // borrowed. The immutable inputs remain live throughout the call,
        // and the caller establishes that the divisor is nonzero.
        unsafe {
            fmpz_tdiv_qr(
                &raw mut quotient.inner,
                &raw mut remainder.inner,
                &raw const self.inner,
                &raw const divisor.inner,
            );
        }
        (quotient, remainder)
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
        // FLINT's LP64 COEFF_IS_MPZ tag is exactly (coefficient >> 62) == 1.
        // Inline coefficients own no allocation and require no clear call.
        if self.inner >> 62 == 1 {
            // SAFETY: the tag identifies this live initialized heap fmpz.
            // Ownership is unique and drop has exclusive access, so clear
            // releases its allocation exactly once.
            unsafe {
                fmpz_clear(&raw mut self.inner);
            }
        }
    }
}
