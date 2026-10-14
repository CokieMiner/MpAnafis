//! Scalar butterflies with exact fused product-error recovery.
//!
//! Catalog admission proves that products below `2*p^2` reduce below `p`
//! in magnitude, and products below `4*p^2` reduce below `3*p/2`.
//! Centered twiddles preserve the `2*p` stage bound without correction branches.

#![expect(
    unsafe_code,
    reason = "Kernel callers prove initialized spans, exclusive ownership, and residue bounds"
)]

use super::Fma;

/// Portable residue kernels.
#[derive(Debug)]
pub struct Scalar;

impl Scalar {
    /// Executes radix-4 butterflies over four disjoint quarter spans.
    ///
    /// # Safety
    /// `values` covers `4*quarter` initialized writable floats; `twiddles`
    /// covers `3*quarter` readable floats. `start <= end <= quarter`.
    /// Residue magnitudes are below `2*p`; twiddles lie in `(-p/2, p/2)`.
    /// `p < 2^50` and `pinv` is the rounded reciprocal of p.
    pub unsafe fn radix4<const INVERSE: bool>(
        values: *mut f64,
        twiddles: *const f64,
        quarter: usize,
        start: usize,
        end: usize,
        modulus: f64,
        pinv: f64,
    ) {
        for index in start..end {
            // SAFETY: index < quarter and all four value quarters and three
            // twiddle quarters are admitted. Quarter products and index sums
            // stay below the initialized slice lengths.
            unsafe {
                let q2 = quarter.unchecked_mul(2);
                let q3 = quarter.unchecked_mul(3);
                let a = *values.add(index);
                let b = *values.add(quarter.unchecked_add(index));
                let c = *values.add(q2.unchecked_add(index));
                let d = *values.add(q3.unchecked_add(index));
                let w0 = *twiddles.add(index);
                let w1 = *twiddles.add(quarter.unchecked_add(index));
                let w2 = *twiddles.add(q2.unchecked_add(index));
                let (y0, y1, y2, y3) = if INVERSE {
                    // b*w2 and d*w2 use less than p^2. Reduced pair sums
                    // have magnitude <= p/2; final products use < p^2/4.
                    let tb = if quarter == 1 {
                        b
                    } else {
                        Self::mulmod(b, w2, modulus, pinv)
                    };
                    let td = if quarter == 1 {
                        d
                    } else {
                        Self::mulmod(d, w2, modulus, pinv)
                    };
                    let s0 = Self::reduce(a + tb, modulus, pinv);
                    let s1 = Self::reduce(a - tb, modulus, pinv);
                    let t0 = Self::reduce(c + td, modulus, pinv);
                    let t1 = Self::reduce(c - td, modulus, pinv);
                    let u0 = if quarter == 1 {
                        t0
                    } else {
                        Self::mulmod(t0, w0, modulus, pinv)
                    };
                    let u1 = Self::mulmod(t1, w1, modulus, pinv);
                    (s0 + u0, s1 + u1, s0 - u0, s1 - u1)
                } else {
                    // Centered roots keep each difference product below
                    // 2*p^2. Their residues are below p, so their sum is below
                    // 2*p and the next stage receives the same lazy bound.
                    let reduced = Self::reduce(a, modulus, pinv);
                    let s0 = Self::reduce(reduced + c, modulus, pinv);
                    let s1 = if quarter == 1 {
                        Self::reduce(reduced - c, modulus, pinv)
                    } else {
                        Self::mulmod(reduced - c, w0, modulus, pinv)
                    };
                    let t0 = Self::reduce(b + d, modulus, pinv);
                    let t1 = Self::mulmod(b - d, w1, modulus, pinv);
                    let y1 = if quarter == 1 {
                        s0 - t0
                    } else {
                        Self::mulmod(s0 - t0, w2, modulus, pinv)
                    };
                    let y3 = if quarter == 1 {
                        s1 - t1
                    } else {
                        Self::mulmod(s1 - t1, w2, modulus, pinv)
                    };
                    (s0 + t0, y1, s1 + t1, y3)
                };
                *values.add(index) = y0;
                *values.add(quarter.unchecked_add(index)) = y1;
                *values.add(q2.unchecked_add(index)) = y2;
                *values.add(q3.unchecked_add(index)) = y3;
            }
        }
    }

    /// Multiplies spectra, or squares the left spectrum when `SQUARE` is true.
    ///
    /// # Safety
    /// `left` covers `len` initialized writable residues below `2*p` in magnitude.
    /// For multiplication, `right` covers a disjoint readable span with the
    /// same bound. For squaring, `right` is ignored and may be null.
    pub unsafe fn pointwise<const SQUARE: bool>(
        left: *mut f64,
        right: *const f64,
        len: usize,
        modulus: f64,
        pinv: f64,
    ) {
        for index in 0..len {
            // SAFETY: index < len; the caller owns the left slot exclusively
            // and supplies an initialized right slot when it is read.
            unsafe {
                let a = *left.add(index);
                let b = if SQUARE { a } else { *right.add(index) };
                *left.add(index) = Self::mulmod(a, b, modulus, pinv);
            }
        }
    }

    /// Scales inverse residues by a canonical inverse transform length.
    ///
    /// # Safety
    /// `values` covers `len` initialized writable residues below `2*p` in
    /// magnitude, and `factor` is a canonical residue below p.
    pub unsafe fn scale(values: *mut f64, len: usize, factor: f64, modulus: f64, pinv: f64) {
        for index in 0..len {
            // SAFETY: index < len and the caller exclusively owns the initialized slot.
            unsafe {
                *values.add(index) = Self::mulmod(*values.add(index), factor, modulus, pinv);
            }
        }
    }

    /// Reduces an integer-valued residue of magnitude below `4*p`.
    #[inline]
    pub fn reduce(value: f64, modulus: f64, pinv: f64) -> f64 {
        #[cfg(feature = "std")]
        let quotient = (value * pinv).round_ties_even();
        #[cfg(not(feature = "std"))]
        let quotient = {
            // abs(value/p) < 4. Adding a signed 2^52 rounds to an integer
            // with ties to even; subtraction removes the exact offset.
            let offset = 4_503_599_627_370_496.0_f64.copysign(value);
            (value * pinv + offset) - offset
        };
        Fma::mul_add(-quotient, modulus, value)
    }

    /// Computes an exact modular product for `abs(a*b) < 4*p^2`.
    #[inline]
    pub fn mulmod(a: f64, b: f64, modulus: f64, pinv: f64) -> f64 {
        let high = a * b;
        #[cfg(feature = "std")]
        let quotient = (high * pinv).round_ties_even();
        #[cfg(not(feature = "std"))]
        let quotient = {
            // abs(high/p) < 4*p < 2^52, the exact domain of this rounding offset.
            let offset = 4_503_599_627_370_496.0_f64.copysign(high);
            (high * pinv + offset) - offset
        };
        Fma::mul_add(a, b, -high) + Fma::mul_add(-quotient, modulus, high)
    }
}
