//! Split ratios for the Mulders recurrence `L(n) = M(n-s) + 2*L(s)`.
//!
//! A ratio applies when its larger block reaches the corresponding full-product
//! tier. Higher tiers use smaller cross products as their multiplication exponent falls.

#![expect(
    unsafe_code,
    reason = "The fixed split ratios are at most one half, and quotient-remainder decomposition bounds every product on 16/32/64-bit targets"
)]

use super::{
    KARATSUBA_THRESHOLD, Multiplication, TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD,
    TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD, Widths,
};

impl Multiplication {
    /// Selects a shared Mulders split from the larger block's reachable tier.
    ///
    /// Low products use `FALLBACK_DIVISOR=2`; high products use three so that
    /// two certification digits remain above both cross-product cuts. The
    /// rational tier ratios approximate the power-law optimum and are not
    /// mathematical constants. Every recursive caller supplies `len>=4`.
    pub fn mulders_small_len<const FALLBACK_DIVISOR: usize>(len: usize) -> usize {
        const TOOM4_THRESHOLD: usize =
            Widths::new(TOOM_COOK_THRESHOLD, TOOM_COOK_4_THRESHOLD).larger;
        const TOOM6_THRESHOLD: usize = Widths::new(TOOM4_THRESHOLD, TOOM_COOK_6_THRESHOLD).larger;
        const TOOM8_THRESHOLD: usize = Widths::new(TOOM6_THRESHOLD, TOOM_COOK_85_THRESHOLD).larger;
        const TOOM4_ENABLED: bool = TOOM_COOK_4_THRESHOLD != 0;
        const TOOM6_ENABLED: bool = TOOM4_ENABLED && TOOM_COOK_6_THRESHOLD != 0;
        const TOOM8_ENABLED: bool = TOOM6_ENABLED && TOOM_COOK_85_THRESHOLD != 0;
        const {
            assert!(
                matches!(FALLBACK_DIVISOR, 2 | 3),
                "Mulders fallback halves or thirds the input"
            );
        }

        let toom8_small = len.div_euclid(10);
        // SAFETY: floor(len/10) <= len on every pointer width.
        let toom8_large = unsafe { len.unchecked_sub(toom8_small) };
        if toom8_small != 0 && TOOM8_ENABLED && toom8_large >= TOOM8_THRESHOLD {
            return toom8_small;
        }
        let toom6_small = len.div_euclid(8);
        // SAFETY: floor(len/8) <= len.
        let toom6_large = unsafe { len.unchecked_sub(toom6_small) };
        if toom6_small != 0 && TOOM6_ENABLED && toom6_large >= TOOM6_THRESHOLD {
            return toom6_small;
        }
        let toom4_small = Self::scaled_split(len, 7, 39);
        // SAFETY: floor(7*len/39) <= len.
        let toom4_large = unsafe { len.unchecked_sub(toom4_small) };
        if toom4_small != 0 && TOOM4_ENABLED && toom4_large >= TOOM4_THRESHOLD {
            return toom4_small;
        }
        let toom3_small = Self::scaled_split(len, 9, 40);
        // SAFETY: floor(9*len/40) <= len.
        let toom3_large = unsafe { len.unchecked_sub(toom3_small) };
        if toom3_small != 0 && toom3_large >= TOOM_COOK_THRESHOLD {
            return toom3_small;
        }
        let karatsuba_small = Self::scaled_split(len, 11, 36);
        // SAFETY: floor(11*len/36) <= len.
        let karatsuba_large = unsafe { len.unchecked_sub(karatsuba_small) };
        if karatsuba_small != 0 && karatsuba_large >= KARATSUBA_THRESHOLD {
            return karatsuba_small;
        }
        len.div_euclid(FALLBACK_DIVISOR)
    }

    /// Computes `floor(len*numerator/denominator)` without an overflowing product.
    fn scaled_split(len: usize, numerator: usize, denominator: usize) -> usize {
        debug_assert!(numerator <= denominator, "split ratio exceeds one");
        let quotient = len.div_euclid(denominator);
        let remainder = len.rem_euclid(denominator);
        // SAFETY: numerator<=denominator<=40 at every call. The first product
        // is <=len, the second is <1600, and the combined quotient is <=len.
        unsafe {
            quotient
                .unchecked_mul(numerator)
                .unchecked_add(remainder.unchecked_mul(numerator).div_euclid(denominator))
        }
    }
}
