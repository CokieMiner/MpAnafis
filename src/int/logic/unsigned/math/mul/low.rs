//! Truncated low-product multiplication tier.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use core::cmp::max;

use super::{
    Addition, KARATSUBA_THRESHOLD, Limb, MulScratch, Multiplication, Schoolbook,
    TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD,
};

/// The triangular kernel remains cheaper beyond the full-product Karatsuba
/// crossover because it omits nearly half the multiplication rows. Scaling
/// from the configured crossover keeps this boundary target-sensitive; the
/// current 20-limb Karatsuba threshold gives an 80-limb low-product boundary.
const MULLO_DC_THRESHOLD: usize = KARATSUBA_THRESHOLD.saturating_mul(4);

/// Namespace for truncated low-product multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LowProduct;

/// Stable const equivalent of [`max`].
const fn const_max(left: usize, right: usize) -> usize {
    if left >= right { left } else { right }
}

/// Returns `floor(len * numerator / denominator)` without overflowing `usize`.
///
/// Splitting the quotient and remainder first proves both products fit: the
/// quotient term is at most `len`, while `remainder * numerator` is bounded by
/// `denominator^2` for every ratio used by the Mulders schedule.
fn mullo_scaled_split(len: usize, numerator: usize, denominator: usize) -> usize {
    debug_assert!(numerator <= denominator, "split ratio exceeds one");
    let quotient = len.div_euclid(denominator);
    let remainder = len.rem_euclid(denominator);
    quotient
        .wrapping_mul(numerator)
        .wrapping_add(remainder.wrapping_mul(numerator).div_euclid(denominator))
}

/// Selects the smaller Mulders block for the tier used by the larger block.
///
/// If a full `m`-limb product costs approximately `m^e`, minimizing
/// `L(n) = M(n-s) + 2*L(s)` gives progressively smaller `s/n` as the full
/// multiplication exponent falls: `1/2` for schoolbook, `11/36` for
/// Karatsuba, `9/40` for Toom-3, `7/39` for Toom-4, `1/8` for Toom-6, and
/// `1/10` for Toom-8. Candidates are tested from the highest reachable tier
/// downward, so a ratio is selected exactly when its own large block clears
/// every crossover that the production dispatcher traverses to that tier.
fn mullo_mulders_small_len(len: usize) -> usize {
    // Every crossover below is a build-time constant, so the tier gates and
    // their `max` chains fold at compile time and only the per-`len`
    // comparisons remain in the recursion.
    const TOOM4_THRESHOLD: usize = const_max(TOOM_COOK_THRESHOLD, TOOM_COOK_4_THRESHOLD);
    const TOOM6_THRESHOLD: usize = const_max(TOOM4_THRESHOLD, TOOM_COOK_6_THRESHOLD);
    const TOOM8_THRESHOLD: usize = const_max(TOOM6_THRESHOLD, TOOM_COOK_85_THRESHOLD);
    const TOOM4_ENABLED: bool = TOOM_COOK_4_THRESHOLD != 0;
    const TOOM6_ENABLED: bool = TOOM4_ENABLED && TOOM_COOK_6_THRESHOLD != 0;
    const TOOM8_ENABLED: bool = TOOM6_ENABLED && TOOM_COOK_85_THRESHOLD != 0;

    let toom8_small = len.div_euclid(10);
    if TOOM8_ENABLED && len.wrapping_sub(toom8_small) >= TOOM8_THRESHOLD {
        return toom8_small;
    }

    let toom6_small = len.div_euclid(8);
    if TOOM6_ENABLED && len.wrapping_sub(toom6_small) >= TOOM6_THRESHOLD {
        return toom6_small;
    }

    let toom4_small = mullo_scaled_split(len, 7, 39);
    if TOOM4_ENABLED && len.wrapping_sub(toom4_small) >= TOOM4_THRESHOLD {
        return toom4_small;
    }

    let toom3_small = mullo_scaled_split(len, 9, 40);
    let toom3_large = len.wrapping_sub(toom3_small);
    if toom3_large >= TOOM_COOK_THRESHOLD {
        return toom3_small;
    }

    let karatsuba_small = mullo_scaled_split(len, 11, 36);
    let karatsuba_large = len.wrapping_sub(karatsuba_small);
    if karatsuba_large >= KARATSUBA_THRESHOLD {
        return karatsuba_small;
    }

    len.div_euclid(2)
}

/// Computes scratch for the two non-overlapping phases of Mulders recursion.
///
/// The full low-block product and either cross product never coexist: after
/// copying the needed low `len` limbs to `dst`, the entire full-product region
/// may be reused. Therefore the recurrence is the maximum, not the sum, of
/// `2*large + full_itch` and `small + low_itch(small)`.
fn mullo_mulders_scratch_len(len: usize) -> usize {
    if len < MULLO_DC_THRESHOLD {
        return 0;
    }
    let small_len = mullo_mulders_small_len(len);
    mullo_mulders_split_scratch_len(len, small_len)
}

/// Computes the reusable full-product/cross-product workspace for one split.
fn mullo_mulders_split_scratch_len(len: usize, small_len: usize) -> usize {
    let large_len = len.wrapping_sub(small_len);
    let full_inner = max(
        Multiplication::required_scratch(large_len, large_len),
        Multiplication::required_sqr_scratch(large_len),
    );
    let full_phase = large_len.saturating_mul(2).saturating_add(full_inner);
    let cross_phase = if small_len < MULLO_DC_THRESHOLD {
        0
    } else {
        small_len.saturating_add(mullo_mulders_scratch_len(small_len))
    };
    max(full_phase, cross_phase)
}

/// Accumulates the triangular schoolbook product into a cleared low span.
///
/// The selected multiply-add kernel writes row `i` only to positions
/// `i..len`; clearing exactly those `len` limbs establishes its initialized
/// destination precondition while avoiding every term whose degree is at
/// least `len`.
impl LowProduct {
    /// Computes the low `len` limbs with the triangular basecase kernel.
    ///
    /// # Safety
    ///
    /// `dst`, `a`, and `b` each contain at least `len` initialized limbs. The
    /// destination must not alias either input.
    pub unsafe fn basecase(dst: &mut [Limb], a: &[Limb], b: &[Limb], len: usize) {
        debug_assert!(
            dst.len() >= len && a.len() >= len && b.len() >= len,
            "slice lengths too short for truncated basecase product"
        );
        // SAFETY: the unsafe function contract gives `dst.len() >= len`.
        unsafe { dst.get_unchecked_mut(..len) }.fill(0);
        // SAFETY: the contract gives three nonaliasing `len`-limb spans, and
        // the destination prefix was initialized to the additive identity.
        unsafe {
            Schoolbook::mullo_basecase_unchecked(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);
        }
    }
}

/// Computes the low `len` limbs without forming the discarded high product.
///
/// For `a = a0 + a1*B^m`, `b = b0 + b1*B^m`, `a0,b0 < B^m`, and
/// `a1,b1 < B^s` where `m+s=len`, reduction modulo `B^len` gives
///
/// `a*b = a0*b0 + B^m*(a1*b0 + a0*b1) (mod B^len)`.
///
/// The `a1*b1*B^(2m)` term vanishes because `2m >= len`, and only the low
/// `s` limbs of each cross product can survive the shift by `m`. The unequal
/// `s:m` split is chosen to minimize work in the currently selected full
/// multiplication tier.
/// # Safety
///
/// `dst`, `a`, and `b` each contain at least `len` initialized limbs, the
/// destination is disjoint from both inputs, and `scratch` contains at least
/// [`mullo_mulders_scratch_len`] limbs.
unsafe fn mullo_mulders(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    len: usize,
    scratch: &mut [Limb],
) {
    debug_assert!(
        dst.len() >= len && a.len() >= len && b.len() >= len,
        "slice lengths too short for Mulders low product"
    );
    if len < MULLO_DC_THRESHOLD {
        // SAFETY: `mullo_mulders` is entered only from the validated root or a
        // recursive split whose three active spans are exactly `len` limbs.
        unsafe {
            LowProduct::basecase(dst, a, b, len);
        }
        return;
    }

    let small_len = mullo_mulders_small_len(len);
    // SAFETY: the current kernel contract supplies the slice and scratch
    // bounds; the schedule returns `0 < small_len <= len / 2` at this tier.
    unsafe {
        mullo_mulders_at_split(dst, a, b, len, small_len, scratch);
    }
}

/// Executes one proven Mulders split; recursive cross products use automatic tiers.
///
/// # Safety
///
/// `dst`, `a`, and `b` each contain at least `len` initialized limbs, `dst` is
/// disjoint from both inputs, `0 < small_len <= len / 2`, and `scratch`
/// contains at least [`mullo_mulders_split_scratch_len`] limbs. The two inputs
/// may alias one another.
unsafe fn mullo_mulders_at_split(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    len: usize,
    small_len: usize,
    scratch: &mut [Limb],
) {
    debug_assert!(
        small_len > 0 && small_len <= len.div_euclid(2),
        "Mulders split must keep a nonempty smaller block no wider than the full-product block"
    );
    // The split invariant proves this subtraction cannot underflow and gives
    // `large_len >= small_len` plus `large_len + small_len == len`.
    let large_len = len.wrapping_sub(small_len);
    // SAFETY: the root validation or recursive invariant gives `a.len() >= len`.
    let (a0, a1) = unsafe { a.get_unchecked(..len) }.split_at(large_len);
    // SAFETY: the same invariant gives `b.len() >= len`.
    let (b0, b1) = unsafe { b.get_unchecked(..len) }.split_at(large_len);

    // A valid non-ZST limb slice is at most `isize::MAX` bytes and `Limb`
    // occupies at least two bytes, so doubling this derived subspan fits.
    let full_product_len = large_len.wrapping_mul(2);
    let full_inner_len = max(
        Multiplication::required_scratch(large_len, large_len),
        Multiplication::required_sqr_scratch(large_len),
    );
    let (full_product, after_full_product) = scratch.split_at_mut(full_product_len);
    let (full_inner, _) = after_full_product.split_at_mut(full_inner_len);
    Multiplication::mul_limbs_with_slice_scratch(a0, b0, full_product, full_inner);
    // `full_product` has `2*large_len >= large_len+small_len == len` limbs.
    // SAFETY: the root invariant also gives `dst.len() >= len`.
    let dst_low = unsafe { dst.get_unchecked_mut(..len) };
    // SAFETY: the inequality above proves the product prefix is in bounds.
    let full_low = unsafe { full_product.get_unchecked(..len) };
    dst_low.copy_from_slice(full_low);

    if small_len < MULLO_DC_THRESHOLD {
        // SAFETY: a1 and the selected b0 prefix each expose small_len limbs;
        // the selected a0 prefix and b1 do as well. Each basecase kernel
        // accumulates exactly small_len limbs into the initialized high span.
        unsafe {
            Schoolbook::mullo_basecase_unchecked(
                dst.as_mut_ptr().add(large_len),
                a1.as_ptr(),
                b0.as_ptr(),
                small_len,
            );
            Schoolbook::mullo_basecase_unchecked(
                dst.as_mut_ptr().add(large_len),
                a0.as_ptr(),
                b1.as_ptr(),
                small_len,
            );
        }
        return;
    }

    let cross_inner_len = mullo_mulders_scratch_len(small_len);
    {
        let (cross_product, after_cross_product) = scratch.split_at_mut(small_len);
        let (cross_inner, _) = after_cross_product.split_at_mut(cross_inner_len);
        // SAFETY: `b0.len() == large_len >= small_len` by the split invariant.
        let b0_low = unsafe { b0.get_unchecked(..small_len) };
        // SAFETY: both inputs and `cross_product` have exactly `small_len`
        // limbs, and `cross_inner` was split to the recursive scratch length.
        unsafe {
            mullo_mulders(cross_product, a1, b0_low, small_len, cross_inner);
        }
        // SAFETY: `dst.len() >= len` and `large_len <= len`; this range has
        // `len-large_len == small_len == cross_product.len()` limbs.
        let dst_high = unsafe { dst.get_unchecked_mut(large_len..len) };
        let _ = Addition::add_slice_in_place(dst_high, cross_product);
    }
    {
        let (cross_product, after_cross_product) = scratch.split_at_mut(small_len);
        let (cross_inner, _) = after_cross_product.split_at_mut(cross_inner_len);
        // SAFETY: `a0.len() == large_len >= small_len` by the split invariant.
        let a0_low = unsafe { a0.get_unchecked(..small_len) };
        // SAFETY: this is the symmetric exact-width recursive cross product.
        unsafe {
            mullo_mulders(cross_product, a0_low, b1, small_len, cross_inner);
        }
        // SAFETY: this is the same exact `small_len`-limb high destination span.
        let dst_high = unsafe { dst.get_unchecked_mut(large_len..len) };
        let _ = Addition::add_slice_in_place(dst_high, cross_product);
    }
}

/// Computes a low product with a forced root split for tier benchmarking.
///
/// Recursive cross products retain automatic Mulders splitting, so this
/// isolates the root geometry without changing production children.
///
/// # Panics
///
/// Panics if the root is below the recursive threshold, if the forced smaller
/// block is zero or wider than half, or if any slice is shorter than `len`.
#[cfg(feature = "_internal-tune")]
impl LowProduct {
    pub fn mul_with_forced_root_split(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        len: usize,
        small_len: usize,
        scratch: &mut MulScratch,
    ) {
        assert!(
            len >= MULLO_DC_THRESHOLD,
            "forced Mulders root is below its recursive threshold"
        );
        assert!(small_len != 0, "forced Mulders smaller block is zero");
        assert!(
            small_len <= len.div_euclid(2),
            "forced Mulders smaller block exceeds half"
        );
        assert!(
            dst.len() >= len && a.len() >= len && b.len() >= len,
            "slice lengths too short for forced Mulders root"
        );

        let scratch_len = mullo_mulders_split_scratch_len(len, small_len);
        scratch.prepare(scratch_len);
        // SAFETY: the boundary assertion above proves `len <= a.len()`.
        let a_slice = unsafe { a.get_unchecked(..len) };
        // SAFETY: the same assertion proves `len <= b.len()`.
        let b_slice = unsafe { b.get_unchecked(..len) };
        // SAFETY: the assertions validate every root span and split relation;
        // the buffer was reserved to the exact forced-split scratch length.
        unsafe {
            mullo_mulders_at_split(dst, a_slice, b_slice, len, small_len, &mut scratch.buf);
        }
    }
}

/// Computes `dst = (a * b) mod B^len` into a preallocated slice.
impl LowProduct {
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], len: usize, scratch: &mut MulScratch) {
        if len == 0 {
            return;
        }
        assert!(
            dst.len() >= len && a.len() >= len && b.len() >= len,
            "slice lengths too short for truncated product"
        );

        if len < MULLO_DC_THRESHOLD {
            // SAFETY: the root assertion above validates all three `len`-limb
            // spans and Rust's borrows make the destination disjoint from inputs.
            unsafe {
                Self::basecase(dst, a, b, len);
            }
        } else {
            let scratch_len = mullo_mulders_scratch_len(len);
            scratch.prepare(scratch_len);
            // SAFETY: the root assertion proves `len <= a.len()`.
            let a_slice = unsafe { a.get_unchecked(..len) };
            // SAFETY: the root assertion proves `len <= b.len()`.
            let b_slice = unsafe { b.get_unchecked(..len) };
            // SAFETY: the root assertion validates the three active spans and
            // the buffer was reserved to the automatic recursion's scratch size.
            unsafe {
                mullo_mulders(dst, a_slice, b_slice, len, &mut scratch.buf);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/kernels/low.rs"]
mod tests;
