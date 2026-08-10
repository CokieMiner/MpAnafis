//! Schoolbook multiplication, squaring, and raw basecase kernels.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use core::ptr::eq;

use super::{ArchKernels, DoubleLimb, LIMB_BITS, Limb};

/// Namespace for schoolbook multiplication and its raw basecase kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Schoolbook;

impl Schoolbook {
    /// Computes the square of a limb slice using schoolbook squaring.
    ///
    /// `dst` must have at least `2 * a_limbs.len()` elements. Every active
    /// destination limb is initialized before it is read.
    pub fn sqr(dst: &mut [Limb], a_limbs: &[Limb]) {
        let len = a_limbs.len();
        if len == 0 {
            return;
        }
        assert!(
            dst.len() >= len.saturating_mul(2),
            "square destination buffer is too small"
        );

        // SAFETY:
        // - dst has length >= 2 * len and does not overlap a_limbs.
        // - a_limbs is valid for reads of length len.
        // - len > 0.
        unsafe {
            Self::sqr_basecase_unchecked(dst.as_mut_ptr(), a_limbs.as_ptr(), len);
        }
    }

    /// Computes the product of two limb slices using schoolbook multiplication.
    ///
    /// `dst` must have at least `a_limbs.len() + b_limbs.len()` elements.
    pub fn mul(dst: &mut [Limb], a_limbs: &[Limb], b_limbs: &[Limb]) {
        if a_limbs.is_empty() || b_limbs.is_empty() {
            return;
        }
        if eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len() {
            Self::sqr(dst, a_limbs);
            return;
        }
        assert!(
            dst.len() >= a_limbs.len().saturating_add(b_limbs.len()),
            "multiplication destination buffer is too small"
        );

        Self::mul_nonempty_distinct(dst, a_limbs, b_limbs);
    }

    /// Multiply already-validated nonempty, distinct operand slices.
    ///
    /// Callers must have excluded the equal-slice squaring case and provided a
    /// destination at least `a_limbs.len() + b_limbs.len()` limbs long. Valid Rust
    /// borrows prove that the mutable destination does not overlap either input.
    #[inline]
    pub fn mul_nonempty_distinct(dst: &mut [Limb], a_limbs: &[Limb], b_limbs: &[Limb]) {
        debug_assert!(
            !a_limbs.is_empty() && !b_limbs.is_empty(),
            "validated schoolbook operands must be nonempty"
        );
        debug_assert!(
            dst.len() >= a_limbs.len().saturating_add(b_limbs.len()),
            "schoolbook destination is shorter than the complete product"
        );

        let (outer, inner) = if a_limbs.len() <= b_limbs.len() {
            (a_limbs, b_limbs)
        } else {
            (b_limbs, a_limbs)
        };

        let outer_len = outer.len();
        let inner_len = inner.len();

        if outer_len == inner_len {
            if outer_len == 2 {
                Self::mul_fixed_equal_distinct::<2>(dst, inner, outer);
                return;
            }
            if outer_len == 3 {
                Self::mul_fixed_equal_distinct::<3>(dst, inner, outer);
                return;
            }
            if outer_len == 4 {
                Self::mul_fixed_equal_distinct::<4>(dst, inner, outer);
                return;
            }
        }

        if outer.len() == 1 {
            // SAFETY: both operands are nonempty; dst covers inner.len()+1 limbs,
            // and schoolbook multiplication requires disjoint input/output spans.
            unsafe {
                Self::mul_limb_unchecked(
                    dst.as_mut_ptr(),
                    inner.as_ptr(),
                    inner.len(),
                    *outer.get_unchecked(0),
                );
            }
            return;
        }

        // SAFETY:
        // - dst has at least outer.len() + inner.len() initialized limbs.
        // - outer has at least two limbs and inner is nonempty.
        // - both inputs are valid for their complete lengths and disjoint from dst.
        unsafe {
            ArchKernels::mul_basecase_unchecked(
                dst.as_mut_ptr(),
                outer.as_ptr(),
                outer.len(),
                inner.as_ptr(),
                inner.len(),
            );
        }
    }

    /// Multiply two distinct fixed-width operands without dynamic width branches.
    ///
    /// The caller selects a benchmark-backed width; the const length then reaches
    /// the architecture boundary without orientation or scalar-case branches.
    #[allow(
        clippy::inline_always,
        reason = "Fixed equal-width entries remove measured branch gaps from two through four limbs"
    )]
    #[inline(always)]
    pub fn mul_fixed_equal_distinct<const LEN: usize>(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
    ) {
        debug_assert_eq!(a_limbs.len(), LEN, "left operand has the wrong fixed width");
        debug_assert_eq!(
            b_limbs.len(),
            LEN,
            "right operand has the wrong fixed width"
        );
        debug_assert!(LEN >= 2, "fixed schoolbook width is below two limbs");
        debug_assert!(
            dst.len() >= LEN.saturating_mul(2),
            "fixed schoolbook destination is shorter than the product"
        );
        if LEN == 2 {
            // SAFETY: the caller proves two two-limb inputs and a four-limb destination.
            unsafe {
                ArchKernels::mul_2x2_portable_unchecked(
                    dst.as_mut_ptr(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }
        if LEN == 3 {
            // SAFETY: the caller proves two three-limb inputs and a six-limb
            // destination when this const specialization is selected.
            unsafe {
                ArchKernels::mul_3x3_portable_unchecked(
                    dst.as_mut_ptr(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }

        // SAFETY: the caller proves two LEN-limb inputs and a 2*LEN-limb
        // destination; valid Rust borrows prove non-overlap.
        unsafe {
            ArchKernels::mul_basecase_unchecked(
                dst.as_mut_ptr(),
                a_limbs.as_ptr(),
                LEN,
                b_limbs.as_ptr(),
                LEN,
            );
        }
    }

    /// Computes `dst = a * a` where `a` has `len` limbs without bounds checks.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for writes of `2 * len` limbs and must not overlap `a`.
    /// - `a` must be valid for reads of `len` limbs.
    /// - `len > 0`.
    #[allow(
        clippy::inline_always,
        reason = "Critical for peak basecase performance"
    )]
    #[inline(always)]
    unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
        if len == 1 {
            // SAFETY: caller guarantees one readable input limb and two writable output limbs.
            let (low, high) = ArchKernels::mul_limb_lo_hi(unsafe { *a }, unsafe { *a });
            // SAFETY: caller guarantees dst is writable for two limbs.
            unsafe {
                *dst = low;
                *dst.add(1) = high;
            }
            return;
        }

        // First form the strict upper triangle
        // sum(a_i*a_j*B^(i+j)), i < j. Row zero initializes dst[1..=len];
        // each later row starts inside the initialized prefix and writes its carry
        // one limb farther. By induction, no add-multiply reads an unwritten limb.
        // SAFETY: len >= 2, so a[0], a[1..len], dst[0], and dst[1..=len]
        // are valid. The input and output regions are disjoint by contract.
        unsafe {
            *dst = 0;
            Self::mul_limb_unchecked(dst.add(1), a.add(1), len.wrapping_sub(1), *a);
        }
        let mut index = 1_usize;
        let add_mul_limbs = ArchKernels::selected_add_mul_limbs_unchecked();
        while index.wrapping_add(1) < len {
            // SAFETY: index < len by the loop condition.
            let limb = unsafe { *a.add(index) };
            let remaining = len.wrapping_sub(1).wrapping_sub(index);
            // SAFETY: the source suffix has `remaining` limbs and the corresponding
            // output window plus its carry limb lies within the 2*len destination.
            unsafe {
                let carry = add_mul_limbs(
                    dst.add(index.wrapping_mul(2).wrapping_add(1)),
                    a.add(index.wrapping_add(1)),
                    remaining,
                    limb,
                );
                *dst.add(index.wrapping_add(len)) = carry;
            }
            index = index.wrapping_add(1);
        }

        // Double the triangle and add the diagonal squares in one pass.
        //
        // The two steps are independent per output position, so splitting them into
        // a shift traversal followed by a diagonal traversal costs one extra
        // read-modify-write pass over the whole `2n`-limb destination. Fused, each
        // limb pair is loaded once, doubled, summed with `a[i]^2`, and stored once.
        //
        // The triangle occupies `dst[1..=2n-2]`, so position zero reads the zero
        // written above and position `2n-1` is treated as a zero that receives the
        // bit shifted out of `dst[2n-2]`.
        let top_index = len.wrapping_mul(2).wrapping_sub(1);
        // Bit carried out of the previous limb by the doubling.
        let mut shift_carry: Limb = 0;
        // Carry of the addition chain, provably at most one (see below).
        let mut add_carry: Limb = 0;
        let mut diagonal_index = 0_usize;
        while diagonal_index < len {
            let low_index = diagonal_index.wrapping_mul(2);
            let high_index = low_index.wrapping_add(1);
            // SAFETY: diagonal_index < len, so both indices are below 2*len.
            let triangle_low = unsafe { *dst.add(low_index) };
            let triangle_high = if high_index == top_index {
                0
            } else {
                // SAFETY: the branch proves high_index < 2*len - 1.
                unsafe { *dst.add(high_index) }
            };

            let doubled_low = triangle_low.wrapping_shl(1) | shift_carry;
            let doubled_high = triangle_high.wrapping_shl(1)
                | triangle_low.wrapping_shr(Limb::BITS.wrapping_sub(1));
            shift_carry = triangle_high.wrapping_shr(Limb::BITS.wrapping_sub(1));

            // SAFETY: diagonal_index < len.
            let limb = unsafe { *a.add(diagonal_index) };
            let (square_low, square_high) = ArchKernels::mul_limb_lo_hi(limb, limb);

            // doubled_low + square_low + add_carry is at most 2B-1, so it emits at
            // most one carry; the two `overflowing_add` flags are never both set.
            let (partial_low, carry_a) = doubled_low.overflowing_add(square_low);
            let (sum_low, carry_b) = partial_low.overflowing_add(add_carry);
            let incoming = Limb::from(carry_a).wrapping_add(Limb::from(carry_b));
            // square_high is at most B-2, so the high column is at most 2B-2 and
            // likewise emits at most one carry.
            let (partial_high, carry_c) = doubled_high.overflowing_add(square_high);
            let (sum_high, carry_d) = partial_high.overflowing_add(incoming);
            add_carry = Limb::from(carry_c).wrapping_add(Limb::from(carry_d));

            // SAFETY: both indices are below 2*len and the spans are disjoint.
            unsafe {
                *dst.add(low_index) = sum_low;
                *dst.add(high_index) = sum_high;
            }
            diagonal_index = diagonal_index.wrapping_add(1);
        }
        // The final high limb is `dst[2n-1]`, whose triangle input was zero, so the
        // doubling leaves no outgoing bit. The bound `a^2 < B^(2n)` likewise proves
        // the addition chain cannot carry past the product width.
        debug_assert_eq!(shift_carry, 0, "square doubling carried past the product");
        debug_assert_eq!(
            add_carry, 0,
            "square diagonal carry exceeded the product width"
        );
    }

    /// Write one scalar product into an uninitialized destination window.
    ///
    /// # Safety
    ///
    /// `dst` must be writable for `len + 1` limbs and `src` readable for `len`
    /// limbs. The regions must either be disjoint or start at the same address;
    /// exact in-place operation is valid because limb `i` is read before it is
    /// overwritten and no later iteration reads it again.
    #[allow(
        clippy::inline_always,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The product is split into a low limb and a mathematically bounded carry in the basecase initializer"
    )]
    #[inline(always)]
    pub unsafe fn mul_limb_unchecked(dst: *mut Limb, src: *const Limb, len: usize, scalar: Limb) {
        let scalar_wide = scalar as DoubleLimb;
        let mut carry: DoubleLimb = 0;
        let mut index = 0_usize;
        while index < len {
            // SAFETY: the caller guarantees both spans for len limbs.
            let value = unsafe { *src.add(index) } as DoubleLimb;
            let product = value.wrapping_mul(scalar_wide).wrapping_add(carry);
            // SAFETY: index < len, so this output position is within dst.
            unsafe {
                *dst.add(index) = product as Limb;
            }
            carry = product >> LIMB_BITS;
            index = index.wrapping_add(1);
        }
        // SAFETY: dst has len + 1 writable limbs.
        unsafe {
            *dst.add(len) = carry as Limb;
        }
    }

    /// Computes the low `len` limbs of `a * b` with a basecase kernel.
    ///
    /// # Safety
    ///
    /// - `dst` must be initialized and writable for `len` limbs.
    /// - `a` and `b` must each be readable for `len` limbs.
    #[allow(
        clippy::inline_always,
        reason = "Critical for truncated-product performance"
    )]
    #[inline(always)]
    pub unsafe fn mullo_basecase_unchecked(
        dst: *mut Limb,
        a: *const Limb,
        b: *const Limb,
        len: usize,
    ) {
        let mut index = 0_usize;
        let add_mul_limbs = ArchKernels::selected_add_mul_limbs_unchecked();
        while index < len {
            // SAFETY: index < len.
            let limb = unsafe { *a.add(index) };
            let inner_len = len.wrapping_sub(index);
            // SAFETY: destination suffix and b prefix each contain inner_len limbs.
            let _ = unsafe { add_mul_limbs(dst.add(index), b, inner_len, limb) };
            index = index.wrapping_add(1);
        }
    }
}

#[cfg(test)]
#[path = "tests/kernels/basecase.rs"]
mod tests;
