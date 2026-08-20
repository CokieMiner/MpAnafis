//! Schoolbook multiplication, squaring, and raw basecase kernels.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use core::ptr::{copy_nonoverlapping, eq};

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
            if outer_len == 5 {
                Self::mul_fixed_equal_distinct::<5>(dst, inner, outer);
                return;
            }
            if outer_len == 6 {
                Self::mul_fixed_equal_distinct::<6>(dst, inner, outer);
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
        if len == 2 {
            // SAFETY: caller guarantees two readable input limbs and four writable output limbs.
            unsafe {
                Self::sqr_2_unchecked(dst, a);
            }
            return;
        }
        if len == 3 {
            // SAFETY: caller guarantees three readable input limbs and six writable output limbs.
            unsafe {
                Self::sqr_3_unchecked(dst, a);
            }
            return;
        }
        if len == 4 {
            // SAFETY: caller guarantees four readable input limbs and eight writable output limbs.
            unsafe {
                Self::sqr_4_unchecked(dst, a);
            }
            return;
        }
        if len == 5 {
            // SAFETY: caller guarantees five readable input limbs and 10 writable output limbs.
            unsafe {
                Self::sqr_5_unchecked(dst, a);
            }
            return;
        }
        if len == 6 {
            // SAFETY: caller guarantees six readable input limbs and 12 writable output limbs.
            unsafe {
                Self::sqr_6_unchecked(dst, a);
            }
            return;
        }
        if len == 8 {
            // SAFETY: caller guarantees eight readable input limbs and 16 writable output limbs.
            unsafe {
                Self::sqr_8_unchecked(dst, a);
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

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        reason = "Unrolled 2-limb squaring kernel uses standard l/h/r notation for peak performance"
    )]
    #[inline(always)]
    unsafe fn sqr_2_unchecked(dst: *mut Limb, a: *const Limb) {
        // SAFETY: caller guarantees two readable input limbs.
        let (a0, a1) = unsafe { (*a, *a.add(1)) };
        let (l0, h0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (l1, h1) = ArchKernels::mul_limb_lo_hi(a1, a1);

        let mid_low = l01 << 1;
        let mid_high = (h01 << 1) | (l01 >> (Limb::BITS - 1));
        let mid_carry = h01 >> (Limb::BITS - 1);

        let (r1, c1) = h0.overflowing_add(mid_low);
        let (r2_tmp, c2a) = l1.overflowing_add(mid_high);
        let (r2, c2b) = r2_tmp.overflowing_add(Limb::from(c1));
        let r3 = h1
            .wrapping_add(mid_carry)
            .wrapping_add(Limb::from(c2a))
            .wrapping_add(Limb::from(c2b));

        // SAFETY: caller guarantees four writable output limbs.
        unsafe {
            *dst = l0;
            *dst.add(1) = r1;
            *dst.add(2) = r2;
            *dst.add(3) = r3;
        }
    }

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        reason = "Unrolled 3-limb squaring kernel uses conventional column and carry identifiers"
    )]
    #[inline(always)]
    unsafe fn sqr_3_unchecked(dst: *mut Limb, a: *const Limb) {
        // SAFETY: caller guarantees three readable input limbs.
        let (a0, a1, a2) = unsafe { (*a, *a.add(1), *a.add(2)) };

        let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (l02, h02) = ArchKernels::mul_limb_lo_hi(a0, a2);
        let (l12, h12) = ArchKernels::mul_limb_lo_hi(a1, a2);

        let t1 = l01;
        let (t2, c_t2) = h01.overflowing_add(l02);
        let (t3_tmp, c_t3_lo) = h02.overflowing_add(l12);
        let (t3, c_t3_hi) = t3_tmp.overflowing_add(Limb::from(c_t2));
        let (t4, c_t4) = h12.overflowing_add(Limb::from(c_t3_lo).wrapping_add(Limb::from(c_t3_hi)));
        let t5 = Limb::from(c_t4);

        let d1 = t1 << 1;
        let d2 = (t2 << 1) | (t1 >> (Limb::BITS - 1));
        let d3 = (t3 << 1) | (t2 >> (Limb::BITS - 1));
        let d4 = (t4 << 1) | (t3 >> (Limb::BITS - 1));
        let d5 = (t5 << 1) | (t4 >> (Limb::BITS - 1));

        let (l0, h0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (l1, h1) = ArchKernels::mul_limb_lo_hi(a1, a1);
        let (l2, h2) = ArchKernels::mul_limb_lo_hi(a2, a2);

        let (r1, c1) = h0.overflowing_add(d1);
        let (r2_tmp, c2a) = l1.overflowing_add(d2);
        let (r2, c2b) = r2_tmp.overflowing_add(Limb::from(c1));
        let c2 = Limb::from(c2a).wrapping_add(Limb::from(c2b));

        let (r3_tmp, c3a) = h1.overflowing_add(d3);
        let (r3, c3b) = r3_tmp.overflowing_add(c2);
        let c3 = Limb::from(c3a).wrapping_add(Limb::from(c3b));

        let (r4_tmp, c4a) = l2.overflowing_add(d4);
        let (r4, c4b) = r4_tmp.overflowing_add(c3);
        let c4 = Limb::from(c4a).wrapping_add(Limb::from(c4b));

        let r5 = h2.wrapping_add(d5).wrapping_add(c4);

        // SAFETY: caller guarantees six writable output limbs.
        unsafe {
            *dst = l0;
            *dst.add(1) = r1;
            *dst.add(2) = r2;
            *dst.add(3) = r3;
            *dst.add(4) = r4;
            *dst.add(5) = r5;
        }
    }

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        reason = "Unrolled 4-limb squaring kernel uses conventional column and carry identifiers"
    )]
    #[inline(always)]
    unsafe fn sqr_4_unchecked(dst: *mut Limb, a: *const Limb) {
        // SAFETY: caller guarantees four readable input limbs.
        let (a0, a1, a2, a3) = unsafe { (*a, *a.add(1), *a.add(2), *a.add(3)) };

        let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (l02, h02) = ArchKernels::mul_limb_lo_hi(a0, a2);
        let (l03, h03) = ArchKernels::mul_limb_lo_hi(a0, a3);
        let (l12, h12) = ArchKernels::mul_limb_lo_hi(a1, a2);
        let (l13, h13) = ArchKernels::mul_limb_lo_hi(a1, a3);
        let (l23, h23) = ArchKernels::mul_limb_lo_hi(a2, a3);

        let t1 = l01;
        let (t2, c2_t) = h01.overflowing_add(l02);

        let (t3a, c3a_t) = h02.overflowing_add(l03);
        let (t3b, c3b_t) = t3a.overflowing_add(l12);
        let (t3, c3c_t) = t3b.overflowing_add(Limb::from(c2_t));
        let c3_carry = Limb::from(c3a_t)
            .wrapping_add(Limb::from(c3b_t))
            .wrapping_add(Limb::from(c3c_t));

        let (t4a, c4a_t) = h03.overflowing_add(h12);
        let (t4b, c4b_t) = t4a.overflowing_add(l13);
        let (t4, c4c_t) = t4b.overflowing_add(c3_carry);
        let c4_carry = Limb::from(c4a_t)
            .wrapping_add(Limb::from(c4b_t))
            .wrapping_add(Limb::from(c4c_t));

        let (t5a, c5a_t) = h13.overflowing_add(l23);
        let (t5, c5b_t) = t5a.overflowing_add(c4_carry);
        let c5_carry = Limb::from(c5a_t).wrapping_add(Limb::from(c5b_t));

        let (t6, c6_t) = h23.overflowing_add(c5_carry);
        let t7 = Limb::from(c6_t);

        let d1 = t1 << 1;
        let d2 = (t2 << 1) | (t1 >> (Limb::BITS - 1));
        let d3 = (t3 << 1) | (t2 >> (Limb::BITS - 1));
        let d4 = (t4 << 1) | (t3 >> (Limb::BITS - 1));
        let d5 = (t5 << 1) | (t4 >> (Limb::BITS - 1));
        let d6 = (t6 << 1) | (t5 >> (Limb::BITS - 1));
        let d7 = (t7 << 1) | (t6 >> (Limb::BITS - 1));

        let (l0, h0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (l1, h1) = ArchKernels::mul_limb_lo_hi(a1, a1);
        let (l2, h2) = ArchKernels::mul_limb_lo_hi(a2, a2);
        let (l3, h3) = ArchKernels::mul_limb_lo_hi(a3, a3);

        let (r1, c1) = h0.overflowing_add(d1);
        let (r2_tmp, c2a) = l1.overflowing_add(d2);
        let (r2, c2b) = r2_tmp.overflowing_add(Limb::from(c1));
        let c2 = Limb::from(c2a).wrapping_add(Limb::from(c2b));

        let (r3_tmp, c3a) = h1.overflowing_add(d3);
        let (r3, c3b) = r3_tmp.overflowing_add(c2);
        let c3 = Limb::from(c3a).wrapping_add(Limb::from(c3b));

        let (r4_tmp, c4a) = l2.overflowing_add(d4);
        let (r4, c4b) = r4_tmp.overflowing_add(c3);
        let c4 = Limb::from(c4a).wrapping_add(Limb::from(c4b));

        let (r5_tmp, c5a) = h2.overflowing_add(d5);
        let (r5, c5b) = r5_tmp.overflowing_add(c4);
        let c5 = Limb::from(c5a).wrapping_add(Limb::from(c5b));

        let (r6_tmp, c6a) = l3.overflowing_add(d6);
        let (r6, c6b) = r6_tmp.overflowing_add(c5);
        let c6 = Limb::from(c6a).wrapping_add(Limb::from(c6b));

        let r7 = h3.wrapping_add(d7).wrapping_add(c6);

        // SAFETY: caller guarantees eight writable output limbs.
        unsafe {
            *dst = l0;
            *dst.add(1) = r1;
            *dst.add(2) = r2;
            *dst.add(3) = r3;
            *dst.add(4) = r4;
            *dst.add(5) = r5;
            *dst.add(6) = r6;
            *dst.add(7) = r7;
        }
    }

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "Unrolled 5-limb squaring kernel uses conventional column and carry identifiers"
    )]
    #[inline(always)]
    unsafe fn sqr_5_unchecked(dst: *mut Limb, a: *const Limb) {
        // SAFETY: caller guarantees five readable input limbs.
        let (a0, a1, a2, a3, a4) = unsafe { (*a, *a.add(1), *a.add(2), *a.add(3), *a.add(4)) };

        let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (l02, h02) = ArchKernels::mul_limb_lo_hi(a0, a2);
        let (l03, h03) = ArchKernels::mul_limb_lo_hi(a0, a3);
        let (l04, h04) = ArchKernels::mul_limb_lo_hi(a0, a4);
        let (l12, h12) = ArchKernels::mul_limb_lo_hi(a1, a2);
        let (l13, h13) = ArchKernels::mul_limb_lo_hi(a1, a3);
        let (l14, h14) = ArchKernels::mul_limb_lo_hi(a1, a4);
        let (l23, h23) = ArchKernels::mul_limb_lo_hi(a2, a3);
        let (l24, h24) = ArchKernels::mul_limb_lo_hi(a2, a4);
        let (l34, h34) = ArchKernels::mul_limb_lo_hi(a3, a4);

        let t1 = l01;
        let (t2, c2_t) = h01.overflowing_add(l02);

        let (t3a, c3a_t) = h02.overflowing_add(l03);
        let (t3b, c3b_t) = t3a.overflowing_add(l12);
        let (t3, c3c_t) = t3b.overflowing_add(Limb::from(c2_t));
        let c3_carry = Limb::from(c3a_t)
            .wrapping_add(Limb::from(c3b_t))
            .wrapping_add(Limb::from(c3c_t));

        let (t4a, c4a_t) = h03.overflowing_add(h12);
        let (t4b, c4b_t) = t4a.overflowing_add(l04);
        let (t4c, c4c_t) = t4b.overflowing_add(l13);
        let (t4, c4d_t) = t4c.overflowing_add(c3_carry);
        let c4_carry = Limb::from(c4a_t)
            .wrapping_add(Limb::from(c4b_t))
            .wrapping_add(Limb::from(c4c_t))
            .wrapping_add(Limb::from(c4d_t));

        let (t5a, c5a_t) = h04.overflowing_add(h13);
        let (t5b, c5b_t) = t5a.overflowing_add(l14);
        let (t5c, c5c_t) = t5b.overflowing_add(l23);
        let (t5, c5d_t) = t5c.overflowing_add(c4_carry);
        let c5_carry = Limb::from(c5a_t)
            .wrapping_add(Limb::from(c5b_t))
            .wrapping_add(Limb::from(c5c_t))
            .wrapping_add(Limb::from(c5d_t));

        let (t6a, c6a_t) = h14.overflowing_add(h23);
        let (t6b, c6b_t) = t6a.overflowing_add(l24);
        let (t6, c6c_t) = t6b.overflowing_add(c5_carry);
        let c6_carry = Limb::from(c6a_t)
            .wrapping_add(Limb::from(c6b_t))
            .wrapping_add(Limb::from(c6c_t));

        let (t7a, c7a_t) = h24.overflowing_add(l34);
        let (t7, c7b_t) = t7a.overflowing_add(c6_carry);
        let c7_carry = Limb::from(c7a_t).wrapping_add(Limb::from(c7b_t));

        let (t8, c8_t) = h34.overflowing_add(c7_carry);
        let t9 = Limb::from(c8_t);

        let d1 = t1 << 1;
        let d2 = (t2 << 1) | (t1 >> (Limb::BITS - 1));
        let d3 = (t3 << 1) | (t2 >> (Limb::BITS - 1));
        let d4 = (t4 << 1) | (t3 >> (Limb::BITS - 1));
        let d5 = (t5 << 1) | (t4 >> (Limb::BITS - 1));
        let d6 = (t6 << 1) | (t5 >> (Limb::BITS - 1));
        let d7 = (t7 << 1) | (t6 >> (Limb::BITS - 1));
        let d8 = (t8 << 1) | (t7 >> (Limb::BITS - 1));
        let d9 = (t9 << 1) | (t8 >> (Limb::BITS - 1));

        let (l0, h0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (l1, h1) = ArchKernels::mul_limb_lo_hi(a1, a1);
        let (l2, h2) = ArchKernels::mul_limb_lo_hi(a2, a2);
        let (l3, h3) = ArchKernels::mul_limb_lo_hi(a3, a3);
        let (l4, h4) = ArchKernels::mul_limb_lo_hi(a4, a4);

        let (r1, c1) = h0.overflowing_add(d1);
        let (r2_tmp, c2a) = l1.overflowing_add(d2);
        let (r2, c2b) = r2_tmp.overflowing_add(Limb::from(c1));
        let c2 = Limb::from(c2a).wrapping_add(Limb::from(c2b));

        let (r3_tmp, c3a) = h1.overflowing_add(d3);
        let (r3, c3b) = r3_tmp.overflowing_add(c2);
        let c3 = Limb::from(c3a).wrapping_add(Limb::from(c3b));

        let (r4_tmp, c4a) = l2.overflowing_add(d4);
        let (r4, c4b) = r4_tmp.overflowing_add(c3);
        let c4 = Limb::from(c4a).wrapping_add(Limb::from(c4b));

        let (r5_tmp, c5a) = h2.overflowing_add(d5);
        let (r5, c5b) = r5_tmp.overflowing_add(c4);
        let c5 = Limb::from(c5a).wrapping_add(Limb::from(c5b));

        let (r6_tmp, c6a) = l3.overflowing_add(d6);
        let (r6, c6b) = r6_tmp.overflowing_add(c5);
        let c6 = Limb::from(c6a).wrapping_add(Limb::from(c6b));

        let (r7_tmp, c7a) = h3.overflowing_add(d7);
        let (r7, c7b) = r7_tmp.overflowing_add(c6);
        let c7 = Limb::from(c7a).wrapping_add(Limb::from(c7b));

        let (r8_tmp, c8a) = l4.overflowing_add(d8);
        let (r8, c8b) = r8_tmp.overflowing_add(c7);
        let c8 = Limb::from(c8a).wrapping_add(Limb::from(c8b));

        let r9 = h4.wrapping_add(d9).wrapping_add(c8);

        // SAFETY: caller guarantees ten writable output limbs.
        unsafe {
            *dst = l0;
            *dst.add(1) = r1;
            *dst.add(2) = r2;
            *dst.add(3) = r3;
            *dst.add(4) = r4;
            *dst.add(5) = r5;
            *dst.add(6) = r6;
            *dst.add(7) = r7;
            *dst.add(8) = r8;
            *dst.add(9) = r9;
        }
    }

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "Unrolled 6-limb squaring kernel uses conventional column and carry identifiers"
    )]
    #[inline(always)]
    unsafe fn sqr_6_unchecked(dst: *mut Limb, a: *const Limb) {
        // SAFETY: caller guarantees six readable input limbs.
        let (a0, a1, a2, a3, a4, a5) =
            unsafe { (*a, *a.add(1), *a.add(2), *a.add(3), *a.add(4), *a.add(5)) };

        let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (l02, h02) = ArchKernels::mul_limb_lo_hi(a0, a2);
        let (l03, h03) = ArchKernels::mul_limb_lo_hi(a0, a3);
        let (l04, h04) = ArchKernels::mul_limb_lo_hi(a0, a4);
        let (l05, h05) = ArchKernels::mul_limb_lo_hi(a0, a5);

        let (l12, h12) = ArchKernels::mul_limb_lo_hi(a1, a2);
        let (l13, h13) = ArchKernels::mul_limb_lo_hi(a1, a3);
        let (l14, h14) = ArchKernels::mul_limb_lo_hi(a1, a4);
        let (l15, h15) = ArchKernels::mul_limb_lo_hi(a1, a5);

        let (l23, h23) = ArchKernels::mul_limb_lo_hi(a2, a3);
        let (l24, h24) = ArchKernels::mul_limb_lo_hi(a2, a4);
        let (l25, h25) = ArchKernels::mul_limb_lo_hi(a2, a5);

        let (l34, h34) = ArchKernels::mul_limb_lo_hi(a3, a4);
        let (l35, h35) = ArchKernels::mul_limb_lo_hi(a3, a5);

        let (l45, h45) = ArchKernels::mul_limb_lo_hi(a4, a5);

        let t1 = l01;
        let (t2, c2_t) = h01.overflowing_add(l02);

        let (t3a, c3a_t) = h02.overflowing_add(l03);
        let (t3b, c3b_t) = t3a.overflowing_add(l12);
        let (t3, c3c_t) = t3b.overflowing_add(Limb::from(c2_t));
        let c3_carry = Limb::from(c3a_t)
            .wrapping_add(Limb::from(c3b_t))
            .wrapping_add(Limb::from(c3c_t));

        let (t4a, c4a_t) = h03.overflowing_add(h12);
        let (t4b, c4b_t) = t4a.overflowing_add(l04);
        let (t4c, c4c_t) = t4b.overflowing_add(l13);
        let (t4, c4d_t) = t4c.overflowing_add(c3_carry);
        let c4_carry = Limb::from(c4a_t)
            .wrapping_add(Limb::from(c4b_t))
            .wrapping_add(Limb::from(c4c_t))
            .wrapping_add(Limb::from(c4d_t));

        let (t5a, c5a_t) = h04.overflowing_add(h13);
        let (t5b, c5b_t) = t5a.overflowing_add(l05);
        let (t5c, c5c_t) = t5b.overflowing_add(l14);
        let (t5d, c5d_t) = t5c.overflowing_add(l23);
        let (t5, c5e_t) = t5d.overflowing_add(c4_carry);
        let c5_carry = Limb::from(c5a_t)
            .wrapping_add(Limb::from(c5b_t))
            .wrapping_add(Limb::from(c5c_t))
            .wrapping_add(Limb::from(c5d_t))
            .wrapping_add(Limb::from(c5e_t));

        let (t6a, c6a_t) = h05.overflowing_add(h14);
        let (t6b, c6b_t) = t6a.overflowing_add(h23);
        let (t6c, c6c_t) = t6b.overflowing_add(l15);
        let (t6d, c6d_t) = t6c.overflowing_add(l24);
        let (t6, c6e_t) = t6d.overflowing_add(c5_carry);
        let c6_carry = Limb::from(c6a_t)
            .wrapping_add(Limb::from(c6b_t))
            .wrapping_add(Limb::from(c6c_t))
            .wrapping_add(Limb::from(c6d_t))
            .wrapping_add(Limb::from(c6e_t));

        let (t7a, c7a_t) = h15.overflowing_add(h24);
        let (t7b, c7b_t) = t7a.overflowing_add(l25);
        let (t7c, c7c_t) = t7b.overflowing_add(l34);
        let (t7, c7d_t) = t7c.overflowing_add(c6_carry);
        let c7_carry = Limb::from(c7a_t)
            .wrapping_add(Limb::from(c7b_t))
            .wrapping_add(Limb::from(c7c_t))
            .wrapping_add(Limb::from(c7d_t));

        let (t8a, c8a_t) = h25.overflowing_add(h34);
        let (t8b, c8b_t) = t8a.overflowing_add(l35);
        let (t8, c8c_t) = t8b.overflowing_add(c7_carry);
        let c8_carry = Limb::from(c8a_t)
            .wrapping_add(Limb::from(c8b_t))
            .wrapping_add(Limb::from(c8c_t));

        let (t9a, c9a_t) = h35.overflowing_add(l45);
        let (t9, c9b_t) = t9a.overflowing_add(c8_carry);
        let c9_carry = Limb::from(c9a_t).wrapping_add(Limb::from(c9b_t));

        let (t10, c10_t) = h45.overflowing_add(c9_carry);
        let t11 = Limb::from(c10_t);

        let d1 = t1 << 1;
        let d2 = (t2 << 1) | (t1 >> (Limb::BITS - 1));
        let d3 = (t3 << 1) | (t2 >> (Limb::BITS - 1));
        let d4 = (t4 << 1) | (t3 >> (Limb::BITS - 1));
        let d5 = (t5 << 1) | (t4 >> (Limb::BITS - 1));
        let d6 = (t6 << 1) | (t5 >> (Limb::BITS - 1));
        let d7 = (t7 << 1) | (t6 >> (Limb::BITS - 1));
        let d8 = (t8 << 1) | (t7 >> (Limb::BITS - 1));
        let d9 = (t9 << 1) | (t8 >> (Limb::BITS - 1));
        let d10 = (t10 << 1) | (t9 >> (Limb::BITS - 1));
        let d11 = (t11 << 1) | (t10 >> (Limb::BITS - 1));

        let (l0, h0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (l1, h1) = ArchKernels::mul_limb_lo_hi(a1, a1);
        let (l2, h2) = ArchKernels::mul_limb_lo_hi(a2, a2);
        let (l3, h3) = ArchKernels::mul_limb_lo_hi(a3, a3);
        let (l4, h4) = ArchKernels::mul_limb_lo_hi(a4, a4);
        let (l5, h5) = ArchKernels::mul_limb_lo_hi(a5, a5);

        let (r1, c1) = h0.overflowing_add(d1);
        let (r2_tmp, c2a) = l1.overflowing_add(d2);
        let (r2, c2b) = r2_tmp.overflowing_add(Limb::from(c1));
        let c2 = Limb::from(c2a).wrapping_add(Limb::from(c2b));

        let (r3_tmp, c3a) = h1.overflowing_add(d3);
        let (r3, c3b) = r3_tmp.overflowing_add(c2);
        let c3 = Limb::from(c3a).wrapping_add(Limb::from(c3b));

        let (r4_tmp, c4a) = l2.overflowing_add(d4);
        let (r4, c4b) = r4_tmp.overflowing_add(c3);
        let c4 = Limb::from(c4a).wrapping_add(Limb::from(c4b));

        let (r5_tmp, c5a) = h2.overflowing_add(d5);
        let (r5, c5b) = r5_tmp.overflowing_add(c4);
        let c5 = Limb::from(c5a).wrapping_add(Limb::from(c5b));

        let (r6_tmp, c6a) = l3.overflowing_add(d6);
        let (r6, c6b) = r6_tmp.overflowing_add(c5);
        let c6 = Limb::from(c6a).wrapping_add(Limb::from(c6b));

        let (r7_tmp, c7a) = h3.overflowing_add(d7);
        let (r7, c7b) = r7_tmp.overflowing_add(c6);
        let c7 = Limb::from(c7a).wrapping_add(Limb::from(c7b));

        let (r8_tmp, c8a) = l4.overflowing_add(d8);
        let (r8, c8b) = r8_tmp.overflowing_add(c7);
        let c8 = Limb::from(c8a).wrapping_add(Limb::from(c8b));

        let (r9_tmp, c9a) = h4.overflowing_add(d9);
        let (r9, c9b) = r9_tmp.overflowing_add(c8);
        let c9 = Limb::from(c9a).wrapping_add(Limb::from(c9b));

        let (r10_tmp, c10a) = l5.overflowing_add(d10);
        let (r10, c10b) = r10_tmp.overflowing_add(c9);
        let c10 = Limb::from(c10a).wrapping_add(Limb::from(c10b));

        let r11 = h5.wrapping_add(d11).wrapping_add(c10);

        // SAFETY: caller guarantees twelve writable output limbs.
        unsafe {
            *dst = l0;
            *dst.add(1) = r1;
            *dst.add(2) = r2;
            *dst.add(3) = r3;
            *dst.add(4) = r4;
            *dst.add(5) = r5;
            *dst.add(6) = r6;
            *dst.add(7) = r7;
            *dst.add(8) = r8;
            *dst.add(9) = r9;
            *dst.add(10) = r10;
            *dst.add(11) = r11;
        }
    }

    #[allow(
        clippy::inline_always,
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "Unrolled 8-limb squaring kernel uses Karatsuba block decomposition for peak performance"
    )]
    #[inline(always)]
    unsafe fn sqr_8_unchecked(dst: *mut Limb, a: *const Limb) {
        let mut s0 = [0_usize; 8];
        let mut s1 = [0_usize; 8];
        let mut m = [0_usize; 8];

        // SAFETY: caller guarantees 8 readable input limbs; a and a.add(4) each have 4 limbs.
        unsafe {
            Self::sqr_4_unchecked(s0.as_mut_ptr(), a);
            Self::sqr_4_unchecked(s1.as_mut_ptr(), a.add(4));
            ArchKernels::mul_basecase_unchecked(m.as_mut_ptr(), a, 4, a.add(4), 4);
        }

        // Double M:
        let d0 = m[0] << 1;
        let d1 = (m[1] << 1) | (m[0] >> (Limb::BITS - 1));
        let d2 = (m[2] << 1) | (m[1] >> (Limb::BITS - 1));
        let d3 = (m[3] << 1) | (m[2] >> (Limb::BITS - 1));
        let d4 = (m[4] << 1) | (m[3] >> (Limb::BITS - 1));
        let d5 = (m[5] << 1) | (m[4] >> (Limb::BITS - 1));
        let d6 = (m[6] << 1) | (m[5] >> (Limb::BITS - 1));
        let d7 = (m[7] << 1) | (m[6] >> (Limb::BITS - 1));
        let d8 = m[7] >> (Limb::BITS - 1);

        let d_lo = [d0, d1, d2, d3];
        let d_hi = [d4, d5, d6, d7];

        // Block 0: dst[0..4] = s0[0..4]
        // Block 1: dst[4..8] = s0[4..8] + d_lo
        let mut b1 = [0_usize; 4];
        // SAFETY: all pointers cover 4 limbs.
        let c1 = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                b1.as_mut_ptr(),
                s0.as_ptr().add(4),
                d_lo.as_ptr(),
                4,
            )
        };

        // Block 2: dst[8..12] = s1[0..4] + d_hi + c1
        let mut b2 = [0_usize; 4];
        // SAFETY: all pointers cover 4 limbs.
        let mut c2 = unsafe {
            ArchKernels::add_limbs_3_unchecked(b2.as_mut_ptr(), s1.as_ptr(), d_hi.as_ptr(), 4)
        };
        if c1 != 0 {
            let (r0, overflow) = b2[0].overflowing_add(c1);
            b2[0] = r0;
            if overflow {
                let (r1, overflow1) = b2[1].overflowing_add(1);
                b2[1] = r1;
                if overflow1 {
                    let (r2, overflow2) = b2[2].overflowing_add(1);
                    b2[2] = r2;
                    if overflow2 {
                        let (r3, overflow3) = b2[3].overflowing_add(1);
                        b2[3] = r3;
                        if overflow3 {
                            c2 = c2.wrapping_add(1);
                        }
                    }
                }
            }
        }

        // Block 3: dst[12..16] = s1[4..8] + d8 + c2
        let mut b3 = [0_usize; 4];
        b3.copy_from_slice(&s1[4..8]);
        let carry_in = d8.wrapping_add(c2);
        if carry_in != 0 {
            let (r0, overflow) = b3[0].overflowing_add(carry_in);
            b3[0] = r0;
            if overflow {
                let (r1, overflow1) = b3[1].overflowing_add(1);
                b3[1] = r1;
                if overflow1 {
                    let (r2, overflow2) = b3[2].overflowing_add(1);
                    b3[2] = r2;
                    if overflow2 {
                        // The square of an 8-limb integer a < B^8 satisfies a^2 < B^16,
                        // which strictly fits in 16 limbs (dst[0..16]). Therefore, no
                        // carry can escape past the most significant limb b3[3] (dst[15]).
                        let (r3, _) = b3[3].overflowing_add(1);
                        b3[3] = r3;
                    }
                }
            }
        }

        // Write all 16 limbs to dst:
        // SAFETY: caller guarantees dst is valid for 16 limbs.
        unsafe {
            copy_nonoverlapping(s0.as_ptr(), dst, 4);
            copy_nonoverlapping(b1.as_ptr(), dst.add(4), 4);
            copy_nonoverlapping(b2.as_ptr(), dst.add(8), 4);
            copy_nonoverlapping(b3.as_ptr(), dst.add(12), 4);
        }
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
