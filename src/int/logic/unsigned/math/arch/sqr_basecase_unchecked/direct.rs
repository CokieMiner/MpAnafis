//! Portable universal schoolbook squaring kernel.
//!
//! Provides specialized unrolled kernels for lengths 2, 3, 4, 5, 6, and 8, as
//! well as the general triangular basecase squaring with fused doubling and
//! diagonal addition for larger operand widths.

use super::{ArchKernels, Limb};

/// Writes the complete square `a^2` into `2 * len` limbs.
///
/// # Safety
///
/// `a` must cover `len` aligned initialized readable limbs. `dst` must cover
/// `2 * len` aligned writable limbs, disjoint from `a`; its contents may be
/// uninitialized. The complete output byte span must fit in `isize::MAX`.
/// `len == 0` performs no pointer access.
pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
    // SAFETY: each nonempty arm reads exactly len initialized input limbs and
    // writes exactly 2*len disjoint output limbs. Empty inputs are not accessed.
    unsafe {
        match len {
            0 => return,
            1 => {
                let limb = *a;
                let (low, high) = ArchKernels::mul_limb_lo_hi(limb, limb);
                *dst = low;
                *dst.add(1) = high;
                return;
            }
            2 => return sqr_2_unchecked(dst, a),
            3 => return sqr_3_unchecked(dst, a),
            4 => return sqr_4_unchecked(dst, a),
            5 => return sqr_5_unchecked(dst, a),
            6 => return sqr_6_unchecked(dst, a),
            8 => return sqr_8_unchecked(dst, a),
            _ => {}
        }
    }

    let add_mul_limbs = ArchKernels::selected_add_mul_limbs_unchecked();
    // SAFETY: this path has len >= 7, and the complete 2*len output span fits
    // in isize::MAX. All row and diagonal indices below are less than 2*len.
    // Row 0 initializes dst[0..=len]. Each subsequent row reads only the
    // initialized prefix and extends it by one limb through dst[2*len-2].
    // Inputs and output are aligned and disjoint. With B=2^Limb::BITS,
    // (B-1)^2 + (B-1) < B^2 proves each row's high-product carry fits a limb.
    // Sums of two Boolean carries are at most two on every pointer width.
    unsafe {
        let last = len.unchecked_sub(1);
        *dst = 0;
        let mut carry = 0;
        let scalar = *a;
        for column in 1..len {
            let (low, high) = ArchKernels::mul_limb_lo_hi(*a.add(column), scalar);
            let (sum, overflow) = low.overflowing_add(carry);
            *dst.add(column) = sum;
            carry = high.unchecked_add(Limb::from(overflow));
        }
        *dst.add(len) = carry;

        // T = sum_{i<j} a[i]*a[j]*B^(i+j); each row adds its strict suffix.
        for row in 1..last {
            let column = row.unchecked_mul(2).unchecked_add(1);
            let remaining = last.unchecked_sub(row);
            let row_carry = add_mul_limbs(
                dst.add(column),
                a.add(row.unchecked_add(1)),
                remaining,
                *a.add(row),
            );
            *dst.add(row.unchecked_add(len)) = row_carry;
        }

        // a^2 = 2*T + sum_i a[i]^2*B^(2*i).
        let mut shift_carry = 0;
        let mut add_carry = 0;
        for diagonal in 0..last {
            let low_index = diagonal.unchecked_mul(2);
            let high_index = low_index.unchecked_add(1);
            let triangle_low = *dst.add(low_index);
            let triangle_high = *dst.add(high_index);
            let doubled_low = (triangle_low << 1) | shift_carry;
            let doubled_high = (triangle_high << 1) | (triangle_low >> (Limb::BITS - 1));
            shift_carry = triangle_high >> (Limb::BITS - 1);
            let limb = *a.add(diagonal);
            let (square_low, square_high) = ArchKernels::mul_limb_lo_hi(limb, limb);
            let (partial_low, carry_a) = doubled_low.overflowing_add(square_low);
            let (sum_low, carry_b) = partial_low.overflowing_add(add_carry);
            let incoming = Limb::from(carry_a).unchecked_add(Limb::from(carry_b));
            let (partial_high, carry_c) = doubled_high.overflowing_add(square_high);
            let (sum_high, carry_d) = partial_high.overflowing_add(incoming);
            add_carry = Limb::from(carry_c).unchecked_add(Limb::from(carry_d));
            *dst.add(low_index) = sum_low;
            *dst.add(high_index) = sum_high;
        }

        // The highest triangle limb is initialized; the final output limb is
        // not. Its doubled value consists only of the preceding shift carry.
        let low_index = last.unchecked_mul(2);
        let high_index = low_index.unchecked_add(1);
        let triangle_low = *dst.add(low_index);
        let doubled_low = (triangle_low << 1) | shift_carry;
        let doubled_high = triangle_low >> (Limb::BITS - 1);
        let limb = *a.add(last);
        let (square_low, square_high) = ArchKernels::mul_limb_lo_hi(limb, limb);
        let (partial_low, carry_a) = doubled_low.overflowing_add(square_low);
        let (sum_low, carry_b) = partial_low.overflowing_add(add_carry);
        let incoming = Limb::from(carry_a).unchecked_add(Limb::from(carry_b));
        let (partial_high, carry_c) = doubled_high.overflowing_add(square_high);
        let (sum_high, carry_d) = partial_high.overflowing_add(incoming);
        *dst.add(low_index) = sum_low;
        *dst.add(high_index) = sum_high;
        // a < B^len implies a^2 < B^(2*len).
        debug_assert!(!carry_c && !carry_d, "square exceeds its complete width");
    }
}

#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the fixed square columns and removes the leaf call boundary"
)]
#[inline(always)]
unsafe fn sqr_2_unchecked(dst: *mut Limb, a: *const Limb) {
    // SAFETY: the driver provides two aligned initialized readable limbs.
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
    // SAFETY: these nonnegative terms form the top square digit. A two-limb
    // operand is less than B^2, so its square is less than B^4.
    let r3 = unsafe {
        h1.unchecked_add(mid_carry)
            .unchecked_add(Limb::from(c2a))
            .unchecked_add(Limb::from(c2b))
    };

    // SAFETY: the driver provides four aligned writable output limbs disjoint
    // from the input. Each digit is written without reading its old contents.
    unsafe {
        *dst = l0;
        *dst.add(1) = r1;
        *dst.add(2) = r2;
        *dst.add(3) = r3;
    }
}

#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the fixed square columns and removes the leaf call boundary"
)]
#[inline(always)]
unsafe fn sqr_3_unchecked(dst: *mut Limb, a: *const Limb) {
    // SAFETY: the driver provides three aligned initialized input limbs and
    // six disjoint writable output limbs. Each carry sum is at most two.
    // The final nonnegative column sum fits a limb because a^2 < B^6.
    unsafe {
    let (a0, a1, a2) = (*a, *a.add(1), *a.add(2));

    let (l01, h01) = ArchKernels::mul_limb_lo_hi(a0, a1);
    let (l02, h02) = ArchKernels::mul_limb_lo_hi(a0, a2);
    let (l12, h12) = ArchKernels::mul_limb_lo_hi(a1, a2);

    let t1 = l01;
    let (t2, c_t2) = h01.overflowing_add(l02);
    let (t3_tmp, c_t3_lo) = h02.overflowing_add(l12);
    let (t3, c_t3_hi) = t3_tmp.overflowing_add(Limb::from(c_t2));
    let (t4, c_t4) = h12.overflowing_add(Limb::from(c_t3_lo).unchecked_add(Limb::from(c_t3_hi)));
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
    let c2 = Limb::from(c2a).unchecked_add(Limb::from(c2b));

    let (r3_tmp, c3a) = h1.overflowing_add(d3);
    let (r3, c3b) = r3_tmp.overflowing_add(c2);
    let c3 = Limb::from(c3a).unchecked_add(Limb::from(c3b));

    let (r4_tmp, c4a) = l2.overflowing_add(d4);
    let (r4, c4b) = r4_tmp.overflowing_add(c3);
    let c4 = Limb::from(c4a).unchecked_add(Limb::from(c4b));

    let r5 = h2.unchecked_add(d5).unchecked_add(c4);

        *dst = l0;
        *dst.add(1) = r1;
        *dst.add(2) = r2;
        *dst.add(3) = r3;
        *dst.add(4) = r4;
        *dst.add(5) = r5;
    }
}

#[expect(
    clippy::inline_always,
    clippy::similar_names,
    reason = "Inlining exposes the fixed square columns; names identify their low/high products and carries"
)]
#[inline(always)]
unsafe fn sqr_4_unchecked(dst: *mut Limb, a: *const Limb) {
    // SAFETY: the driver provides four aligned initialized input limbs and
    // eight disjoint writable output limbs. Carry sums are at most three.
    // The final nonnegative column sum fits a limb because a^2 < B^8.
    unsafe {
    let (a0, a1, a2, a3) = (*a, *a.add(1), *a.add(2), *a.add(3));

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
        .unchecked_add(Limb::from(c3b_t))
        .unchecked_add(Limb::from(c3c_t));

    let (t4a, c4a_t) = h03.overflowing_add(h12);
    let (t4b, c4b_t) = t4a.overflowing_add(l13);
    let (t4, c4c_t) = t4b.overflowing_add(c3_carry);
    let c4_carry = Limb::from(c4a_t)
        .unchecked_add(Limb::from(c4b_t))
        .unchecked_add(Limb::from(c4c_t));

    let (t5a, c5a_t) = h13.overflowing_add(l23);
    let (t5, c5b_t) = t5a.overflowing_add(c4_carry);
    let c5_carry = Limb::from(c5a_t).unchecked_add(Limb::from(c5b_t));

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
    let c2 = Limb::from(c2a).unchecked_add(Limb::from(c2b));

    let (r3_tmp, c3a) = h1.overflowing_add(d3);
    let (r3, c3b) = r3_tmp.overflowing_add(c2);
    let c3 = Limb::from(c3a).unchecked_add(Limb::from(c3b));

    let (r4_tmp, c4a) = l2.overflowing_add(d4);
    let (r4, c4b) = r4_tmp.overflowing_add(c3);
    let c4 = Limb::from(c4a).unchecked_add(Limb::from(c4b));

    let (r5_tmp, c5a) = h2.overflowing_add(d5);
    let (r5, c5b) = r5_tmp.overflowing_add(c4);
    let c5 = Limb::from(c5a).unchecked_add(Limb::from(c5b));

    let (r6_tmp, c6a) = l3.overflowing_add(d6);
    let (r6, c6b) = r6_tmp.overflowing_add(c5);
    let c6 = Limb::from(c6a).unchecked_add(Limb::from(c6b));

    let r7 = h3.unchecked_add(d7).unchecked_add(c6);

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

#[expect(
    clippy::inline_always,
    clippy::similar_names,
    reason = "Inlining exposes the fixed square columns; names identify their low/high products and carries"
)]
#[inline(always)]
unsafe fn sqr_5_unchecked(dst: *mut Limb, a: *const Limb) {
    // SAFETY: the driver provides five aligned initialized input limbs and
    // ten disjoint writable output limbs. Carry sums are at most four.
    // The final nonnegative column sum fits a limb because a^2 < B^10.
    unsafe {
    let (a0, a1, a2, a3, a4) = (*a, *a.add(1), *a.add(2), *a.add(3), *a.add(4));

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
        .unchecked_add(Limb::from(c3b_t))
        .unchecked_add(Limb::from(c3c_t));

    let (t4a, c4a_t) = h03.overflowing_add(h12);
    let (t4b, c4b_t) = t4a.overflowing_add(l04);
    let (t4c, c4c_t) = t4b.overflowing_add(l13);
    let (t4, c4d_t) = t4c.overflowing_add(c3_carry);
    let c4_carry = Limb::from(c4a_t)
        .unchecked_add(Limb::from(c4b_t))
        .unchecked_add(Limb::from(c4c_t))
        .unchecked_add(Limb::from(c4d_t));

    let (t5a, c5a_t) = h04.overflowing_add(h13);
    let (t5b, c5b_t) = t5a.overflowing_add(l14);
    let (t5c, c5c_t) = t5b.overflowing_add(l23);
    let (t5, c5d_t) = t5c.overflowing_add(c4_carry);
    let c5_carry = Limb::from(c5a_t)
        .unchecked_add(Limb::from(c5b_t))
        .unchecked_add(Limb::from(c5c_t))
        .unchecked_add(Limb::from(c5d_t));

    let (t6a, c6a_t) = h14.overflowing_add(h23);
    let (t6b, c6b_t) = t6a.overflowing_add(l24);
    let (t6, c6c_t) = t6b.overflowing_add(c5_carry);
    let c6_carry = Limb::from(c6a_t)
        .unchecked_add(Limb::from(c6b_t))
        .unchecked_add(Limb::from(c6c_t));

    let (t7a, c7a_t) = h24.overflowing_add(l34);
    let (t7, c7b_t) = t7a.overflowing_add(c6_carry);
    let c7_carry = Limb::from(c7a_t).unchecked_add(Limb::from(c7b_t));

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
    let c2 = Limb::from(c2a).unchecked_add(Limb::from(c2b));

    let (r3_tmp, c3a) = h1.overflowing_add(d3);
    let (r3, c3b) = r3_tmp.overflowing_add(c2);
    let c3 = Limb::from(c3a).unchecked_add(Limb::from(c3b));

    let (r4_tmp, c4a) = l2.overflowing_add(d4);
    let (r4, c4b) = r4_tmp.overflowing_add(c3);
    let c4 = Limb::from(c4a).unchecked_add(Limb::from(c4b));

    let (r5_tmp, c5a) = h2.overflowing_add(d5);
    let (r5, c5b) = r5_tmp.overflowing_add(c4);
    let c5 = Limb::from(c5a).unchecked_add(Limb::from(c5b));

    let (r6_tmp, c6a) = l3.overflowing_add(d6);
    let (r6, c6b) = r6_tmp.overflowing_add(c5);
    let c6 = Limb::from(c6a).unchecked_add(Limb::from(c6b));

    let (r7_tmp, c7a) = h3.overflowing_add(d7);
    let (r7, c7b) = r7_tmp.overflowing_add(c6);
    let c7 = Limb::from(c7a).unchecked_add(Limb::from(c7b));

    let (r8_tmp, c8a) = l4.overflowing_add(d8);
    let (r8, c8b) = r8_tmp.overflowing_add(c7);
    let c8 = Limb::from(c8a).unchecked_add(Limb::from(c8b));

    let r9 = h4.unchecked_add(d9).unchecked_add(c8);

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

#[expect(
    clippy::inline_always,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "The six-limb square keeps each fixed column visible to code generation; names identify products and carries"
)]
#[inline(always)]
unsafe fn sqr_6_unchecked(dst: *mut Limb, a: *const Limb) {
    // SAFETY: the driver provides six aligned initialized input limbs and
    // twelve disjoint writable output limbs. Carry sums are at most five.
    // The final nonnegative column sum fits a limb because a^2 < B^12.
    unsafe {
    let (a0, a1, a2, a3, a4, a5) = (*a, *a.add(1), *a.add(2), *a.add(3), *a.add(4), *a.add(5));

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
        .unchecked_add(Limb::from(c3b_t))
        .unchecked_add(Limb::from(c3c_t));

    let (t4a, c4a_t) = h03.overflowing_add(h12);
    let (t4b, c4b_t) = t4a.overflowing_add(l04);
    let (t4c, c4c_t) = t4b.overflowing_add(l13);
    let (t4, c4d_t) = t4c.overflowing_add(c3_carry);
    let c4_carry = Limb::from(c4a_t)
        .unchecked_add(Limb::from(c4b_t))
        .unchecked_add(Limb::from(c4c_t))
        .unchecked_add(Limb::from(c4d_t));

    let (t5a, c5a_t) = h04.overflowing_add(h13);
    let (t5b, c5b_t) = t5a.overflowing_add(l05);
    let (t5c, c5c_t) = t5b.overflowing_add(l14);
    let (t5d, c5d_t) = t5c.overflowing_add(l23);
    let (t5, c5e_t) = t5d.overflowing_add(c4_carry);
    let c5_carry = Limb::from(c5a_t)
        .unchecked_add(Limb::from(c5b_t))
        .unchecked_add(Limb::from(c5c_t))
        .unchecked_add(Limb::from(c5d_t))
        .unchecked_add(Limb::from(c5e_t));

    let (t6a, c6a_t) = h05.overflowing_add(h14);
    let (t6b, c6b_t) = t6a.overflowing_add(h23);
    let (t6c, c6c_t) = t6b.overflowing_add(l15);
    let (t6d, c6d_t) = t6c.overflowing_add(l24);
    let (t6, c6e_t) = t6d.overflowing_add(c5_carry);
    let c6_carry = Limb::from(c6a_t)
        .unchecked_add(Limb::from(c6b_t))
        .unchecked_add(Limb::from(c6c_t))
        .unchecked_add(Limb::from(c6d_t))
        .unchecked_add(Limb::from(c6e_t));

    let (t7a, c7a_t) = h15.overflowing_add(h24);
    let (t7b, c7b_t) = t7a.overflowing_add(l25);
    let (t7c, c7c_t) = t7b.overflowing_add(l34);
    let (t7, c7d_t) = t7c.overflowing_add(c6_carry);
    let c7_carry = Limb::from(c7a_t)
        .unchecked_add(Limb::from(c7b_t))
        .unchecked_add(Limb::from(c7c_t))
        .unchecked_add(Limb::from(c7d_t));

    let (t8a, c8a_t) = h25.overflowing_add(h34);
    let (t8b, c8b_t) = t8a.overflowing_add(l35);
    let (t8, c8c_t) = t8b.overflowing_add(c7_carry);
    let c8_carry = Limb::from(c8a_t)
        .unchecked_add(Limb::from(c8b_t))
        .unchecked_add(Limb::from(c8c_t));

    let (t9a, c9a_t) = h35.overflowing_add(l45);
    let (t9, c9b_t) = t9a.overflowing_add(c8_carry);
    let c9_carry = Limb::from(c9a_t).unchecked_add(Limb::from(c9b_t));

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
    let c2 = Limb::from(c2a).unchecked_add(Limb::from(c2b));

    let (r3_tmp, c3a) = h1.overflowing_add(d3);
    let (r3, c3b) = r3_tmp.overflowing_add(c2);
    let c3 = Limb::from(c3a).unchecked_add(Limb::from(c3b));

    let (r4_tmp, c4a) = l2.overflowing_add(d4);
    let (r4, c4b) = r4_tmp.overflowing_add(c3);
    let c4 = Limb::from(c4a).unchecked_add(Limb::from(c4b));

    let (r5_tmp, c5a) = h2.overflowing_add(d5);
    let (r5, c5b) = r5_tmp.overflowing_add(c4);
    let c5 = Limb::from(c5a).unchecked_add(Limb::from(c5b));

    let (r6_tmp, c6a) = l3.overflowing_add(d6);
    let (r6, c6b) = r6_tmp.overflowing_add(c5);
    let c6 = Limb::from(c6a).unchecked_add(Limb::from(c6b));

    let (r7_tmp, c7a) = h3.overflowing_add(d7);
    let (r7, c7b) = r7_tmp.overflowing_add(c6);
    let c7 = Limb::from(c7a).unchecked_add(Limb::from(c7b));

    let (r8_tmp, c8a) = l4.overflowing_add(d8);
    let (r8, c8b) = r8_tmp.overflowing_add(c7);
    let c8 = Limb::from(c8a).unchecked_add(Limb::from(c8b));

    let (r9_tmp, c9a) = h4.overflowing_add(d9);
    let (r9, c9b) = r9_tmp.overflowing_add(c8);
    let c9 = Limb::from(c9a).unchecked_add(Limb::from(c9b));

    let (r10_tmp, c10a) = l5.overflowing_add(d10);
    let (r10, c10b) = r10_tmp.overflowing_add(c9);
    let c10 = Limb::from(c10a).unchecked_add(Limb::from(c10b));

    let r11 = h5.unchecked_add(d11).unchecked_add(c10);

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

#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the two fixed four-limb squares and their shared cross product"
)]
#[inline(always)]
unsafe fn sqr_8_unchecked(dst: *mut Limb, a: *const Limb) {
    let mut s0 = [0_usize; 8];
    let mut s1 = [0_usize; 8];
    let mut m = [0_usize; 8];

    // SAFETY: the driver provides eight aligned initialized input limbs, so
    // both four-limb halves exist. The three local eight-limb arrays are
    // aligned writable outputs disjoint from the input and each other.
    unsafe {
        sqr_4_unchecked(s0.as_mut_ptr(), a);
        sqr_4_unchecked(s1.as_mut_ptr(), a.add(4));
        ArchKernels::mul_basecase_unchecked(m.as_mut_ptr(), a, 4, a.add(4), 4);
    }

    // For a = low + high*B^4, a^2 = low^2 + 2*low*high*B^4 + high^2*B^8.
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

    let mut b1 = [0_usize; 4];
    // SAFETY: the four-limb sources are initialized by the local square and
    // doubling above. b1 is a separate aligned four-limb writable array.
    let c1 = unsafe {
        ArchKernels::add_limbs_3_unchecked(b1.as_mut_ptr(), s0.as_ptr().add(4), d_lo.as_ptr(), 4)
    };

    let mut b2 = [0_usize; 4];
    // SAFETY: the four-limb sources are initialized by the local square and
    // doubling above. b2 is a separate aligned four-limb writable array.
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
                        // SAFETY: c2 is a Boolean carry before this increment.
                        c2 = unsafe { c2.unchecked_add(1) };
                    }
                }
            }
        }
    }

    let mut b3 = [0_usize; 4];
    let (r0_lo, o0) = s1[4].overflowing_add(d8);
    let (r0_final, o1) = r0_lo.overflowing_add(c2);
    b3[0] = r0_final;
    // SAFETY: two Boolean carry bits sum to at most two.
    let carry = unsafe { Limb::from(o0).unchecked_add(Limb::from(o1)) };
    if carry != 0 {
        let (r1, o2) = s1[5].overflowing_add(carry);
        b3[1] = r1;
        if o2 {
            let (r2, o3) = s1[6].overflowing_add(1);
            b3[2] = r2;
            if o3 {
                // SAFETY: this is the final nonnegative square column; an
                // eight-limb operand has a square below B^16.
                b3[3] = unsafe { s1[7].unchecked_add(1) };
            } else {
                b3[3] = s1[7];
            }
        } else {
            b3[2] = s1[6];
            b3[3] = s1[7];
        }
    } else {
        b3[1] = s1[5];
        b3[2] = s1[6];
        b3[3] = s1[7];
    }

    // SAFETY: the driver provides sixteen aligned writable output limbs. All
    // local block digits are initialized and stored without reading dst.
    unsafe {
        *dst = s0[0];
        *dst.add(1) = s0[1];
        *dst.add(2) = s0[2];
        *dst.add(3) = s0[3];
        *dst.add(4) = b1[0];
        *dst.add(5) = b1[1];
        *dst.add(6) = b1[2];
        *dst.add(7) = b1[3];
        *dst.add(8) = b2[0];
        *dst.add(9) = b2[1];
        *dst.add(10) = b2[2];
        *dst.add(11) = b2[3];
        *dst.add(12) = b3[0];
        *dst.add(13) = b3[1];
        *dst.add(14) = b3[2];
        *dst.add(15) = b3[3];
    }
}
