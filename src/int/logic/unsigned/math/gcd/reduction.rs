//! Reduction steps for the Lehmer tier.
//!
//! Unequal-width batches retain the longer operand's high contribution.
//! A rejected batch falls back to exact single-limb-quotient division.

#![expect(
    unsafe_code,
    reason = "ordered operand widths and caller-owned output spans establish scalar reduction and unequal matrix-kernel bounds"
)]

use core::cmp::{Ordering, max, min};

use super::{
    Addition, ArchKernels, Division, DoubleLimb, Gcd, InternalMpUint, LIMB_BITS, Limb,
    SignedLimbCarry,
};

impl Gcd {
    #[expect(
        clippy::too_many_arguments,
        reason = "Unequal Lehmer matrix update coordinates partitioned asymmetric passes with four conventional coefficients"
    )]
    pub unsafe fn lehmer_update_unequal(
        u: &[Limb],
        v: &[Limb],
        next_u: *mut Limb,
        next_v: *mut Limb,
        u_pos_coeff: Limb,
        u_neg_coeff: Limb,
        v_pos_coeff: Limb,
        v_neg_coeff: Limb,
        even: bool,
    ) -> bool {
        let u_len = u.len();
        let v_len = v.len();
        let min_len = min(u_len, v_len);
        let max_len = max(u_len, v_len);

        let mut carry_u = SignedLimbCarry::ZERO;
        let mut carry_v = SignedLimbCarry::ZERO;

        if even {
            for i in 0..min_len {
                // SAFETY: i < min_len <= u.len() and i < min_len <= v.len().
                let (ui, vi) = unsafe { (*u.get_unchecked(i), *v.get_unchecked(i)) };

                let (u_limb, next_carry_u) =
                    subtract_products_with_carry(ui, u_pos_coeff, vi, u_neg_coeff, carry_u);
                carry_u = next_carry_u;

                let (v_limb, next_carry_v) =
                    subtract_products_with_carry(vi, v_pos_coeff, ui, v_neg_coeff, carry_v);
                carry_v = next_carry_v;

                // SAFETY: destinations hold max_len limbs and i < min_len <= max_len.
                unsafe {
                    *next_u.add(i) = u_limb;
                    *next_v.add(i) = v_limb;
                }
            }
            if u_len > v_len {
                for i in min_len..max_len {
                    // SAFETY: i < u.len() == max_len.
                    let ui = unsafe { *u.get_unchecked(i) };
                    let (u_limb, next_carry_u) =
                        subtract_products_with_carry(ui, u_pos_coeff, 0, u_neg_coeff, carry_u);
                    carry_u = next_carry_u;

                    let (v_limb, next_carry_v) =
                        subtract_products_with_carry(0, v_pos_coeff, ui, v_neg_coeff, carry_v);
                    carry_v = next_carry_v;

                    // SAFETY: destinations hold max_len limbs and i < max_len.
                    unsafe {
                        *next_u.add(i) = u_limb;
                        *next_v.add(i) = v_limb;
                    }
                }
            } else {
                for i in min_len..max_len {
                    // SAFETY: i < v.len() == max_len.
                    let vi = unsafe { *v.get_unchecked(i) };
                    let (u_limb, next_carry_u) =
                        subtract_products_with_carry(0, u_pos_coeff, vi, u_neg_coeff, carry_u);
                    carry_u = next_carry_u;

                    let (v_limb, next_carry_v) =
                        subtract_products_with_carry(vi, v_pos_coeff, 0, v_neg_coeff, carry_v);
                    carry_v = next_carry_v;

                    // SAFETY: destinations hold max_len limbs and i < max_len.
                    unsafe {
                        *next_u.add(i) = u_limb;
                        *next_v.add(i) = v_limb;
                    }
                }
            }
        } else {
            for i in 0..min_len {
                // SAFETY: i < min_len <= u.len() and i < min_len <= v.len().
                let (ui, vi) = unsafe { (*u.get_unchecked(i), *v.get_unchecked(i)) };

                let (u_limb, next_carry_u) =
                    subtract_products_with_carry(vi, u_pos_coeff, ui, u_neg_coeff, carry_u);
                carry_u = next_carry_u;

                let (v_limb, next_carry_v) =
                    subtract_products_with_carry(ui, v_pos_coeff, vi, v_neg_coeff, carry_v);
                carry_v = next_carry_v;

                // SAFETY: destinations hold max_len limbs and i < min_len <= max_len.
                unsafe {
                    *next_u.add(i) = u_limb;
                    *next_v.add(i) = v_limb;
                }
            }
            if u_len > v_len {
                for i in min_len..max_len {
                    // SAFETY: i < u.len() == max_len.
                    let ui = unsafe { *u.get_unchecked(i) };
                    let (u_limb, next_carry_u) =
                        subtract_products_with_carry(0, u_pos_coeff, ui, u_neg_coeff, carry_u);
                    carry_u = next_carry_u;

                    let (v_limb, next_carry_v) =
                        subtract_products_with_carry(ui, v_pos_coeff, 0, v_neg_coeff, carry_v);
                    carry_v = next_carry_v;

                    // SAFETY: destinations hold max_len limbs and i < max_len.
                    unsafe {
                        *next_u.add(i) = u_limb;
                        *next_v.add(i) = v_limb;
                    }
                }
            } else {
                for i in min_len..max_len {
                    // SAFETY: i < v.len() == max_len.
                    let vi = unsafe { *v.get_unchecked(i) };
                    let (u_limb, next_carry_u) =
                        subtract_products_with_carry(vi, u_pos_coeff, 0, u_neg_coeff, carry_u);
                    carry_u = next_carry_u;

                    let (v_limb, next_carry_v) =
                        subtract_products_with_carry(0, v_pos_coeff, vi, v_neg_coeff, carry_v);
                    carry_v = next_carry_v;

                    // SAFETY: destinations hold max_len limbs and i < max_len.
                    unsafe {
                        *next_u.add(i) = u_limb;
                        *next_v.add(i) = v_limb;
                    }
                }
            }
        }

        (carry_u.low | carry_v.low) == 0 && !(carry_u.negative | carry_v.negative)
    }
}

#[expect(
    clippy::as_conversions,
    reason = "Products are split at LIMB_BITS into exact Limb halves after Limb-by-Limb multiplication."
)]
#[cfg_attr(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    expect(
        clippy::cast_possible_truncation,
        reason = "Products are split at LIMB_BITS into exact Limb halves after Limb-by-Limb multiplication."
    )
)]
#[inline]
pub fn subtract_products_with_carry(
    positive_value: Limb,
    positive_coefficient: Limb,
    negative_value: Limb,
    negative_coefficient: Limb,
    carry: SignedLimbCarry,
) -> (Limb, SignedLimbCarry) {
    let positive_product =
        (positive_value as DoubleLimb).wrapping_mul(positive_coefficient as DoubleLimb);
    let positive_low = positive_product as Limb;
    let positive_high = (positive_product >> LIMB_BITS) as Limb;
    let negative_product =
        (negative_value as DoubleLimb).wrapping_mul(negative_coefficient as DoubleLimb);
    let negative_low = negative_product as Limb;
    let negative_high = (negative_product >> LIMB_BITS) as Limb;

    let (sum, overflow) = positive_low.overflowing_add(carry.low);
    let (low, borrow) = sum.overflowing_sub(negative_low);

    // If B = 2^LIMB_BITS, the incoming carry is in [-B, B-1] and is
    // represented as `low - negative*B`. Splitting the low-limb operation
    // leaves
    //
    //   next = positive_high + overflow
    //        - (negative_high + borrow + negative).
    //
    // A Limb product has high half at most B-2, so the positive side is at
    // most B-1 and the negative side at most B. Therefore `next` remains in
    // [-B, B-1], preserving the representation invariant without i128.
    let positive_carry_low = positive_high.wrapping_add(Limb::from(overflow));
    let negative_with_borrow = negative_high.wrapping_add(Limb::from(borrow));
    let (negative_carry_low, negative_sign_high) =
        negative_with_borrow.overflowing_add(Limb::from(carry.negative));
    let next_negative = negative_sign_high || positive_carry_low < negative_carry_low;
    let next_low = positive_carry_low.wrapping_sub(negative_carry_low);
    (
        low,
        SignedLimbCarry {
            low: next_low,
            negative: next_negative,
        },
    )
}

impl Gcd {
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "LIMB_BITS fits in u32, and register shifts extract proven single-limb values on all pointer widths"
    )]
    pub fn fast_small_div_step(u: &mut InternalMpUint, v: &InternalMpUint) -> Option<Limb> {
        let u_len = u.limbs().len();
        let v_len = v.limbs().len();
        if v_len < 2 || u_len < v_len || u_len > v_len.wrapping_add(1) {
            return None;
        }

        // SAFETY: u_len >= v_len >= 2 proves index u_len - 1 exists.
        let u_top = unsafe { *u.limbs().get_unchecked(u_len.wrapping_sub(1)) };
        // SAFETY: v_len >= 2 proves index v_len - 1 exists.
        let v_top = unsafe { *v.limbs().get_unchecked(v_len.wrapping_sub(1)) };

        // When u is strictly longer than v and u_top > v_top, q >= 2^LIMB_BITS
        // provably exceeds a single limb. Rejecting early hoists this check before
        // computing leading zeros, shifts, or reading secondary/tertiary limbs.
        // Equal leading limbs still admit a single-limb quotient (e.g. u_top ==
        // v_top == B/2 can yield q == B-2), so only strict inequality rejects.
        if u_len > v_len && u_top > v_top {
            return None;
        }

        // Fast path for equal-length operands with matching top limbs: q is provably 1.
        if u_len == v_len && u_top == v_top {
            u.sub_assign(v);
            return Some(1);
        }

        // SAFETY: u_len >= v_len >= 2 proves index u_len - 2 exists.
        let u_second = unsafe { *u.limbs().get_unchecked(u_len.wrapping_sub(2)) };
        // SAFETY: v_len >= 2 proves index v_len - 2 exists.
        let v_second = unsafe { *v.limbs().get_unchecked(v_len.wrapping_sub(2)) };
        let v_third = if v_len >= 3 {
            // SAFETY: v_len >= 3 proves index v_len - 3 exists.
            unsafe { *v.limbs().get_unchecked(v_len.wrapping_sub(3)) }
        } else {
            0
        };

        let shift = v_top.leading_zeros();
        let (d1, d0) = if shift == 0 {
            (v_top, v_second)
        } else {
            let r_shift = (LIMB_BITS as u32).wrapping_sub(shift);
            (
                v_top.wrapping_shl(shift) | v_second.wrapping_shr(r_shift),
                v_second.wrapping_shl(shift) | v_third.wrapping_shr(r_shift),
            )
        };

        let (un2, un1, un0) = if u_len == v_len {
            let u_third = if u_len >= 3 {
                // SAFETY: u_len >= 3 proves index u_len - 3 exists.
                unsafe { *u.limbs().get_unchecked(u_len.wrapping_sub(3)) }
            } else {
                0
            };
            if shift == 0 {
                (0, u_top, u_second)
            } else {
                let r_shift = (LIMB_BITS as u32).wrapping_sub(shift);
                (
                    u_top.wrapping_shr(r_shift),
                    u_top.wrapping_shl(shift) | u_second.wrapping_shr(r_shift),
                    u_second.wrapping_shl(shift) | u_third.wrapping_shr(r_shift),
                )
            }
        } else {
            // SAFETY: u_len == v_len + 1 >= 3 proves index u_len - 3 exists.
            let u_third = unsafe { *u.limbs().get_unchecked(u_len.wrapping_sub(3)) };
            let u_fourth = if u_len >= 4 {
                // SAFETY: u_len >= 4 proves index u_len - 4 exists.
                unsafe { *u.limbs().get_unchecked(u_len.wrapping_sub(4)) }
            } else {
                0
            };
            if shift == 0 {
                (u_top, u_second, u_third)
            } else {
                let r_shift = (LIMB_BITS as u32).wrapping_sub(shift);
                (
                    u_top.wrapping_shl(shift) | u_second.wrapping_shr(r_shift),
                    u_second.wrapping_shl(shift) | u_third.wrapping_shr(r_shift),
                    u_third.wrapping_shl(shift) | u_fourth.wrapping_shr(r_shift),
                )
            }
        };

        if un2 > d1 || (un2 == d1 && un1 >= d0) {
            return None;
        }

        let dinv = Division::invert_pi1(d1, d0);
        let mut q_hat = Division::udiv_qr_3by2(un2, un1, un0, d1, d0, dinv).0;
        if q_hat == 0 {
            q_hat = 1;
        }

        let sub_mul = ArchKernels::selected_sub_mul_limbs_unchecked();
        let u_window = u.ensure_capacity_set_len_get_limbs(v_len.wrapping_add(1));
        let borrow = mul_sub_in_place(u_window, v.limbs(), q_hat, sub_mul);
        if borrow != 0 {
            let carry = Addition::add_slice_in_place(u_window, v.limbs());
            // SAFETY: the initialized window has v_len + 1 limbs. Add-back
            // absorbs its low-part carry in the top limb modulo the window.
            unsafe {
                let top = u_window.get_unchecked_mut(v_len);
                *top = top.wrapping_add(carry);
            }
            q_hat = q_hat.wrapping_sub(1);
        }
        u.normalize();
        let u_len_norm = u.limbs().len();
        if u_len_norm > v_len {
            u.sub_assign(v);
            q_hat = q_hat.wrapping_add(1);
        } else if u_len_norm == v_len {
            // SAFETY: u_len_norm == v_len >= 2 proves index u_len_norm - 1 exists in u.
            let u_top_rem = unsafe { *u.limbs().get_unchecked(u_len_norm.wrapping_sub(1)) };
            if u_top_rem > v_top || (u_top_rem == v_top && (*u).cmp(v) != Ordering::Less) {
                u.sub_assign(v);
                q_hat = q_hat.wrapping_add(1);
            }
        }
        Some(q_hat)
    }
}

/// Subtracts `q_hat*v_limbs` and returns the borrow above its high product.
/// The destination contains at least `v_limbs.len()+1` initialized limbs.
#[expect(
    clippy::inline_always,
    reason = "inlining keeps the selected limb kernel and high-digit borrow inside the GCD reduction step"
)]
#[inline(always)]
fn mul_sub_in_place(
    u_window: &mut [Limb],
    v_limbs: &[Limb],
    q_hat: Limb,
    sub_mul: unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb),
) -> Limb {
    let v_len = v_limbs.len();
    // SAFETY: the reduction driver initializes v_len+1 exclusive destination
    // limbs, disjoint from the divisor, and selects a supported CPU kernel.
    let (carry, borrow) = unsafe { sub_mul(u_window.as_mut_ptr(), v_limbs.as_ptr(), v_len, q_hat) };
    // SAFETY: the initialized destination includes its high guard at v_len.
    let u_val = unsafe { *u_window.get_unchecked(v_len) };
    let (diff1, b1) = u_val.overflowing_sub(carry);
    let (diff2, b2) = diff1.overflowing_sub(borrow);
    // SAFETY: the high guard remains exclusively borrowed throughout subtraction.
    unsafe {
        *u_window.get_unchecked_mut(v_len) = diff2;
    }
    Limb::from(b1 | b2)
}
