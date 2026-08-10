//! Construction of the Newton reciprocal `V = floor((B^2n - 1) / D)`.
//!
//! The recursion computes a reciprocal for the leading half of the divisor and
//! refines it against the full divisor, doubling the number of correct limbs per
//! level. Below [`NEWTON_RAPHSON_BASECASE_LIMBS`] it is cheaper to get the
//! reciprocal from one Algorithm D division of `B^2n - 1` by `D`.

use core::mem::replace;

use super::{
    Addition, DivScratch, Division, InternalArbiUint, Limb, Multiplication,
    NEWTON_RAPHSON_BASECASE_LIMBS, ScratchBuffer,
};

impl Division {
    /// Computes the reciprocal of a normalized divisor `den` (MSB set).
    ///
    /// Returns `V = floor((B^2n - 1) / den)`, which has at most `n + 1` limbs.
    #[allow(
        clippy::too_many_lines,
        reason = "Newton reciprocal combines initialization, multiplication, correction, and bit-level shifts in one state machine."
    )]
    pub fn newton_reciprocal(den: &[Limb], scratch: &mut DivScratch) -> InternalArbiUint {
        let n = den.len();
        if n <= NEWTON_RAPHSON_BASECASE_LIMBS || n <= 4 {
            return compute_reciprocal_basecase(den, scratch);
        }

        let k = n.wrapping_add(1).wrapping_div(2);
        // SAFETY: k = (n+1)/2, so n-k is <= n and >= 0. Both n and n-k are valid bounds for den.
        let den_hi = unsafe { den.get_unchecked(n.wrapping_sub(k)..n) };
        let mut v_hi = Self::newton_reciprocal(den_hi, scratch);

        // P = den * v_hi
        let v_hi_limbs = v_hi.limbs();
        let p_len = n.wrapping_add(v_hi_limbs.len());
        let mut p_buf = ScratchBuffer::acquire(p_len);
        p_buf.resize(p_len, 0);
        Multiplication::mul_limbs_with_scratch(
            den,
            v_hi_limbs,
            &mut p_buf,
            &mut scratch.mul_scratch,
        );

        let target_len = n.wrapping_add(k);
        // Trim trailing zeros above target_len from the initial product.
        while p_buf.last() == Some(&0) && p_buf.len() > target_len {
            let _ = p_buf.pop();
        }
        // Correct only while the product overflows target_len limbs, i.e.,
        // P >= 2^(64 * target_len). When p_buf.len() == target_len the value
        // fits in target_len limbs (< 2^(64 * target_len)) and no correction
        // is needed — the two's-complement computation of R below handles it.
        while p_buf.len() > target_len {
            let mut borrow: Limb = 1;
            for x in v_hi.limbs_mut() {
                if borrow == 0 {
                    break;
                }
                let (diff, b) = x.overflowing_sub(borrow);
                *x = diff;
                borrow = Limb::from(b);
            }
            v_hi.normalize();

            let borrow_p = Addition::sub_slice_in_place(&mut p_buf, den);
            if borrow_p != 0 {
                let den_len = den.len();
                // SAFETY: Addition::sub_slice_in_place requires p_buf to be >= den.len(), which it is.
                let _ = Addition::propagate_borrow(
                    unsafe { p_buf.get_unchecked_mut(den_len..) },
                    borrow_p,
                );
            }
            while p_buf.last() == Some(&0) && p_buf.len() > target_len {
                let _ = p_buf.pop();
            }
        }

        // R = 2^(64*(n+k)) - P
        scratch.newton_r_cur.resize(target_len, 0);
        let mut carry: Limb = 1;
        for (i, item) in scratch.newton_r_cur.iter_mut().enumerate().take(target_len) {
            let p_val = p_buf.get(i).copied().unwrap_or(0);
            let (inv, _) = (!p_val).overflowing_add(carry);
            *item = inv;
            carry = Limb::from(p_val == 0 && carry == 1);
        }
        while scratch.newton_r_cur.last() == Some(&0) {
            let _ = scratch.newton_r_cur.pop();
        }
        if scratch.newton_r_cur.is_empty() {
            scratch.newton_r_cur.push(0);
        }

        // C = R * v_hi
        let v_hi_limbs_ref = v_hi.limbs();
        let c_len = scratch
            .newton_r_cur
            .len()
            .wrapping_add(v_hi_limbs_ref.len());
        let mut c_buf = ScratchBuffer::acquire(c_len);
        c_buf.resize(c_len, 0);
        Multiplication::mul_limbs_with_scratch(
            &scratch.newton_r_cur,
            v_hi_limbs_ref,
            &mut c_buf,
            &mut scratch.mul_scratch,
        );

        // V = (v_hi << 64*(n - k)) + (C >> 64*(2*k))
        let v_hi_len = v_hi_limbs_ref.len();
        let v_total = n.wrapping_sub(k).wrapping_add(v_hi_len);
        scratch.newton_v_norm.clear();
        scratch.newton_v_norm.resize(v_total, 0);
        // Zero-initialized by resize, then copy v_hi into the upper part.
        // SAFETY: v_total = (n - k) + v_hi_len, which exactly fits the bounds n-k..v_total for scratch.newton_v_norm.
        unsafe {
            scratch
                .newton_v_norm
                .get_unchecked_mut(n.wrapping_sub(k)..v_total)
        }
        .copy_from_slice(v_hi_limbs_ref);

        let double_k = k.wrapping_mul(2);
        if c_buf.len() > double_k {
            // SAFETY: Checked that c_buf.len() > double_k.
            let c_slice = unsafe { c_buf.get_unchecked(double_k..) };
            let carry_out = Addition::add_slice_in_place(&mut scratch.newton_v_norm, c_slice);
            if carry_out == 1 {
                scratch.newton_v_norm.push(1);
            }
        }
        while scratch.newton_v_norm.last() == Some(&0) {
            let _ = scratch.newton_v_norm.pop();
        }
        if scratch.newton_v_norm.is_empty() {
            scratch.newton_v_norm.push(0);
        }

        // Transfer the result into dummy_rem to avoid a Vec allocation.
        let mut result = replace(&mut scratch.dummy_rem, InternalArbiUint::zero());
        let result_len = scratch.newton_v_norm.len();
        result.reserve(result_len);
        // SAFETY: reserve ensures capacity >= result_len.
        unsafe {
            result.set_len(result_len);
        }
        result.limbs_mut().copy_from_slice(&scratch.newton_v_norm);
        result.normalize();
        result
    }
}

/// Derives the reciprocal directly, by dividing `B^2n - 1` by `den`.
fn compute_reciprocal_basecase(den: &[Limb], scratch: &mut DivScratch) -> InternalArbiUint {
    let n = den.len();
    // Reuse scratch dummy fields to avoid Vec allocations in the base case.
    let mut n_a = replace(&mut scratch.dummy_u, InternalArbiUint::zero());
    n_a.reserve(n.wrapping_mul(2));
    // SAFETY: reserve ensures capacity >= n.wrapping_mul(2).
    unsafe {
        n_a.set_len(n.wrapping_mul(2));
    }
    n_a.limbs_mut().fill(Limb::MAX);
    let mut n_b = replace(&mut scratch.dummy_quot, InternalArbiUint::zero());
    n_b.reserve(n);
    // SAFETY: reserve ensures capacity >= n.
    unsafe {
        n_b.set_len(n);
    }
    n_b.limbs_mut().copy_from_slice(den);
    let mut q = replace(&mut scratch.dummy_rem, InternalArbiUint::zero());
    let mut rem = InternalArbiUint::zero();
    Division::algorithm_d(&n_a, &n_b, &mut q, &mut rem, scratch);
    scratch.dummy_rem = rem;
    scratch.dummy_u = n_a;
    scratch.dummy_quot = n_b;
    q
}
