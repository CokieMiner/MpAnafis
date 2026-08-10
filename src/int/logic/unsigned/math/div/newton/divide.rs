//! Applying the Newton reciprocal to produce an exact quotient and remainder.
//!
//! A single reciprocal serves the whole division: the dividend is walked one
//! `n`-limb block at a time, each block yielding a quotient estimate that is
//! correct to within one or two units. Those are corrected against the divisor
//! directly; if the estimate is somehow further off than the reciprocal's bound
//! allows, the block falls back to Algorithm D rather than looping unbounded.

use core::{
    cmp::{Ordering, max, min},
    mem::{replace, swap},
};

use super::{
    Addition, ArchKernels, DivScratch, Division, InternalArbiUint, Limb, LowProduct,
    Multiplication, ScratchBuffer,
};

impl Division {
    /// Divide a `2n`-limb number `num` by an `n`-limb normalized divisor `den`.
    pub fn newton_2n1n(
        num: &[Limb],
        den: &[Limb],
        quo: &mut [Limb],
        rem: &mut [Limb],
        scratch: &mut DivScratch,
    ) {
        let v_recip = Self::newton_reciprocal(den, scratch);
        newton_div_2n1n_with_reciprocal(num, den, &v_recip, quo, rem, scratch);
    }

    /// Main entry point for Newton-Raphson division with remainder.
    pub fn newton(
        num_a: &InternalArbiUint,
        den_b: &InternalArbiUint,
        quotient_out: &mut InternalArbiUint,
        rem_out: &mut InternalArbiUint,
        scratch: &mut DivScratch,
    ) {
        let v_limbs = Self::significant_limbs(den_b.limbs());
        debug_assert!(!v_limbs.is_empty(), "division requires a non-zero divisor");
        let u_limbs = Self::significant_limbs(num_a.limbs());

        if u_limbs.len() < v_limbs.len()
            || (u_limbs.len() == v_limbs.len()
                && InternalArbiUint::cmp_limbs(u_limbs, v_limbs) == Ordering::Less)
        {
            quotient_out.clear();
            rem_out.clone_from(num_a);
            return;
        }

        // SAFETY: v_limbs is not empty
        let shift =
            unsafe { *v_limbs.get_unchecked(v_limbs.len().wrapping_sub(1)) }.leading_zeros();
        scratch.v_norm.clear();
        scratch.u_norm.clear();
        Self::shift_limbs_left(v_limbs, shift, &mut scratch.v_norm);
        Self::shift_limbs_left(u_limbs, shift, &mut scratch.u_norm);

        let n = scratch.v_norm.len();
        let m_norm = scratch.u_norm.len();

        let mut v_norm_buf = replace(&mut scratch.newton_v_norm, ScratchBuffer::acquire(0));
        v_norm_buf.clear();
        v_norm_buf.resize(n, 0);
        v_norm_buf.copy_from_slice(&scratch.v_norm);

        let mut u_norm_saved = replace(&mut scratch.newton_u_norm, ScratchBuffer::acquire(0));
        u_norm_saved.clear();
        u_norm_saved.resize(m_norm, 0);
        // SAFETY: u_norm_saved was resized to m_norm, and u_norm is >= m_norm.
        let src = unsafe { scratch.u_norm.get_unchecked(..m_norm) };
        u_norm_saved.copy_from_slice(src);

        let v_recip = Self::newton_reciprocal(&v_norm_buf, scratch);

        let blocks = max(
            2,
            // SAFETY: n > 0 by normalization (n is the normalized bit-width of the divisor).
            unsafe {
                (m_norm.wrapping_add(n).wrapping_sub(1))
                    .checked_div(n)
                    .unwrap_unchecked()
            },
        );
        let pad_len = blocks.wrapping_mul(n);
        let mut a_pad = replace(&mut scratch.newton_a_pad, ScratchBuffer::acquire(0));
        a_pad.clear();
        a_pad.resize(pad_len, 0);
        // SAFETY: a_pad was resized to pad_len >= m_norm.
        unsafe { a_pad.get_unchecked_mut(..m_norm) }.copy_from_slice(&u_norm_saved);

        let top_start = blocks.wrapping_sub(1).wrapping_mul(n);
        let total_quo_len = blocks.wrapping_sub(1).wrapping_mul(n).wrapping_add(2);
        let mut total_quo = replace(&mut scratch.newton_total_quo, ScratchBuffer::acquire(0));
        total_quo.clear();
        total_quo.resize(total_quo_len, 0);

        let mut cur_2n = replace(&mut scratch.newton_cur_2n, ScratchBuffer::acquire(0));
        cur_2n.clear();
        cur_2n.resize(n.wrapping_mul(2), 0);

        // P2 fix: Hoist q_i and r_next out of the loop — allocate once, reuse.
        let q_i_len = n.wrapping_add(2);
        let mut q_i = replace(&mut scratch.newton_q_i, ScratchBuffer::acquire(0));
        q_i.clear();
        q_i.resize(q_i_len, 0);
        let mut r_cur = replace(&mut scratch.newton_r_cur, ScratchBuffer::acquire(0));
        r_cur.clear();
        r_cur.resize(n, 0);
        // SAFETY: pad_len - top_start == n, which matches r_cur.len().
        r_cur.copy_from_slice(unsafe { a_pad.get_unchecked(top_start..pad_len) });
        let mut r_next = replace(&mut scratch.newton_r_next, ScratchBuffer::acquire(0));
        r_next.clear();
        r_next.resize(n, 0);

        for i in (0..blocks.wrapping_sub(1)).rev() {
            let blk_start = i.wrapping_mul(n);
            let blk_end = blk_start.wrapping_add(n);
            // SAFETY: blk_end - blk_start == n, which fits in cur_2n.
            unsafe { cur_2n.get_unchecked_mut(..n) }
                .copy_from_slice(unsafe { a_pad.get_unchecked(blk_start..blk_end) });
            // SAFETY: cur_2n is size 2n, and r_cur is size n.
            unsafe { cur_2n.get_unchecked_mut(n..n.wrapping_mul(2)) }.copy_from_slice(&r_cur);

            q_i.fill(0);
            r_next.fill(0);
            newton_div_2n1n_with_reciprocal(
                &cur_2n,
                &v_norm_buf,
                &v_recip,
                &mut q_i,
                &mut r_next,
                scratch,
            );

            let q_i_active = active_len(&q_i);
            if q_i_active > 0 {
                // SAFETY: total_quo is sized to cover all blk_start, and q_i_active <= q_i.len().
                let _ = Addition::add_slice_in_place(
                    unsafe { total_quo.get_unchecked_mut(blk_start..) },
                    unsafe { q_i.get_unchecked(..q_i_active) },
                );
            }
            // Swap r_cur and r_next (no allocation, just pointer swap)
            swap(&mut r_cur, &mut r_next);
        }

        write_newton_outputs(shift, &total_quo, &mut r_cur, quotient_out, rem_out);

        scratch.newton_v_norm = v_norm_buf;
        scratch.newton_u_norm = u_norm_saved;
        scratch.newton_a_pad = a_pad;
        scratch.newton_total_quo = total_quo;
        scratch.newton_cur_2n = cur_2n;
        scratch.newton_q_i = q_i;
        scratch.newton_r_cur = r_cur;
        scratch.newton_r_next = r_next;
    }
}

/// Divide a `2n`-limb number `num` by an `n`-limb normalized divisor `den`
/// using a precomputed reciprocal `v_recip`.
///
/// Reuses scratch buffers from `DivScratch` to avoid per-call heap allocations.
#[allow(
    clippy::too_many_lines,
    reason = "Newton division keeps the quotient estimate, correction, and exact fallback in one hot-path state machine; all slices are verified relative to n."
)]
fn newton_div_2n1n_with_reciprocal(
    num: &[Limb],
    den: &[Limb],
    v_recip: &InternalArbiUint,
    quo: &mut [Limb],
    rem: &mut [Limb],
    scratch: &mut DivScratch,
) {
    let n = den.len();
    let v_limbs = v_recip.limbs();

    // Q_0 = floor((num * V) / 2^(64 * 2n))
    let prod_len = num.len().wrapping_add(v_limbs.len());
    scratch.v_padded.clear();
    scratch.v_padded.resize(prod_len, 0);
    Multiplication::mul_limbs_with_scratch(
        num,
        v_limbs,
        &mut scratch.v_padded,
        &mut scratch.mul_scratch,
    );

    let double_n = n.wrapping_mul(2);

    // Extract quotient estimate from high limbs of product (avoid to_vec)
    let q0_start = min(double_n, scratch.v_padded.len());
    let q0_raw_len = scratch.v_padded.len().wrapping_sub(q0_start);
    // Reuse dummy_quot's backing storage for q0
    scratch
        .dummy_quot
        .resize(if q0_raw_len > 0 { q0_raw_len } else { 1 });
    let q0_limbs = scratch.dummy_quot.limbs_mut();
    if q0_raw_len > 0 {
        // SAFETY: q0_limbs has length q0_raw_len. v_padded has length >= q0_start + q0_raw_len.
        unsafe { q0_limbs.get_unchecked_mut(..q0_raw_len) }
            .copy_from_slice(unsafe { scratch.v_padded.get_unchecked(q0_start..) });
    } else {
        // SAFETY: q0_limbs was resized to at least 1.
        unsafe {
            *q0_limbs.get_unchecked_mut(0) = 0;
        }
    }
    scratch.dummy_quot.normalize();

    // Remainder: num - q0 * den fits in n + 1 limbs.
    let q0_sl = scratch.dummy_quot.limbs();
    let check_len = min(n.wrapping_add(1), q0_sl.len().wrapping_add(n));

    // Pad q0 into v_padded (reuse), and den into u_norm (reuse)
    scratch.v_padded.clear();
    scratch.v_padded.resize(check_len, 0);
    let copy_q = min(q0_sl.len(), check_len);
    // SAFETY: copy_q is <= check_len (v_padded length) and <= q0_sl.len().
    unsafe { scratch.v_padded.get_unchecked_mut(..copy_q) }
        .copy_from_slice(unsafe { q0_sl.get_unchecked(..copy_q) });

    scratch.q_den_low.clear();
    scratch.q_den_low.resize(check_len, 0);

    scratch.den_pad.clear();
    scratch.den_pad.resize(check_len, 0);
    let copy_d = min(den.len(), check_len);
    // SAFETY: copy_d is <= check_len (den_pad length) and <= den.len().
    unsafe { scratch.den_pad.get_unchecked_mut(..copy_d) }
        .copy_from_slice(unsafe { den.get_unchecked(..copy_d) });

    LowProduct::mul(
        &mut scratch.q_den_low,
        &scratch.v_padded,
        &scratch.den_pad,
        check_len,
        &mut scratch.mul_scratch,
    );

    // Compute remainder in-place using u_norm scratch:
    // R = num[0..check_len] - (Q₀ * den)[0..check_len]
    let num_slice_len = min(num.len(), check_len);
    scratch.u_norm.clear();
    scratch.u_norm.resize(check_len, 0);
    // SAFETY: num_slice_len <= check_len (u_norm length) and <= num.len().
    unsafe { scratch.u_norm.get_unchecked_mut(..num_slice_len) }
        .copy_from_slice(unsafe { num.get_unchecked(..num_slice_len) });
    let borrow = Addition::sub_slice_in_place(&mut scratch.u_norm, &scratch.q_den_low);

    // If borrow != 0, Q₀ was too large: num < Q₀ * den, so the subtraction
    // wrapped modulo 2^(64 * check_len). Correct by adding den back to the
    // remainder and decrementing Q₀.
    if borrow != 0 {
        let _ = Addition::add_slice_in_place(&mut scratch.u_norm, den);
        let q0_mut = scratch.dummy_quot.limbs_mut();
        let mut br: Limb = 1;
        for x in q0_mut.iter_mut() {
            if br == 0 {
                break;
            }
            let (diff, b) = x.overflowing_sub(br);
            *x = diff;
            br = Limb::from(b);
        }
    }

    // Normalize remainder: strip trailing zeros so that cmp_limbs' length-first
    // comparison produces the correct mathematical ordering.
    while scratch.u_norm.last() == Some(&0) && scratch.u_norm.len() > 1 {
        let _ = scratch.u_norm.pop();
    }

    // Correction loop: the Newton reciprocal guarantees the quotient
    // estimate is off by at most 1-2. If that invariant is violated, use
    // exact Algorithm D rather than allowing an unbounded subtraction loop.
    for _ in 0..2 {
        if InternalArbiUint::cmp_limbs(&scratch.u_norm, den) == Ordering::Less {
            break;
        }
        // Increment q0 in dummy_quot
        let q0_mut = scratch.dummy_quot.limbs_mut();
        let mut carry: Limb = 1;
        for x in q0_mut.iter_mut() {
            if carry == 0 {
                break;
            }
            let (sum, c) = x.overflowing_add(carry);
            *x = sum;
            carry = Limb::from(c);
        }
        let borrow_c = Addition::sub_slice_in_place(&mut scratch.u_norm, den);
        if borrow_c != 0 && scratch.u_norm.len() > den.len() {
            let den_len = den.len();
            // SAFETY: Addition::sub_slice_in_place requires u_norm >= den.len().
            let _ = Addition::propagate_borrow(
                unsafe { scratch.u_norm.get_unchecked_mut(den_len..) },
                borrow_c,
            );
        }
        // Re-normalize after subtraction
        while scratch.u_norm.last() == Some(&0) && scratch.u_norm.len() > 1 {
            let _ = scratch.u_norm.pop();
        }
    }

    if InternalArbiUint::cmp_limbs(&scratch.u_norm, den) != Ordering::Less {
        fallback_algorithm_d(num, den, quo, rem, scratch);
        return;
    }

    // Write results
    quo.fill(0);
    let q0_final = scratch.dummy_quot.limbs();
    let q_copy = min(quo.len(), q0_final.len());
    // SAFETY: q_copy <= quo.len() and <= q0_final.len().
    unsafe { quo.get_unchecked_mut(..q_copy) }
        .copy_from_slice(unsafe { q0_final.get_unchecked(..q_copy) });

    rem.fill(0);
    // Strip trailing zeros for accurate copy
    let mut rem_active = scratch.u_norm.len();
    while rem_active > 0
        // SAFETY: rem_active is checked to be > 0 in the loop condition.
        && unsafe { *scratch.u_norm.get_unchecked(rem_active.wrapping_sub(1)) } == 0
    {
        rem_active = rem_active.wrapping_sub(1);
    }
    let r_copy = min(rem.len(), rem_active);
    if r_copy > 0 {
        // SAFETY: r_copy <= rem.len() and <= rem_active (which is <= u_norm.len()).
        unsafe { rem.get_unchecked_mut(..r_copy) }
            .copy_from_slice(unsafe { scratch.u_norm.get_unchecked(..r_copy) });
    }
}

/// Exact divider for the block whose Newton estimate fell outside the
/// reciprocal's error bound.
fn fallback_algorithm_d(
    num: &[Limb],
    den: &[Limb],
    quo: &mut [Limb],
    rem: &mut [Limb],
    scratch: &mut DivScratch,
) {
    let numerator = InternalArbiUint::from_limbs(num.to_vec());
    let denominator = InternalArbiUint::from_limbs(den.to_vec());
    let mut quotient = InternalArbiUint::zero();
    let mut remainder = InternalArbiUint::zero();
    Division::algorithm_d(
        &numerator,
        &denominator,
        &mut quotient,
        &mut remainder,
        scratch,
    );

    quo.fill(0);
    let quotient_limbs = quotient.limbs();
    let quotient_len = min(quo.len(), quotient_limbs.len());
    // SAFETY: quotient_len <= quo.len() and <= quotient_limbs.len().
    unsafe { quo.get_unchecked_mut(..quotient_len) }
        .copy_from_slice(unsafe { quotient_limbs.get_unchecked(..quotient_len) });

    rem.fill(0);
    let remainder_limbs = remainder.limbs();
    let remainder_len = min(rem.len(), remainder_limbs.len());
    // SAFETY: remainder_len <= rem.len() and <= remainder_limbs.len().
    unsafe { rem.get_unchecked_mut(..remainder_len) }
        .copy_from_slice(unsafe { remainder_limbs.get_unchecked(..remainder_len) });
}

/// Copies the accumulated quotient and the de-normalized remainder out.
fn write_newton_outputs(
    shift: u32,
    total_quo: &[Limb],
    r_cur: &mut ScratchBuffer,
    quotient_out: &mut InternalArbiUint,
    rem_out: &mut InternalArbiUint,
) {
    let quotient_len = active_len(total_quo);
    // SAFETY: `active_len` starts at `total_quo.len()` and only decreases.
    let active_quotient = unsafe { total_quo.get_unchecked(..quotient_len) };
    quotient_out.clone_from_slice(active_quotient);

    // P10 fix: Write remainder directly into rem_out instead of double-copying.
    if shift > 0 && !r_cur.is_empty() {
        let len = r_cur.len();
        // SAFETY: r_cur has valid pointers and shift is in (0, LIMB_BITS).
        let _ = unsafe { ArchKernels::rshift_unchecked(r_cur.as_mut_ptr(), len, shift) };
    }
    while r_cur.last() == Some(&0) {
        let _ = r_cur.pop();
    }
    rem_out.clone_from_slice(r_cur);
}

/// Returns the length of `limbs` with its trailing zero limbs discounted.
fn active_len(limbs: &[Limb]) -> usize {
    let mut len = limbs.len();
    // SAFETY: the right operand is evaluated only while `len > 0`, and `len`
    // starts at `limbs.len()` and only decreases.
    while len > 0 && unsafe { *limbs.get_unchecked(len.wrapping_sub(1)) == 0 } {
        len = len.wrapping_sub(1);
    }
    len
}
