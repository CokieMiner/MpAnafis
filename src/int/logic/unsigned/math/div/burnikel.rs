//! Burnikel-Ziegler recursive division.
//!
//! Splits a `2n / n` division into two `3n / 2n` steps, each of which recurses.
//! The recursion bottoms out in Algorithm D below the generated crossover, and
//! hands off to Newton-Raphson above it, so this layer covers the middle band of
//! divisor lengths.

use core::{
    cmp::{Ordering, min},
    hint::unreachable_unchecked,
    mem::{replace, swap},
    slice::from_raw_parts_mut,
};

use super::{
    ArchKernels, BURNIKEL_ZIEGLER_BLOCK_LIMBS, BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division,
    InternalArbiUint, Limb, Multiplication, NEWTON_RAPHSON_THRESHOLD, ScratchBuffer,
};

impl Division {
    /// Divides `num_a` by `den_b` with the Burnikel-Ziegler recursion.
    ///
    #[allow(
        clippy::too_many_lines,
        reason = "Burnikel-Ziegler recursive division is long; non-performance lint."
    )]
    pub fn burnikel_ziegler(
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
        scratch
            .v_norm
            .reset_with_capacity(v_limbs.len().wrapping_add(1));
        scratch
            .u_norm
            .reset_with_capacity(u_limbs.len().wrapping_add(1));
        Self::shift_limbs_left(v_limbs, shift, &mut scratch.v_norm);
        Self::shift_limbs_left(u_limbs, shift, &mut scratch.u_norm);

        let n = scratch.v_norm.len();

        // A normalized `3m / 2m` input is already the exact shape consumed by
        // the recursive kernel below.  The general block driver would pad it
        // with a zero block and perform three top-level steps, although this
        // shape needs only one.  Keeping the normalized buffers local also
        // leaves the scratch fields available to Algorithm D at the recursion
        // basecase; they are restored after the direct call.
        let direct_3n2_len = n
            .checked_mul(3)
            .and_then(|len| len.checked_div(2))
            .filter(|&len| n.is_multiple_of(2) && len == scratch.u_norm.len())
            .filter(|&len| {
                // `bz_div_3n2n` writes the quotient into an `m`-limb buffer,
                // which is exact only while the whole numerator stays below
                // `v_norm * B^m`; equivalently, the top `2m` limbs of the
                // normalized numerator must be smaller than the normalized
                // divisor. Otherwise the quotient needs `m + 1` limbs and the
                // general block driver below must run.
                // SAFETY: the previous filter proved `len == u_norm.len()` and
                // `len == 3 * (n / 2)`, so the window is inside `u_norm`.
                let u_upper = unsafe { scratch.u_norm.get_unchecked(n.wrapping_div(2)..len) };
                // SAFETY: `v_norm` holds exactly the `n` normalized divisor limbs.
                InternalArbiUint::cmp_limbs(u_upper, scratch.v_norm.as_slice()) == Ordering::Less
            });
        if let Some(normalized_3n2_len) = direct_3n2_len {
            let v_norm = replace(&mut scratch.v_norm, ScratchBuffer::acquire(0));
            let u_norm = replace(&mut scratch.u_norm, ScratchBuffer::acquire(0));
            let half = n.wrapping_div(2);
            let rem_len = half
                .checked_mul(2)
                .expect("normalized divisor length fits the remainder width");

            let mut q = replace(&mut scratch.bz_q, ScratchBuffer::acquire(0));
            q.reset_with_capacity(half);
            q.resize(half, 0);
            let mut rem_final = replace(&mut scratch.bz_rem_final, ScratchBuffer::acquire(0));
            rem_final.reset_with_capacity(rem_len);
            rem_final.resize(rem_len, 0);

            // SAFETY: `direct_len == 3 * half`; the normalized numerator is
            // laid out as low `a0` followed by the two-limb `a21` block.
            let a21 = unsafe { u_norm.get_unchecked(half..normalized_3n2_len) };
            // SAFETY: `half <= u_norm.len()` by the direct-shape equality.
            let a0 = unsafe { u_norm.get_unchecked(..half) };
            bz_div_3n2n(a21, a0, &v_norm, &mut q, &mut rem_final, scratch);

            if shift > 0 {
                // SAFETY: the direct kernel writes exactly `2 * half` limbs;
                // `shift` is in `1..LIMB_BITS` when this branch is reached.
                let _ = unsafe {
                    ArchKernels::rshift_unchecked(rem_final.as_mut_ptr(), rem_len, shift)
                };
            }

            // SAFETY: the direct kernel initialized every quotient limb.
            let q_out = unsafe { quotient_out.ensure_capacity_set_len_get_limbs(q.len()) };
            q_out.copy_from_slice(&q);
            quotient_out.normalize();

            // SAFETY: the direct kernel initialized every remainder limb.
            let r_out = unsafe { rem_out.ensure_capacity_set_len_get_limbs(rem_final.len()) };
            r_out.copy_from_slice(&rem_final);
            rem_out.normalize();

            scratch.v_norm = v_norm;
            scratch.u_norm = u_norm;
            scratch.bz_q = q;
            scratch.bz_rem_final = rem_final;
            return;
        }

        // A normalized `2n / n` input can use the two-step recursive kernel
        // directly after removing its one possible leading quotient limb.
        // The normalized divisor `V` has its top bit set, so `B^n / 2 <= V`,
        // while the `n`-limb upper numerator block `H` is always below `B^n`.
        // Hence `H < 2V` and its quotient digit is zero or one.  Subtracting
        // that digit leaves `H < V`, the precondition required by
        // `bz_div_2n1n`: its first `3m / 2m` step cannot overflow its `m`-limb
        // quotient buffer.
        let direct_2n_len = n
            .checked_mul(2)
            .filter(|&len| n.is_multiple_of(2) && len == scratch.u_norm.len());
        if let Some(normalized_2n_len) = direct_2n_len {
            let v_norm = replace(&mut scratch.v_norm, ScratchBuffer::acquire(0));
            let mut u_norm = replace(&mut scratch.u_norm, ScratchBuffer::acquire(0));
            let mut q = replace(&mut scratch.bz_q, ScratchBuffer::acquire(0));
            let q_capacity = n
                .checked_add(1)
                .expect("normalized quotient width fits the destination");
            q.reset_with_capacity(q_capacity);
            q.resize(n, 0);
            let mut rem_final = replace(&mut scratch.bz_rem_final, ScratchBuffer::acquire(0));
            rem_final.reset_with_capacity(n);
            rem_final.resize(n, 0);

            // SAFETY: `direct_len == 2 * n`; the upper half is exactly one
            // normalized divisor wide.
            let upper = unsafe { u_norm.get_unchecked_mut(n..normalized_2n_len) };
            let q_top: Limb = if InternalArbiUint::cmp_limbs(upper, &v_norm) == Ordering::Less {
                0
            } else {
                let borrow = Self::sub_limbs_in_place(upper, &v_norm);
                debug_assert_eq!(borrow, 0, "leading quotient subtraction must not borrow");
                1
            };

            // SAFETY: `u_norm` has exactly `2 * n` limbs and `q`/`rem_final`
            // have exactly `n` limbs, matching `bz_div_2n1n`'s contract.
            bz_div_2n1n(&u_norm, &v_norm, &mut q, &mut rem_final, scratch);
            q.push(q_top);

            if shift > 0 {
                // SAFETY: the direct kernel writes exactly `n` remainder
                // limbs; `shift` is in `1..LIMB_BITS` here.
                let _ = unsafe { ArchKernels::rshift_unchecked(rem_final.as_mut_ptr(), n, shift) };
            }

            // SAFETY: the direct kernel initialized every quotient limb.
            let q_out = unsafe { quotient_out.ensure_capacity_set_len_get_limbs(q.len()) };
            q_out.copy_from_slice(&q);
            quotient_out.normalize();

            // SAFETY: the direct kernel initialized every remainder limb.
            let r_out = unsafe { rem_out.ensure_capacity_set_len_get_limbs(rem_final.len()) };
            r_out.copy_from_slice(&rem_final);
            rem_out.normalize();

            scratch.v_norm = v_norm;
            scratch.u_norm = u_norm;
            scratch.bz_q = q;
            scratch.bz_rem_final = rem_final;
            return;
        }

        // The base block size is its own tuning constant, independent of the
        // dispatch threshold: the threshold decides whether this tier runs at
        // all (and is `usize::MAX - 1` when disabled), while the block decides
        // the recursion's geometry and must stay sane even for a forced run.
        // Doubling stops as soon as the block covers the divisor, so `b_len`
        // is always in `[n, 2n)` and every allocation below is bounded by the
        // operands.
        let mut b_len = BURNIKEL_ZIEGLER_BLOCK_LIMBS;
        while b_len < n {
            b_len = b_len.saturating_mul(2);
        }
        let m = b_len.wrapping_div(2);

        let pad_b = b_len.wrapping_sub(n);
        if pad_b > 0 {
            let mut new_v = replace(&mut scratch.v_padded, ScratchBuffer::acquire(0));
            new_v.reset_with_capacity(b_len);
            new_v.resize(b_len, 0);
            // SAFETY: pad_b <= b_len and scratch.v_norm.len() == n == b_len - pad_b
            unsafe {
                new_v
                    .get_unchecked_mut(pad_b..)
                    .copy_from_slice(&scratch.v_norm);
            }
            scratch.v_padded = replace(&mut scratch.v_norm, new_v);
        }

        let a_base_len = scratch.u_norm.len().wrapping_add(pad_b);
        let mut a_blocks = a_base_len.div_ceil(m);
        a_blocks = a_blocks.wrapping_add(1); // Extra zero block so R < B
        if a_blocks < 3 {
            a_blocks = 3;
        }

        let mut a_pad = replace(&mut scratch.bz_a_pad, ScratchBuffer::acquire(0));
        let a_pad_len = a_blocks.wrapping_mul(m);
        a_pad.reset_with_capacity(a_pad_len);
        a_pad.resize(a_pad_len, 0);
        // SAFETY: bounds are mathematically valid by block selection
        unsafe {
            a_pad
                .get_unchecked_mut(pad_b..pad_b.wrapping_add(scratch.u_norm.len()))
                .copy_from_slice(&scratch.u_norm);
        }

        scratch.v_padded.reset_with_capacity(scratch.v_norm.len());
        ScratchBuffer::clone_into(&mut scratch.v_norm, &mut scratch.v_padded);
        let b_padded = replace(&mut scratch.v_padded, ScratchBuffer::acquire(0));

        let mut q = replace(&mut scratch.bz_q, ScratchBuffer::acquire(0));
        let q_len = a_blocks.wrapping_sub(1).wrapping_mul(m);
        q.reset_with_capacity(q_len);
        q.resize(q_len, 0);

        let mut r0 = replace(&mut scratch.bz_r0, ScratchBuffer::acquire(0));
        let remainder_len = m.wrapping_mul(2);
        r0.reset_with_capacity(remainder_len);
        r0.resize(remainder_len, 0);
        let mut r1 = replace(&mut scratch.bz_r1_iter, ScratchBuffer::acquire(0));
        r1.reset_with_capacity(remainder_len);
        r1.resize(remainder_len, 0);

        // R = A_{a_blocks-1} * beta^m + A_{a_blocks-2}
        let m2 = m.wrapping_mul(2);
        let block_1 = a_blocks.wrapping_sub(1).wrapping_mul(m);
        let block_2 = a_blocks.wrapping_sub(2).wrapping_mul(m);
        let block_max = a_blocks.wrapping_mul(m);

        // SAFETY: block variables are within bounds of a_pad, and m..m2, 0..m within bounds of r0
        unsafe {
            r0.get_unchecked_mut(m..m2)
                .copy_from_slice(a_pad.get_unchecked(block_1..block_max));
            r0.get_unchecked_mut(0..m)
                .copy_from_slice(a_pad.get_unchecked(block_2..block_1));
        }

        // Double-buffered iteration: r0 is always the current remainder. Each
        // result is written into r1, then the owned scratch buffers are
        // exchanged; swapping their pointer-bearing structs is equivalent to
        // swapping mutable references and avoids copying any limb storage.
        for i in (0..a_blocks.wrapping_sub(2)).rev() {
            let start = i.wrapping_mul(m);
            let end = i.wrapping_add(1).wrapping_mul(m);
            // SAFETY: start and end are mathematically guaranteed to be within bounds of q and a_pad by construction of the BZ algorithm
            let (q_i, a_i) = unsafe {
                (
                    q.get_unchecked_mut(start..end),
                    a_pad.get_unchecked(start..end),
                )
            };
            bz_div_3n2n(&r0, a_i, &b_padded, q_i, &mut r1, scratch);
            swap(&mut r0, &mut r1);
        }

        let mut rem_final = replace(&mut scratch.bz_rem_final, ScratchBuffer::acquire(0));
        rem_final.reset_with_capacity(n);
        rem_final.resize(n, 0);
        let pad_b_plus_n = pad_b.wrapping_add(n);
        // SAFETY: pad_b and pad_b_plus_n are within bounds of the final remainder buffer
        unsafe {
            rem_final
                .as_mut_slice()
                .copy_from_slice(r0.get_unchecked(pad_b..pad_b_plus_n));
        }

        if shift > 0 {
            // SAFETY: rem_final has n limbs; shift is in (0, LIMB_BITS).
            let _ = unsafe { ArchKernels::rshift_unchecked(rem_final.as_mut_ptr(), n, shift) };
        }

        // SAFETY: quotient_out immediately initialized.
        let q_out = unsafe { quotient_out.ensure_capacity_set_len_get_limbs(q.len()) };
        q_out.copy_from_slice(&q);
        quotient_out.normalize();

        // SAFETY: rem_out immediately initialized.
        let r_out_mut = unsafe { rem_out.ensure_capacity_set_len_get_limbs(rem_final.len()) };
        r_out_mut.copy_from_slice(&rem_final);
        rem_out.normalize();

        scratch.bz_a_pad = a_pad;
        scratch.v_padded = b_padded;
        scratch.bz_q = q;
        scratch.bz_r0 = r0;
        scratch.bz_r1_iter = r1;
        scratch.bz_rem_final = rem_final;
    }
}

/// Divides the `3m`-limb value `a21 * beta^m + a0` by the `2m`-limb `b`.
fn bz_div_3n2n(
    a21: &[Limb],       // 2m limbs
    a0: &[Limb],        // m limbs
    b: &[Limb],         // 2m limbs
    q_out: &mut [Limb], // m limbs
    r_out: &mut [Limb], // 2m limbs
    scratch: &mut DivScratch,
) {
    let m = a0.len();
    let m2 = m.wrapping_mul(2);
    // SAFETY: caller guarantees b has length 2m and m2 = 2 * m
    let b1 = unsafe { b.get_unchecked(m..m2) };
    // SAFETY: caller guarantees b has length 2m and m < 2m
    let b0 = unsafe { b.get_unchecked(0..m) };
    // SAFETY: caller guarantees a21 has length 2m and m2 = 2 * m
    let a2 = unsafe { a21.get_unchecked(m..m2) };
    // SAFETY: caller guarantees a21 has length 2m and m < 2m
    let a1 = unsafe { a21.get_unchecked(0..m) };

    let r1_c = if InternalArbiUint::cmp_limbs(a2, b1) == Ordering::Equal {
        for x in q_out.iter_mut() {
            *x = Limb::MAX;
        }
        // SAFETY: r_out is guaranteed to have 2m limbs
        unsafe {
            r_out.get_unchecked_mut(m..m2).copy_from_slice(a1);
            Division::add_limbs_in_place(r_out.get_unchecked_mut(m..m2), b1)
        }
    } else {
        // SAFETY: r_out is guaranteed to have 2m limbs
        let r_upper = unsafe { r_out.get_unchecked_mut(m..m2) };
        bz_div_2n1n(a21, b1, q_out, r_upper, scratch);
        0
    };

    // SAFETY: caller guarantees r_out has length 2m
    unsafe {
        r_out.get_unchecked_mut(0..m).copy_from_slice(a0);
    }

    let mut d_scratch = replace(&mut scratch.bz_d, ScratchBuffer::acquire(0));
    d_scratch.reset_with_capacity(m2);
    d_scratch.resize(m2, 0);
    Multiplication::mul_limbs_with_scratch(q_out, b0, &mut d_scratch, &mut scratch.mul_scratch);

    let borrow = Division::sub_limbs_in_place(r_out, &d_scratch);
    scratch.bz_d = d_scratch;

    if r1_c == 0 && borrow == 1 {
        let mut underflow_count = 0_usize;
        loop {
            let mut q_borrow = 1;
            for x in q_out.iter_mut() {
                if q_borrow == 0 {
                    break;
                }
                let (diff, b_out) = x.overflowing_sub(q_borrow);
                *x = diff;
                q_borrow = Limb::from(b_out);
            }
            let carry = Division::add_limbs_in_place(r_out, b);
            if carry == 1 {
                break;
            }
            underflow_count = underflow_count.wrapping_add(1);
            if underflow_count > 2 {
                #[allow(
                    unsafe_code,
                    reason = "unwrap_unchecked/unreachable_unchecked tells the optimizer this path is impossible, pruning conditional branches."
                )]
                // SAFETY: Burnikel-Ziegler mathematical properties guarantee that the underflow adjustment loop runs at most twice.
                unsafe {
                    unreachable_unchecked()
                }
            }
        }
    }
}

/// Divides a `2n`-limb numerator by an `n`-limb divisor, recursing through two
/// `3n / 2n` steps once the halves are large enough to pay for the split.
fn bz_div_2n1n(
    num: &[Limb],
    den: &[Limb],
    quo: &mut [Limb],
    rem: &mut [Limb],
    scratch: &mut DivScratch,
) {
    let m_len = den.len();
    if m_len >= NEWTON_RAPHSON_THRESHOLD {
        Division::newton_2n1n(num, den, quo, rem, scratch);
        return;
    }
    if m_len <= BURNIKEL_ZIEGLER_THRESHOLD || !m_len.is_multiple_of(2) {
        let mut n_a = replace(&mut scratch.dummy_u, InternalArbiUint::zero());
        let mut n_b = replace(&mut scratch.dummy_rem, InternalArbiUint::zero());
        let mut n_q = replace(&mut scratch.dummy_quot, InternalArbiUint::zero());
        let mut n_r = replace(&mut scratch.mod_rem, InternalArbiUint::zero());
        n_a.clone_from_slice(num);
        n_b.clone_from_slice(den);
        Division::algorithm_d(&n_a, &n_b, &mut n_q, &mut n_r, scratch);

        quo.fill(0);
        let q_limbs = n_q.limbs();
        let q_len = min(quo.len(), q_limbs.len());
        // SAFETY: q_len is computed as min(quo.len(), q_limbs.len()) which bounds both slices
        unsafe {
            quo.get_unchecked_mut(..q_len)
                .copy_from_slice(q_limbs.get_unchecked(..q_len));
        }

        rem.fill(0);
        let r_limbs = n_r.limbs();
        let r_len = min(rem.len(), r_limbs.len());
        // SAFETY: r_len is computed as min(rem.len(), r_limbs.len()) which bounds both slices
        unsafe {
            rem.get_unchecked_mut(..r_len)
                .copy_from_slice(r_limbs.get_unchecked(..r_len));
        }
        scratch.dummy_u = n_a;
        scratch.dummy_rem = n_b;
        scratch.dummy_quot = n_q;
        scratch.mod_rem = n_r;
        return;
    }

    let half = m_len.wrapping_div(2);
    let half2 = half.wrapping_mul(2);
    let half4 = half.wrapping_mul(4);
    // SAFETY: caller guarantees num has length 4*half
    let a32 = unsafe { num.get_unchecked(half2..half4) };
    // SAFETY: caller guarantees num has length 4*half
    let a1 = unsafe { num.get_unchecked(half..half2) };
    // SAFETY: caller guarantees num has length 4*half
    let a0 = unsafe { num.get_unchecked(0..half) };

    // SAFETY: half < quo.len() by preconditions
    let (q_lo, q_hi) = unsafe {
        let len = quo.len();
        let ptr = quo.as_mut_ptr();
        (
            from_raw_parts_mut(ptr, half),
            from_raw_parts_mut(ptr.add(half), len.wrapping_sub(half)),
        )
    };

    let mut r1_scratch = replace(&mut scratch.bz_r1, ScratchBuffer::acquire(0));
    r1_scratch.reset_with_capacity(m_len);
    r1_scratch.resize(m_len, 0);
    bz_div_3n2n(a32, a1, den, q_hi, &mut r1_scratch, scratch);
    bz_div_3n2n(&r1_scratch, a0, den, q_lo, rem, scratch);
    scratch.bz_r1 = r1_scratch;
}
