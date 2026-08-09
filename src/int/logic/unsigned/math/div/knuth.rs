//! Knuth's Algorithm D, the basecase of the division tower.
//!
//! Every other divider bottoms out here. Quotient digits are estimated with the
//! 3-by-2 reciprocal in [`reciprocal3by2`](super::reciprocal3by2), so the inner
//! loop uses only multiplications; the hardware divide runs once per division,
//! when that reciprocal is built.

use core::{
    mem::{MaybeUninit, replace},
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{
    ArchKernels, BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb,
};

/// Number of limbs the Algorithm D fast path normalizes on the stack before
/// falling back to pooled heap scratch. Covers common small/medium divisions
/// (dividend up to ~4000 bits with a sub-`BURNIKEL_ZIEGLER_THRESHOLD` divisor)
/// while keeping the two `[Limb; DIV_STACK_LIMBS]` frames at 1 KiB total on
/// 64-bit targets.
const DIV_STACK_LIMBS: usize = 64;

impl Division {
    /// Computes the quotient and remainder of `num_a / den_b` with Algorithm D.
    ///
    pub fn algorithm_d(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        let completed = Self::algorithm_d_impl::<true, true, true>(
            num_a,
            den_b,
            quotient_out,
            rem_out,
            Some(scratch),
        );
        debug_assert!(completed, "caller supplied reusable division scratch");
    }

    /// Computes an Algorithm D remainder without retaining quotient limbs.
    pub fn algorithm_d_rem(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        let mut dummy_quot = replace(&mut scratch.dummy_quot, InternalMpUint::zero());
        let completed = Self::algorithm_d_impl::<false, true, true>(
            num_a,
            den_b,
            &mut dummy_quot,
            rem_out,
            Some(scratch),
        );
        debug_assert!(completed, "caller supplied reusable division scratch");
        scratch.dummy_quot = dummy_quot;
    }

    /// Attempts Algorithm D without constructing heap-backed division scratch.
    ///
    /// Returns `false` when the configured dispatch or normalization size
    /// requires the reusable scratch path. `CHECK_TRIVIAL` must be `true`
    /// unless the caller has already proved that the quotient exceeds one.
    pub fn try_algorithm_d_unscratched<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const CHECK_TRIVIAL: bool,
    >(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if CHECK_TRIVIAL
            && Self::trivial::<WRITE_QUOTIENT, WRITE_REMAINDER>(num_a, den_b, quotient_out, rem_out)
        {
            return true;
        }
        let v_limbs = Self::significant_limbs(den_b.limbs());
        let u_limbs = Self::significant_limbs(num_a.limbs());
        if v_limbs.len() >= BURNIKEL_ZIEGLER_THRESHOLD
            && u_limbs.len() > v_limbs.len().wrapping_add(1)
        {
            return false;
        }
        Self::algorithm_d_impl::<WRITE_QUOTIENT, WRITE_REMAINDER, false>(
            num_a,
            den_b,
            quotient_out,
            rem_out,
            None,
        )
    }

    #[allow(
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Knuth Algorithm D is long. 'as' conversions are safe and branchless on every supported limb width."
    )]
    fn algorithm_d_impl<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const CHECK_TRIVIAL: bool,
    >(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch_opt: Option<&mut DivScratch>,
    ) -> bool {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if CHECK_TRIVIAL
            && Self::trivial::<WRITE_QUOTIENT, WRITE_REMAINDER>(num_a, den_b, quotient_out, rem_out)
        {
            return true;
        }
        let v_limbs = Self::significant_limbs(den_b.limbs());
        let u_limbs = Self::significant_limbs(num_a.limbs());

        if v_limbs.len() == 1 {
            // SAFETY: v_limbs.len() == 1
            let v_single = unsafe { *v_limbs.get_unchecked(0) };
            let rem = Self::div_rem_1::<WRITE_QUOTIENT>(u_limbs, v_single, quotient_out);
            if WRITE_REMAINDER {
                if rem == 0 {
                    rem_out.clear();
                } else {
                    *rem_out = InternalMpUint::from_limb(rem);
                }
            }
            return true;
        }

        let n_len = v_limbs.len();
        let m_len = u_limbs.len().wrapping_sub(n_len);

        // SAFETY: v_limbs is not empty
        let shift =
            unsafe { *v_limbs.get_unchecked(v_limbs.len().wrapping_sub(1)) }.leading_zeros();
        let u_norm_len = m_len.wrapping_add(n_len).wrapping_add(1);

        // Fast path: for small operands, normalize the dividend and divisor onto
        // the stack instead of into the `scratch.u_norm` / `scratch.v_norm` heap
        // buffers. Those buffers fall below the arena's
        // `SMALL_BUFFER_DROP_THRESHOLD`, so they are freed (never pooled) after
        // every call, costing 2-3 malloc/free pairs per division. Keeping that
        // scratch on the stack makes small divisions allocation-free.
        if u_norm_len <= DIV_STACK_LIMBS {
            if shift == 0 {
                let mut u_stack: [MaybeUninit<Limb>; DIV_STACK_LIMBS] =
                    [MaybeUninit::uninit(); DIV_STACK_LIMBS];
                let u_in_len = u_limbs.len();

                // SAFETY: `u_in_len < u_norm_len <= DIV_STACK_LIMBS`, so the
                // `MaybeUninit` destination slice is in bounds. Writing through
                // that slice initializes every input limb without first
                // creating a `Limb` reference to uninitialized storage.
                let u_dst = unsafe { u_stack.get_unchecked_mut(..u_in_len) };
                let _ = shl_bits_into(u_dst, u_limbs, 0);
                // SAFETY: `u_in_len < u_norm_len <= DIV_STACK_LIMBS`, so this
                // is the one initialized high limb required by Algorithm D.
                unsafe {
                    let _ = u_stack.get_unchecked_mut(u_in_len).write(0);
                }

                // SAFETY: the preceding copy and zero write initialized all
                // `u_norm_len` limbs; `v_limbs` is already normalized because
                // `shift == 0`, and both slices remain within their owners.
                let u_norm =
                    unsafe { from_raw_parts_mut(u_stack.as_mut_ptr().cast::<Limb>(), u_norm_len) };
                knuth_d_divide::<WRITE_QUOTIENT, WRITE_REMAINDER>(
                    u_norm,
                    v_limbs,
                    n_len,
                    m_len,
                    shift,
                    quotient_out,
                    rem_out,
                );
                return true;
            }

            let mut v_stack: [MaybeUninit<Limb>; DIV_STACK_LIMBS] =
                [MaybeUninit::uninit(); DIV_STACK_LIMBS];
            let mut u_stack: [MaybeUninit<Limb>; DIV_STACK_LIMBS] =
                [MaybeUninit::uninit(); DIV_STACK_LIMBS];
            let u_in_len = u_limbs.len();

            // SAFETY: `n_len <= u_norm_len <= DIV_STACK_LIMBS`; a
            // `MaybeUninit<Limb>` slice may cover uninitialized storage, and
            // the helper writes each element before a `Limb` slice is formed.
            let v_dst = unsafe { v_stack.get_unchecked_mut(..n_len) };
            let _ = shl_bits_into(v_dst, v_limbs, shift);
            // SAFETY: `u_in_len = m_len + n_len < u_norm_len <= DIV_STACK_LIMBS`.
            // The `MaybeUninit` slice is in bounds, and the shift helper
            // initializes this entire prefix before it is read as `Limb`.
            let u_dst = unsafe { u_stack.get_unchecked_mut(..u_in_len) };
            let u_carry = shl_bits_into(u_dst, u_limbs, shift);
            // SAFETY: `u_norm_len = u_in_len + 1 <= DIV_STACK_LIMBS`; this writes
            // the sole limb not initialized by `shl_bits_into`.
            unsafe {
                let _ = u_stack.get_unchecked_mut(u_in_len).write(u_carry);
            }

            // SAFETY: the two writes above initialized exactly `u_norm_len` limbs
            // in `u_stack` and `n_len` limbs in `v_stack`; both lengths are
            // bounded by `DIV_STACK_LIMBS`, and the slices do not outlive arrays.
            let u_norm =
                unsafe { from_raw_parts_mut(u_stack.as_mut_ptr().cast::<Limb>(), u_norm_len) };
            // SAFETY: all `n_len` limbs were initialized by `shl_bits_into`, and
            // `n_len <= DIV_STACK_LIMBS`.
            let v_norm = unsafe { from_raw_parts(v_stack.as_ptr().cast::<Limb>(), n_len) };
            knuth_d_divide::<WRITE_QUOTIENT, WRITE_REMAINDER>(
                u_norm,
                v_norm,
                n_len,
                m_len,
                shift,
                quotient_out,
                rem_out,
            );
            return true;
        }

        // Slow path: operands too large for the stack buffer; normalize into the
        // pooled heap scratch buffers.
        let Some(scratch) = scratch_opt else {
            return false;
        };
        scratch
            .v_norm
            .reset_with_capacity(v_limbs.len().wrapping_add(1));
        scratch
            .u_norm
            .reset_with_capacity(u_limbs.len().wrapping_add(1));
        Self::shift_limbs_left(v_limbs, shift, &mut scratch.v_norm);
        Self::shift_limbs_left(u_limbs, shift, &mut scratch.u_norm);
        scratch.u_norm.resize(u_norm_len, 0);

        knuth_d_divide::<WRITE_QUOTIENT, WRITE_REMAINDER>(
            scratch.u_norm.as_mut_slice(),
            scratch.v_norm.as_slice(),
            n_len,
            m_len,
            shift,
            quotient_out,
            rem_out,
        );

        true
    }
}

/// Shifts `src` left by `bits` (`0 <= bits < LIMB_BITS`) into uninitialized
/// destination limbs, returning the carry shifted out of the most-significant
/// source limb.
///
/// `dst.len()` must be `>= src.len()`; only the first `src.len()` limbs of
/// `dst` are written.
#[allow(
    clippy::inline_always,
    reason = "Called on the division hot path; inlining removes call overhead for the short normalization loop."
)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "LIMB_BITS is at most 64 and always fits in u32 on all target pointer widths."
)]
#[inline(always)]
fn shl_bits_into(dst: &mut [MaybeUninit<Limb>], src: &[Limb], bits: u32) -> Limb {
    if bits == 0 {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            let _ = d.write(s);
        }
        return 0;
    }
    let c_shift = (LIMB_BITS as u32).wrapping_sub(bits);
    let mut carry: Limb = 0;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let _ = d.write(s.wrapping_shl(bits) | carry);
        carry = s.wrapping_shr(c_shift);
    }
    carry
}

/// Core of Knuth's Algorithm D operating on pre-normalized limb slices.
///
/// `u_norm` (length `m_len + n_len + 1`) is the shifted dividend and `v_norm`
/// (length `n_len`, top bit set) the shifted divisor. Writes `m_len + 1`
/// quotient limbs into `quotient_out`, and the de-normalized remainder into
/// `rem_out`. After return `u_norm[0..n_len]` holds the still-shifted remainder.
#[allow(
    clippy::inline_always,
    reason = "Single hot call site per division; inlining exposes the quotient loop to the caller's register allocation and branch pruning."
)]
#[inline(always)]
fn knuth_d_divide<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool>(
    u_norm: &mut [Limb],
    v_norm: &[Limb],
    n_len: usize,
    m_len: usize,
    shift: u32,
    quotient_out: &mut InternalMpUint,
    rem_out: &mut InternalMpUint,
) {
    if WRITE_QUOTIENT {
        // SAFETY: quotient_out has capacity for m_len + 1 limbs, which we immediately initialize.
        let _ = unsafe { quotient_out.ensure_capacity_set_len_get_limbs(m_len.wrapping_add(1)) };
    }

    // SAFETY: n_len >= 2
    let vn1 = unsafe { *v_norm.get_unchecked(n_len.wrapping_sub(1)) };
    // SAFETY: n_len >= 2
    let vn2 = unsafe { *v_norm.get_unchecked(n_len.wrapping_sub(2)) };

    // Precompute the 3-by-2 reciprocal of the divisor top once; each quotient
    // digit is then estimated with multiplies instead of a hardware division.
    let dinv = Division::invert_pi1(vn1, vn2);
    // Runtime backend selection is process-stable, so one cached pointer serves
    // every quotient digit without a OnceLock check inside the Algorithm-D loop.
    let sub_mul = ArchKernels::selected_sub_mul_limbs_unchecked();

    for j in (0..=m_len).rev() {
        let jn = j.wrapping_add(n_len);
        // SAFETY: jn < u_norm.len()
        let uj_n = unsafe { *u_norm.get_unchecked(jn) };
        // SAFETY: jn >= 2
        let uj_n1 = unsafe { *u_norm.get_unchecked(jn.wrapping_sub(1)) };

        // SAFETY: jn >= 2, so jn - 2 is a valid index into u_norm.
        let ujn2 = unsafe { *u_norm.get_unchecked(jn.wrapping_sub(2)) };
        // Estimate the quotient digit from the top three dividend limbs and the
        // reciprocal. The loop invariant `uj_n <= vn1` (running remainder is
        // below the divisor) makes the exact top-two-limb tie the only case
        // where the true quotient reaches `B`, handled explicitly; every other
        // case gives `q_hat` in `{q_j, q_j + 1}`, so the mul-sub plus single
        // add-back below still corrects a possible one-too-large estimate.
        let q_hat = if uj_n == vn1 && uj_n1 == vn2 {
            Limb::MAX
        } else {
            Division::udiv_qr_3by2(uj_n, uj_n1, ujn2, vn1, vn2, dinv).0
        };

        let end_idx = j.wrapping_add(n_len).wrapping_add(1);
        // SAFETY: j..end_idx is within bounds of u_norm
        let u_window = unsafe { u_norm.get_unchecked_mut(j..end_idx) };
        let borrow = Division::mul_sub_in_place(u_window, v_norm, q_hat, sub_mul);

        if WRITE_QUOTIENT {
            // SAFETY: quotient_out was initialized to m_len + 1 limbs above.
            let quotient = quotient_out.limbs_mut();
            // SAFETY: j <= m_len, so j is within the initialized quotient.
            unsafe {
                *quotient.get_unchecked_mut(j) = q_hat;
            }
        }
        if borrow != 0 {
            if WRITE_QUOTIENT {
                // SAFETY: quotient_out was initialized to m_len + 1 limbs above.
                let quotient = quotient_out.limbs_mut();
                // SAFETY: j <= m_len, so j is within the initialized quotient.
                unsafe {
                    let val = *quotient.get_unchecked(j);
                    *quotient.get_unchecked_mut(j) = val.wrapping_sub(1);
                }
            }
            // SAFETY: j..end_idx is within bounds of u_norm
            let u_window_add = unsafe { u_norm.get_unchecked_mut(j..end_idx) };
            Division::add_in_place(u_window_add, v_norm);
        }
    }

    if WRITE_QUOTIENT {
        quotient_out.normalize();
    }

    if WRITE_REMAINDER {
        // SAFETY: `0..n_len` is within bounds of the normalized dividend.
        let rem_norm = unsafe { u_norm.get_unchecked(0..n_len) };

        // SAFETY: `rem_out` has capacity for `n_len` limbs, all of which are
        // initialized immediately by the copy below.
        let r_out = unsafe { rem_out.ensure_capacity_set_len_get_limbs(n_len) };

        r_out.copy_from_slice(rem_norm);
        if shift != 0 {
            // SAFETY: `r_out` has `n_len` limbs and `shift` is in
            // `1..LIMB_BITS`, so the kernel reads and writes only initialized
            // limbs and the shift count is valid.
            let _ = unsafe { ArchKernels::rshift_unchecked(r_out.as_mut_ptr(), n_len, shift) };
        }
        rem_out.normalize();
    }
}
