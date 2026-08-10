//! Chinese Remainder Theorem modulo $B^n - 1$ and associated folding utilities.

#![allow(
    clippy::similar_names,
    reason = "Standard mathematical notation for CRT halves (xp, xm, etc)"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Standard mathematical notation (a, b, h, n, k)"
)]
use super::{
    Addition, ArchKernels, FftPlan, LIMB_BITS, Limb, Multiplication, SSA_BNM1_BASECASE_LIMBS,
    SsaCarry, SsaTransform,
};

/// Namespace for the `B^n - 1` half of the CRT split and its reconstructions.
///
/// The top-level entry points pair one `B^n + 1` transform with one `B^n - 1`
/// product and merge the two residues; this is that second half, together with
/// the scratch layout both halves are cut from and the two reconstructions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaCrt;

impl SsaCrt {
    /// Computes `dst = a * b mod (B^n - 1)`, where `n` is the common width of all
    /// three slices.
    ///
    /// Splits `n` in half at every level until it reaches
    /// [`SSA_BNM1_BASECASE_LIMBS`], so `n` must stay even the whole way down. `n` need
    /// not be a power of two — the planner's
    /// [`crt_half_width`](super::plan::crt_half_width) guarantees the weaker
    /// property that suffices, namely that the odd part of `n` already fits the
    /// basecase.
    #[allow(
        clippy::too_many_lines,
        reason = "CRT orchestration requires sequential multi-step inline processing"
    )]
    pub fn mul_mod_bnm1(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let n = a.len();
        assert_eq!(n, b.len(), "mul_mod_bnm1 widths must match");
        assert_eq!(n, dst.len(), "mul_mod_bnm1 dst width must match");

        if n <= SSA_BNM1_BASECASE_LIMBS {
            // Basecase: perform exact multiplication and fold.
            let prod_len = n.wrapping_mul(2);
            let (prod, scratch_rest) = scratch.split_at_mut(prod_len);

            let mul_scratch_len = Multiplication::required_scratch(n, n);
            let (mul_scratch, _) = scratch_rest.split_at_mut(mul_scratch_len);

            Multiplication::mul_limbs_with_slice_scratch(a, b, prod, mul_scratch);

            // Fold prod into dst
            // SAFETY: prod holds exactly 2n limbs, so the low half is in range.
            dst.copy_from_slice(unsafe { prod.get_unchecked(..n) });
            // SAFETY: prod holds exactly 2n limbs, so the high half is in range.
            let mut carry = Addition::add_slice_in_place(dst, unsafe { prod.get_unchecked(n..) });
            if carry > 0 {
                carry = SsaCarry::add_full_in_place(dst, &[carry]);
                if carry > 0 {
                    let _ = SsaCarry::add_full_in_place(dst, &[carry]);
                }
            }
            return;
        }

        assert!(
            n.is_multiple_of(2),
            "recursive mul_mod_bnm1 width must be even"
        );
        let h = n >> 1;
        let (xp, rest1) = scratch.split_at_mut(h.wrapping_add(1));
        let (xm, rest2) = rest1.split_at_mut(h);

        // 1. Compute xp = a * b mod (B^h + 1)
        {
            let (left_padded, rest3) = rest2.split_at_mut(h.wrapping_add(1));
            let (right_padded, ring_scratch) = rest3.split_at_mut(h.wrapping_add(1));

            // SAFETY: each padded buffer holds exactly h + 1 limbs, so the guard
            // slot at index h and the h-limb span are both in range.
            *unsafe { left_padded.get_unchecked_mut(h) } = 0;
            // SAFETY: the three h-limb spans are complete and disjoint; this writes
            // a_low-a_high directly instead of copying low and subtracting in place.
            let mut borrow_a = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    left_padded.as_mut_ptr(),
                    a.as_ptr(),
                    a.as_ptr().add(h),
                    h,
                )
            };
            if borrow_a > 0 {
                // SAFETY: the padded buffer holds exactly h + 1 limbs.
                let span = unsafe { left_padded.get_unchecked_mut(..h) };
                borrow_a = borrow_a.wrapping_sub(SsaCarry::add_full_in_place(span, &[1]));
                // SAFETY: index h is the guard slot of the h + 1 limb buffer.
                *unsafe { left_padded.get_unchecked_mut(h) } = 1_usize.wrapping_sub(borrow_a);
            }

            // SAFETY: each padded buffer holds exactly h + 1 limbs, so the guard
            // slot at index h and the h-limb span are both in range.
            *unsafe { right_padded.get_unchecked_mut(h) } = 0;
            // SAFETY: same exact-width and disjointness proof as the left operand.
            let mut borrow_b = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    right_padded.as_mut_ptr(),
                    b.as_ptr(),
                    b.as_ptr().add(h),
                    h,
                )
            };
            if borrow_b > 0 {
                // SAFETY: the padded buffer holds exactly h + 1 limbs.
                let span = unsafe { right_padded.get_unchecked_mut(..h) };
                borrow_b = borrow_b.wrapping_sub(SsaCarry::add_full_in_place(span, &[1]));
                // SAFETY: index h is the guard slot of the h + 1 limb buffer.
                *unsafe { right_padded.get_unchecked_mut(h) } = 1_usize.wrapping_sub(borrow_b);
            }

            let modulus_bits = ring_modulus_bits(h);
            // SAFETY: Buffers exactly sized to bounds, sig bits not tracked.
            unsafe {
                SsaTransform::fft_mul_mod_slices(
                    xp,
                    left_padded,
                    right_padded,
                    modulus_bits,
                    None,
                    false,
                    None,
                    ring_scratch,
                );
            }
        }

        // 2. Compute xm = a * b mod (B^h - 1)
        {
            let (left_folded, rest3) = rest2.split_at_mut(h);
            let (right_folded, xm_scratch) = rest3.split_at_mut(h);

            // SAFETY: the destination and both h-limb input halves are complete
            // and disjoint, so the fused kernel writes a_low+a_high in one pass.
            let mut carry_a = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    left_folded.as_mut_ptr(),
                    a.as_ptr(),
                    a.as_ptr().add(h),
                    h,
                )
            };
            if carry_a > 0 {
                carry_a = SsaCarry::add_full_in_place(left_folded, &[carry_a]);
                if carry_a > 0 {
                    let _ = SsaCarry::add_full_in_place(left_folded, &[carry_a]);
                }
            }

            // SAFETY: same exact-width and disjointness proof as the left operand.
            let mut carry_b = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    right_folded.as_mut_ptr(),
                    b.as_ptr(),
                    b.as_ptr().add(h),
                    h,
                )
            };
            if carry_b > 0 {
                carry_b = SsaCarry::add_full_in_place(right_folded, &[carry_b]);
                if carry_b > 0 {
                    let _ = SsaCarry::add_full_in_place(right_folded, &[carry_b]);
                }
            }

            Self::mul_mod_bnm1(xm, left_folded, right_folded, xm_scratch);
        }

        merge_crt_halves(dst, xp, xm, h, n);
    }

    /// Computes `dst = a * a mod (B^n - 1)`, where `n` is the common width of both
    /// slices.
    ///
    /// The same split as [`Self::mul_mod_bnm1`], specialised throughout. Modulo `B^h + 1`
    /// the base `B^h` is `-1`, so `a` reduces to `a_low - a_high` and the square of
    /// the residue is the square of that difference; modulo `B^h - 1` the base is
    /// `1` and the operand is `a_low + a_high`. Every level therefore runs one
    /// forward transform where the general product runs two, and the basecase
    /// reaches the tower's squaring tier rather than its product tier.
    ///
    /// Routing `Self::mul_mod_bnm1(dst, a, a, ..)` instead would recover the
    /// discount only at the basecase, where
    /// [`Multiplication::mul_limbs_with_slice_scratch`] detects the aliased
    /// operands. Every transform level above it would still run two forward
    /// transforms over identical data.
    pub fn sqr_mod_bnm1(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        let n = a.len();
        assert_eq!(n, dst.len(), "sqr_mod_bnm1 dst width must match");

        if n <= SSA_BNM1_BASECASE_LIMBS {
            let prod_len = n.wrapping_mul(2);
            let (prod, scratch_rest) = scratch.split_at_mut(prod_len);
            let (sqr_scratch, _) =
                scratch_rest.split_at_mut(Multiplication::required_sqr_scratch(n));

            Multiplication::sqr_limbs_with_slice_scratch(a, prod, sqr_scratch);

            // SAFETY: prod holds exactly 2n limbs, so the low half is in range.
            dst.copy_from_slice(unsafe { prod.get_unchecked(..n) });
            // SAFETY: prod holds exactly 2n limbs, so the high half is in range.
            let mut carry = Addition::add_slice_in_place(dst, unsafe { prod.get_unchecked(n..) });
            if carry > 0 {
                carry = SsaCarry::add_full_in_place(dst, &[carry]);
                if carry > 0 {
                    let _ = SsaCarry::add_full_in_place(dst, &[carry]);
                }
            }
            return;
        }

        assert!(
            n.is_multiple_of(2),
            "recursive sqr_mod_bnm1 width must be even"
        );
        let h = n >> 1;
        let (xp, rest1) = scratch.split_at_mut(h.wrapping_add(1));
        let (xm, rest2) = rest1.split_at_mut(h);

        // 1. Compute xp = a^2 mod (B^h + 1) from a_low - a_high.
        {
            let (padded, ring_scratch) = rest2.split_at_mut(h.wrapping_add(1));

            // SAFETY: the padded buffer holds exactly h + 1 limbs, so the guard
            // slot at index h and the h-limb span are both in range.
            *unsafe { padded.get_unchecked_mut(h) } = 0;
            // SAFETY: the three h-limb spans are complete and disjoint; this writes
            // a_low-a_high directly instead of copying low and subtracting in place.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    padded.as_mut_ptr(),
                    a.as_ptr(),
                    a.as_ptr().add(h),
                    h,
                )
            };
            if borrow > 0 {
                // SAFETY: the padded buffer holds exactly h + 1 limbs.
                let span = unsafe { padded.get_unchecked_mut(..h) };
                borrow = borrow.wrapping_sub(SsaCarry::add_full_in_place(span, &[1]));
                // SAFETY: index h is the guard slot of the h + 1 limb buffer.
                *unsafe { padded.get_unchecked_mut(h) } = 1_usize.wrapping_sub(borrow);
            }

            let modulus_bits = ring_modulus_bits(h);
            // SAFETY: the operand is one complete guarded coefficient, disjoint from
            // xp, and the ring scratch is sized for this exact modulus width.
            unsafe {
                SsaTransform::fft_sqr_mod_slices(xp, padded, modulus_bits, false, ring_scratch);
            }
        }

        // 2. Compute xm = a^2 mod (B^h - 1) from a_low + a_high.
        {
            let (folded, xm_scratch) = rest2.split_at_mut(h);

            // SAFETY: the destination and both h-limb input halves are complete
            // and disjoint, so the fused kernel writes a_low+a_high in one pass.
            let mut carry = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    folded.as_mut_ptr(),
                    a.as_ptr(),
                    a.as_ptr().add(h),
                    h,
                )
            };
            if carry > 0 {
                carry = SsaCarry::add_full_in_place(folded, &[carry]);
                if carry > 0 {
                    let _ = SsaCarry::add_full_in_place(folded, &[carry]);
                }
            }

            Self::sqr_mod_bnm1(xm, folded, xm_scratch);
        }

        merge_crt_halves(dst, xp, xm, h, n);
    }

    /// Reconstructs the exact product `dst = xp + k * B^n + k` from the two
    /// top-level CRT residues, where `k = (xm - xp) / 2 mod (B^n - 1)` and `n` is
    /// the width of `xm`.
    ///
    /// The entry points' counterpart to [`merge_crt_halves`], and distinct from it in
    /// two ways that are not incidental: this one reconstructs an *exact integer*
    /// rather than a residue, so it never folds a final carry back around the
    /// modulus, and it honours a destination shorter than the full `2n` limbs, which
    /// is how a caller asks for a truncated product width.
    ///
    /// Shared by the tower's product and its square, which arrive here with residues
    /// of identical shape and differ only in how they computed them. `xm` is dead on
    /// entry, so it becomes `k` in place rather than being copied to a third span.
    pub fn merge_exact_product(dst: &mut [Limb], xp: &[Limb], xm: &mut [Limb]) {
        let n = xm.len();
        assert!(n > 0, "the CRT half-width must be nonzero");
        // n is the length of a live xm slice, so on every supported usize width
        // n + 1 fits: the slice itself occupies n * size_of::<Limb>() bytes.
        let expected_xp_len = n.wrapping_add(1);
        assert_eq!(
            xp.len(),
            expected_xp_len,
            "the Fermat residue carries one guard limb above the CRT half-width"
        );

        // D = X_m - X_p mod (B^n - 1)
        let k = xm;
        // SAFETY: xp carries one guard limb above the n-limb residue, so both
        // the n-limb span and the guard slot at index n are in range.
        let mut borrow = Addition::sub_slice_in_place(k, unsafe { xp.get_unchecked(..n) });
        // SAFETY: xp carries one guard limb above the n-limb residue.
        borrow = borrow.wrapping_add(SsaCarry::sub_full_in_place(
            k,
            &[unsafe { *xp.get_unchecked(n) }],
        ));

        // Modulo B^n-1, a borrow of B^n is equivalent to 1.
        let b2 = SsaCarry::sub_full_in_place(k, &[borrow]);
        if b2 > 0 {
            let _ = SsaCarry::sub_full_in_place(k, &[b2]);
        }

        // k = D * 2^{-1} mod (B^n - 1)
        Self::halve_mod_bnm1(k, n);

        // The all-ones representative is the redundant form of zero.
        if k.iter().all(|limb| *limb == Limb::MAX) {
            k.fill(0);
        }

        // The n limb count is the length of a caller-provided xm slice, and the
        // xp width is capped by xp.len() == n + 1 in the assert above. A slice of
        // n limbs holds n * size_of::<Limb>() bytes, so on every supported usize
        // width n <= isize::MAX / size_of::<Limb>(); doubling that is far below
        // usize::MAX, and the recursion never drives n above its validated top.
        let full_width = n.wrapping_mul(2);
        let max_len = dst.len().min(full_width);
        // Every limb below max_len is overwritten by the two assignments below. Only
        // a caller-provided tail beyond the complete CRT width needs clearing; for
        // the dominant equal-width product that range is empty.
        // SAFETY: max_len is min(dst.len(), 2n), so the tail is within dst.
        unsafe { dst.get_unchecked_mut(max_len..) }.fill(0);
        if max_len == 0 {
            return;
        }

        let copy_xp = max_len.min(n);
        // SAFETY: dst, xp, and k each span copy_xp limbs and are disjoint. Fusing
        // the copy and addition removes one complete output-width memory pass.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(dst.as_mut_ptr(), xp.as_ptr(), k.as_ptr(), copy_xp)
        };

        if max_len > n {
            let copy_k = max_len.wrapping_sub(n);
            // SAFETY: max_len <= dst.len() and max_len <= 2n, so dst[n..max_len]
            // is in range and k[..copy_k] holds exactly copy_k of k's n limbs.
            let dst_span = unsafe { dst.get_unchecked_mut(n..max_len) };
            // SAFETY: copy_k == max_len - n <= n == k.len(), so k[..copy_k] is in range.
            let k_span = unsafe { k.get_unchecked(..copy_k) };
            dst_span.copy_from_slice(k_span);
            // SAFETY: max_len <= dst.len(), so dst[n..max_len] is in range.
            let mut c2 =
                SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(n..max_len) }, &[carry]);
            // SAFETY: xp carries one guard limb above the n-limb residue.
            let xp_guard = unsafe { *xp.get_unchecked(n) };
            // SAFETY: max_len <= dst.len(), so dst[n..max_len] is in range.
            c2 = c2.wrapping_add(SsaCarry::add_full_in_place(
                unsafe { dst.get_unchecked_mut(n..max_len) },
                &[xp_guard],
            ));
            let _ = c2;
        }
    }

    /// Scratch required by one [`mul_mod_bnm1`] call on `n`-limb operands.
    pub fn mul_mod_bnm1_scratch_len(n: usize) -> usize {
        if n <= SSA_BNM1_BASECASE_LIMBS {
            // SAFETY: n <= SSA_BNM1_BASECASE_LIMBS is a small compile-time
            // constant, so 2n and the scratch for n*n operands at those widths
            // are each bounded by a constant; the sum is far below usize::MAX
            // on every supported width.
            let prod = n.wrapping_mul(2);
            return prod.wrapping_add(Multiplication::required_scratch(n, n));
        }
        let h = n >> 1;
        let ring_bits = ring_modulus_bits(h);
        Self::layout_len(h, FftPlan::new(ring_bits).required_mul_scratch())
    }

    /// Total buffer the squaring CRT split partitions.
    ///
    /// The same layout as [`Self::layout_len`] with one operand per half instead of
    /// two, because a square stages only `a_low - a_high` for the Fermat residue
    /// and only `a_low + a_high` for the Mersenne one.
    pub fn sqr_layout_len(half_width: usize, ring_scratch: usize) -> usize {
        // Every call site bounds half_width first: the SSA entries reject
        // half_width * LIMB_BITS overflow with checked arithmetic before this
        // runs. On every supported usize width that bound keeps each
        // intermediate below usize::MAX, so the sums wrap only if a future
        // caller skips the validation; debug builds recompute each step.
        let coefficient_width = half_width.wrapping_add(1);
        let residues = coefficient_width.wrapping_add(half_width);
        let fermat_half = coefficient_width.wrapping_add(ring_scratch);
        let mersenne_half = half_width.wrapping_add(sqr_mod_bnm1_scratch_len(half_width));
        let total = residues.wrapping_add(fermat_half.max(mersenne_half));
        debug_assert!(
            half_width.checked_add(1) == Some(coefficient_width)
                && coefficient_width.checked_add(half_width) == Some(residues)
                && coefficient_width.checked_add(ring_scratch) == Some(fermat_half)
                && half_width.checked_add(sqr_mod_bnm1_scratch_len(half_width))
                    == Some(mersenne_half)
                && residues.checked_add(fermat_half.max(mersenne_half)) == Some(total),
            "CRT square scratch layout overflowed"
        );
        total
    }

    /// Total buffer a CRT split partitions, for a given half-width and a given cost
    /// of the `B^h + 1` ring product.
    ///
    /// Both the top-level entry point and [`mul_mod_bnm1`] lay their scratch out
    /// as `[xp: h+1] [xm: h]` followed by a region the two halves reuse in turn.
    /// The dead `xm` residue itself becomes the CRT `k` buffer, so no third residue
    /// span is retained. The shared tail only has to fit the larger of:
    ///
    /// - the `B^h + 1` half needs two padded operands of `h + 1` limbs plus the
    ///   ring's own scratch;
    /// - the `B^h - 1` half needs two folded operands of `h` limbs plus whatever
    ///   its recursion requires.
    ///
    /// The two callers differ only in `ring_scratch`, because the top level forces
    /// the transform where this one lets a narrow ring take the basecase.
    pub fn layout_len(half_width: usize, ring_scratch: usize) -> usize {
        // Every call site bounds half_width first: the SSA entries reject
        // half_width * LIMB_BITS overflow with checked arithmetic before this
        // runs, and the forced-plan paths go through the same gate. On every
        // supported usize width that bound keeps each intermediate below
        // usize::MAX, so the sums wrap only if a future caller skips the
        // validation; debug builds recompute each step to catch that caller.
        let coefficient_width = half_width.wrapping_add(1);
        let residues = coefficient_width.wrapping_add(half_width);
        let fermat_half = coefficient_width.wrapping_mul(2).wrapping_add(ring_scratch);
        let mersenne_half = half_width
            .wrapping_mul(2)
            .wrapping_add(Self::mul_mod_bnm1_scratch_len(half_width));
        let total = residues.wrapping_add(fermat_half.max(mersenne_half));
        debug_assert!(
            half_width.checked_add(1) == Some(coefficient_width)
                && coefficient_width.checked_add(half_width) == Some(residues)
                && coefficient_width
                    .checked_mul(2)
                    .and_then(|w| w.checked_add(ring_scratch))
                    == Some(fermat_half)
                && half_width
                    .checked_mul(2)
                    .and_then(|w| w.checked_add(Self::mul_mod_bnm1_scratch_len(half_width)))
                    == Some(mersenne_half)
                && residues.checked_add(fermat_half.max(mersenne_half)) == Some(total),
            "CRT product scratch layout overflowed"
        );
        total
    }

    /// Halves `k` modulo `B^size - 1` using bitwise right-shift with wraparound.
    ///
    /// Since `2^{-1} ≡ 2^(LIMB_BITS * size - 1) mod (2^(LIMB_BITS * size) - 1)`,
    /// the modular inverse is a logical right shift that rotates the LSB into the
    /// top of the most significant limb.
    pub fn halve_mod_bnm1(k: &mut [Limb], size: usize) {
        assert!(
            size > 0 && size <= k.len(),
            "halve_mod_bnm1 needs 1 <= size <= k.len()"
        );
        // SAFETY: size >= 1 is checked above, so k[0] is in range.
        let lowest_bit = unsafe { k.get_unchecked(0) } & 1;
        let mut carry_down = 0;
        // SAFETY: size <= k.len() is checked above, so the size-limb span is in range.
        for limb in unsafe { k.get_unchecked_mut(..size) }.iter_mut().rev() {
            let next_carry_down = *limb & 1;
            *limb = (*limb >> 1) | (carry_down << LIMB_BITS.wrapping_sub(1));
            carry_down = next_carry_down;
        }
        // SAFETY: size >= 1 is checked above, so k[size - 1] is in range.
        *unsafe { k.get_unchecked_mut(size.wrapping_sub(1)) } |=
            lowest_bit << LIMB_BITS.wrapping_sub(1);
    }
}

/// Bit width of a Fermat ring for a `half_width`-limb CRT half.
///
/// `wrapping_mul` is sound because every call site derives `half_width` from a
/// validated top-level width: the public entry bounds `n * LIMB_BITS` before any
/// CRT recursion starts, and every recursion here halves `n`, so
/// `half_width * LIMB_BITS <= top_width * LIMB_BITS < 2^usize`. Only a direct
/// unvalidated internal call could overflow, and it would fail its caller's
/// validation before ever reaching this module.
fn ring_modulus_bits(half_width: usize) -> usize {
    debug_assert!(
        half_width.checked_mul(LIMB_BITS).is_some(),
        "CRT half-width modulus bits overflowed"
    );
    half_width.wrapping_mul(LIMB_BITS)
}

/// Scratch required by one [`SsaCrt::sqr_mod_bnm1`] call on an `n`-limb operand.
fn sqr_mod_bnm1_scratch_len(n: usize) -> usize {
    if n <= SSA_BNM1_BASECASE_LIMBS {
        // SAFETY: n <= SSA_BNM1_BASECASE_LIMBS is a small compile-time
        // constant, so 2n and the scratch for a basecase squaring at that width
        // are each bounded by a constant; the sum is far below usize::MAX on
        // every supported width.
        let prod = n.wrapping_mul(2);
        return prod.wrapping_add(Multiplication::required_sqr_scratch(n));
    }
    let h = n >> 1;
    let ring_bits = ring_modulus_bits(h);
    SsaCrt::sqr_layout_len(h, FftPlan::new(ring_bits).required_sqr_scratch())
}

/// Reconstructs `dst = xp + k * B^h + k mod (B^n - 1)` from the two residues,
/// where `k = (xm - xp) / 2 mod (B^h - 1)`.
///
/// Shared by the product and square recursions, which differ only in how they
/// obtain `xp` and `xm`. `xm` is dead once `k` is derived, so it is transformed
/// into `k` in place rather than copied into a third residue span.
fn merge_crt_halves(dst: &mut [Limb], xp: &[Limb], xm: &mut [Limb], h: usize, n: usize) {
    assert!(h > 0, "the recursive CRT half-width must be nonzero");
    assert_eq!(
        h.checked_mul(2),
        Some(n),
        "the recursive CRT width must equal twice its half-width"
    );
    assert_eq!(dst.len(), n, "the recursive CRT destination width differs");
    assert_eq!(xm.len(), h, "the Mersenne CRT residue width differs");
    // h == xm.len() == k.len() is the length of a live slice below, so h + 1
    // cannot wrap: the slice itself occupies h * size_of::<Limb>() bytes.
    assert_eq!(
        xp.len(),
        h.wrapping_add(1),
        "the Fermat CRT residue width differs"
    );
    let k = xm;
    // SAFETY: xp holds h + 1 limbs, so the h-limb span is in range.
    let mut borrow = Addition::sub_slice_in_place(k, unsafe { xp.get_unchecked(..h) });
    // SAFETY: xp holds h + 1 limbs, so the guard slot at index h is in range.
    borrow = borrow.wrapping_add(unsafe { *xp.get_unchecked(h) });

    let b2 = SsaCarry::sub_full_in_place(k, &[borrow]);
    if b2 > 0 {
        let _ = SsaCarry::sub_full_in_place(k, &[b2]);
    }

    SsaCrt::halve_mod_bnm1(k, h);

    // SAFETY: the h-limb destination and both inputs are complete and disjoint.
    // This writes xp+k directly instead of copying xp and adding in place.
    let carry1 =
        unsafe { ArchKernels::add_limbs_3_unchecked(dst.as_mut_ptr(), xp.as_ptr(), k.as_ptr(), h) };

    // SAFETY: dst holds n = 2h limbs and k holds h limbs, so dst[h..] spans
    // exactly k's width and the wrap-around fold below stays within dst.
    unsafe { dst.get_unchecked_mut(h..) }.copy_from_slice(k);
    // SAFETY: xp holds h + 1 limbs, so the guard slot at index h is in range.
    let carry_guard = carry1.wrapping_add(unsafe { *xp.get_unchecked(h) });
    // SAFETY: dst holds n = 2h limbs, so dst[h..] spans exactly k's width.
    let mut carry2 =
        SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(h..) }, &[carry_guard]);
    if carry2 > 0 {
        // the remaining carry can wrap around
        // SAFETY: n == 2h <= dst.len(), so the leading span is in range.
        carry2 = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(..n) }, &[carry2]);
        if carry2 > 0 {
            // SAFETY: n == 2h <= dst.len(), so the leading span is in range.
            let _ = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(..n) }, &[carry2]);
        }
    }
}
