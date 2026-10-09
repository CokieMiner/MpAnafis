//! Scratch layout and active-operand staging for the CRT halves.
//!
//! Layouts include every nested ring's executor budget. Staging maps at most
//! `2n` active limbs to guarded Fermat or folded Mersenne residues.

#![expect(
    unsafe_code,
    reason = "Checked executor-aware layouts and admitted half-widths bound disjoint staging prefixes and modular carry corrections"
)]

use super::{
    ArchKernels, FftPlan, LIMB_BITS, Limb, Multiplication, SSA_BASE_MODULUS_BITS,
    SSA_BNM1_BASECASE_LIMBS, SsaCarry, SsaCrt,
};

impl SsaCrt {
    /// Scratch required by one [`Self::mul_mod_bnm1_prepared`] call on `n`-limb operands
    /// under an executor reporting `parallelism` scheduling lanes.
    ///
    /// Every nested `B^h + 1` ring below this call is executed with the same
    /// executor, so its transform scratch must be sized from that executor's
    /// slot budget rather than from the two-slot structural minimum.
    pub fn mul_mod_bnm1_scratch_len_for_parallelism(n: usize, parallelism: usize) -> usize {
        if n <= SSA_BNM1_BASECASE_LIMBS {
            // SAFETY: n <= SSA_BNM1_BASECASE_LIMBS is a small compile-time
            // constant, so 2n and the scratch for n*n operands at those widths
            // are each bounded by a constant; the sum is far below usize::MAX
            // on every supported width.
            let prod = unsafe { n.unchecked_mul(2) };
            return prod.saturating_add(Multiplication::required_scratch_for_parallelism(
                n,
                n,
                parallelism,
            ));
        }
        let h = n >> 1;
        let Some(ring_bits) = h.checked_mul(LIMB_BITS) else {
            return usize::MAX;
        };
        let plan = FftPlan::new(ring_bits);
        let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
            plan.required_mul_scratch()
        } else {
            plan.transform_mul_scratch(parallelism)
        };
        let Some(input_width) = h.checked_add(1) else {
            return usize::MAX;
        };
        let Some(folded_width) = h.checked_mul(2) else {
            return usize::MAX;
        };
        Self::layout_len(h, ring_scratch, parallelism, input_width, folded_width)
    }

    /// Total buffer the squaring CRT split partitions.
    ///
    /// The same layout as [`Self::layout_len`] with one operand per half instead of
    /// two, because a square stages only `a_low - a_high` for the Fermat residue
    /// and only `a_low + a_high` for the Mersenne one.
    /// `fermat_input_width` is zero when the top-level transform borrows its
    /// operand with implicit high zeros, and `half_width+1` when staging is required.
    /// `mersenne_input_width` is zero for a borrowed complete operand and
    /// `half_width` for a staged operand.
    pub fn sqr_layout_len(
        half_width: usize,
        ring_scratch: usize,
        parallelism: usize,
        fermat_input_width: usize,
        mersenne_input_width: usize,
    ) -> usize {
        let Some(fermat_half) = fermat_input_width.checked_add(ring_scratch) else {
            return usize::MAX;
        };
        let mersenne_scratch = sqr_mod_bnm1_scratch_len_for_parallelism(half_width, parallelism);
        let Some(mersenne_half) = mersenne_input_width.checked_add(mersenne_scratch) else {
            return usize::MAX;
        };
        half_width.saturating_add(fermat_half.max(mersenne_half))
    }

    /// Total buffer a CRT split partitions, for a given half-width, a given cost
    /// of the `B^h + 1` ring product, and one executor width.
    ///
    /// Both the top-level entry and [`Self::mul_mod_bnm1_prepared`] lay their scratch out
    /// as `[xm: h]` followed by a region the two halves reuse in turn. The Fermat
    /// residue occupies the destination's initialized `h+1`-limb prefix. The
    /// quotient is formed directly in the high output half; shorter exact outputs
    /// instead reuse `xm` for `k`. The shared tail fits the larger of:
    ///
    /// - the `B^h + 1` half needs two operands of `fermat_input_width` limbs
    ///   plus the ring's own scratch. This width is `h+1` for staged residues
    ///   and zero when a top-level transform borrows both active operands;
    /// - the `B^h - 1` half needs `mersenne_input_width` staged limbs plus its
    ///   recursive workspace. Complete operands with no nonzero high limbs
    ///   are borrowed directly; each remaining operand reserves `h` limbs.
    ///
    /// The top level separately reserves a Fermat residue when its exact output
    /// is shorter than `h+1`; recursive `2h`-limb destinations always fit it.
    /// The callers also differ in `ring_scratch`, because the top level can force
    /// the transform where recursion lets a narrow ring take the basecase.
    /// `parallelism` sizes every nested ring the way the executor will execute
    /// it, so no level relies on the caller's top ring leaving spare capacity.
    pub fn layout_len(
        half_width: usize,
        ring_scratch: usize,
        parallelism: usize,
        fermat_input_width: usize,
        mersenne_input_width: usize,
    ) -> usize {
        let Some(fermat_half) = fermat_input_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(ring_scratch))
        else {
            return usize::MAX;
        };
        let mersenne_scratch =
            Self::mul_mod_bnm1_scratch_len_for_parallelism(half_width, parallelism);
        let Some(mersenne_half) = mersenne_input_width.checked_add(mersenne_scratch) else {
            return usize::MAX;
        };
        half_width.saturating_add(fermat_half.max(mersenne_half))
    }

    /// Total buffer the top-level CRT split partitions when the two halves run
    /// concurrently, so both halves' staging operands and recursive workspaces
    /// are live at once instead of reused in turn.
    ///
    /// The nested `B^n - 1` recursion keeps its sequential [`Self::layout_len`];
    /// only the top-level split pays this larger simultaneous footprint, which
    /// a parallel executor repays by evaluating the two independent residues
    /// together.
    pub fn layout_len_concurrent(
        half_width: usize,
        ring_scratch: usize,
        parallelism: usize,
        fermat_input_width: usize,
        mersenne_input_width: usize,
    ) -> usize {
        let Some(fermat_half) = fermat_input_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(ring_scratch))
        else {
            return usize::MAX;
        };
        let mersenne_scratch =
            Self::mul_mod_bnm1_scratch_len_for_parallelism(half_width, parallelism);
        let Some(mersenne_half) = mersenne_input_width.checked_add(mersenne_scratch) else {
            return usize::MAX;
        };
        half_width
            .saturating_add(fermat_half)
            .saturating_add(mersenne_half)
    }

    /// Stages an active operand of at most `2n` limbs into a guarded `n + 1`
    /// limb negacyclic residue for the `B^n + 1` half: the low `n` limbs are
    /// copied, the tail is subtracted, and the guard absorbs any borrow.
    #[inline]
    pub fn stage_padded_operand(padded: &mut [Limb], active: &[Limb], n: usize) {
        let copy = active.len().min(n);
        // SAFETY: copy <= n < padded.len() == n + 1.
        let prefix = unsafe { padded.get_unchecked_mut(..copy) };
        // SAFETY: copy <= active.len().
        let a_prefix = unsafe { active.get_unchecked(..copy) };
        prefix.copy_from_slice(a_prefix);
        // SAFETY: copy <= n, n < padded.len() == n + 1.
        unsafe { padded.get_unchecked_mut(copy..=n) }.fill(0);

        if active.len() > n {
            // SAFETY: active.len() > n and padded has n + 1 limbs.
            let left_data = unsafe { padded.get_unchecked_mut(..n) };
            // SAFETY: active.len() > n.
            let a_tail = unsafe { active.get_unchecked(n..) };
            let borrow = SsaCarry::sub_full_in_place(left_data, a_tail);
            if borrow > 0 {
                // SAFETY: padded has at least n + 1 limbs.
                let sub_data = unsafe { padded.get_unchecked_mut(..n) };
                let carry = SsaCarry::propagate_carry(sub_data);
                // SAFETY: padded has at least n + 1 limbs.
                *unsafe { padded.get_unchecked_mut(n) } = Limb::from(carry);
            }
        }
    }

    /// Stages an active operand of at most `2n` limbs into an `n`-limb folded
    /// operand for the `B^n - 1` half: the low `n` limbs are copied and the
    /// tail is added with the carry folded back around.
    #[inline]
    pub fn stage_folded_operand(folded: &mut [Limb], active: &[Limb], n: usize) {
        let copy = active.len().min(n);
        // SAFETY: copy <= n == folded.len().
        let prefix = unsafe { folded.get_unchecked_mut(..copy) };
        // SAFETY: copy <= active.len().
        let a_prefix = unsafe { active.get_unchecked(..copy) };
        prefix.copy_from_slice(a_prefix);
        // SAFETY: copy <= n == folded.len().
        unsafe { folded.get_unchecked_mut(copy..n) }.fill(0);

        if active.len() > n {
            // SAFETY: active.len() > n and folded has n limbs.
            let a_tail = unsafe { active.get_unchecked(n..) };
            let carry = SsaCarry::add_full_in_place(folded, a_tail);
            if carry > 0 {
                // Both n-limb addends are <=B^n-1; a carried sum has
                // low<=B^n-2, which absorbs the end-around +1.
                let escaped = SsaCarry::propagate_carry(folded);
                debug_assert!(!escaped, "folded operands absorb their single carry");
            }
        }
    }

    /// Stages an active operand of at most `2n` limbs simultaneously into both
    /// the guarded `n + 1` limb residue for `B^n + 1` and the `n`-limb folded
    /// residue for `B^n - 1`, reading `active` in a single memory pass.
    pub fn stage_padded_and_folded_operand(
        padded: &mut [Limb],
        folded: &mut [Limb],
        active: &[Limb],
        n: usize,
    ) {
        let copy = active.len().min(n);
        // SAFETY: copy <= active.len().
        let a_prefix = unsafe { active.get_unchecked(..copy) };
        // SAFETY: copy <= n < padded.len() == n + 1, and copy <= n == folded.len().
        unsafe {
            padded.get_unchecked_mut(..copy).copy_from_slice(a_prefix);
            padded.get_unchecked_mut(copy..=n).fill(0);
            folded.get_unchecked_mut(..copy).copy_from_slice(a_prefix);
            folded.get_unchecked_mut(copy..n).fill(0);
        }

        if active.len() > n {
            // SAFETY: active.len() > n.
            let a_tail = unsafe { active.get_unchecked(n..) };
            let tail_len = a_tail.len();

            let kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();
            // SAFETY: folded and padded each have at least n >= tail_len limbs,
            // do not overlap, and a_tail has tail_len limbs.
            let (mut carry, mut borrow) = unsafe {
                kernel(
                    folded.as_mut_ptr(),
                    padded.as_mut_ptr(),
                    a_tail.as_ptr(),
                    tail_len,
                )
            };
            if tail_len < n {
                // SAFETY: tail_len < n, so tail_len..n is within folded's n limbs.
                let fold_tail = unsafe { folded.get_unchecked_mut(tail_len..n) };
                if carry > 0 {
                    carry = SsaCarry::add_full_in_place(fold_tail, &[carry]);
                }
                // SAFETY: tail_len < n, so tail_len..n is within padded's n + 1 limbs.
                let pad_tail = unsafe { padded.get_unchecked_mut(tail_len..n) };
                if borrow > 0 {
                    borrow = SsaCarry::sub_full_in_place(pad_tail, &[borrow]);
                }
            }
            if borrow > 0 {
                // SAFETY: padded has n + 1 limbs, so ..n is in bounds.
                let sub_data = unsafe { padded.get_unchecked_mut(..n) };
                let escaped = SsaCarry::propagate_carry(sub_data);
                // SAFETY: n is the guard slot index of the n + 1 limb buffer.
                *unsafe { padded.get_unchecked_mut(n) } = Limb::from(escaped);
            }
            if carry > 0 {
                // The fused kernel and tail propagation add two values
                // below B^n; their carried low sum is <=B^n-2.
                let escaped = SsaCarry::propagate_carry(folded);
                debug_assert!(!escaped, "paired folded operands absorb their carry");
            }
        }
    }

    /// Stages `a_low-a_high` modulo `B^h+1` from exactly `2h` source limbs.
    /// A fused subtraction writes all data limbs once; a negative difference
    /// receives the modulus's +1 and carries into the guard only for -1.
    pub fn stage_padded_difference(padded: &mut [Limb], a: &[Limb]) {
        let h = a.len() >> 1;
        // SAFETY: h=a.len()/2 is below half a representable slice length.
        let coefficient_len = unsafe { h.unchecked_add(1) };
        debug_assert_eq!(
            padded.len(),
            coefficient_len,
            "the padded difference requires one guard limb"
        );
        debug_assert!(h > 0, "bnm1 staging requires a nonempty half-width");
        // SAFETY: a holds two h-limb halves; padded holds h initialized data
        // limbs and a guard. All three data spans are aligned and disjoint.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                padded.as_mut_ptr(),
                a.as_ptr(),
                a.as_ptr().add(h),
                h,
            )
        };
        // SAFETY: the fixed-width contract gives the complete data and guard.
        // If subtraction borrowed, its wrapped value plus one is the residue;
        // only an escaped +1 needs the canonical guard B^h.
        unsafe {
            let guard = borrow != 0 && SsaCarry::propagate_carry(padded.get_unchecked_mut(..h));
            *padded.get_unchecked_mut(h) = Limb::from(guard);
        }
    }

    /// Stages `a_low+a_high` modulo `B^h-1` from exactly `2h` source limbs.
    pub fn stage_folded_sum(folded: &mut [Limb], a: &[Limb]) {
        let h = a.len() >> 1;
        debug_assert_eq!(folded.len(), h, "the folded operand width is exact");
        debug_assert!(h > 0, "bnm1 staging requires a nonempty half-width");
        // SAFETY: a contains two initialized h-limb halves; folded is a
        // complete aligned, exclusive, disjoint h-limb destination.
        let carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                folded.as_mut_ptr(),
                a.as_ptr(),
                a.as_ptr().add(h),
                h,
            )
        };
        if carry != 0 {
            // Both inputs are <=B^h-1. A carried sum has low<=B^h-2;
            // its end-around +1 is exact and cannot carry again.
            let escaped = SsaCarry::propagate_carry(folded);
            debug_assert!(!escaped, "two h-limb addends absorb the folded carry");
        }
    }

    /// Halves `k` modulo `B^size - 1` using bitwise right-shift with wraparound.
    ///
    /// Since `2^{-1} == 2^(LIMB_BITS * size - 1) mod (2^(LIMB_BITS * size) - 1)`,
    /// the modular inverse is a logical right shift that rotates the LSB into the
    /// top of the most significant limb.
    pub fn halve_mod_bnm1(k: &mut [Limb], size: usize) {
        debug_assert!(
            size > 0 && size <= k.len(),
            "halve_mod_bnm1 needs 1 <= size <= k.len()"
        );
        // SAFETY: `1 <= size <= k.len()`, k is valid for reads and writes of
        // `size` limbs, and `shift = 1` satisfies `0 < 1 < Limb::BITS`.
        // `rshift_unchecked` shifts the data right in-place and returns the bit
        // shifted out of limb zero positioned at `Limb::BITS - 1`.
        let top_bit = unsafe { ArchKernels::rshift_unchecked(k.as_mut_ptr(), size, 1) };
        // SAFETY: the planned CRT half has size >= 1, so size - 1 is in range.
        unsafe {
            *k.get_unchecked_mut(size.unchecked_sub(1)) |= top_bit;
        }
    }
}

/// Sizes a prepared Mersenne square for its executor budget.
fn sqr_mod_bnm1_scratch_len_for_parallelism(n: usize, parallelism: usize) -> usize {
    if n <= SSA_BNM1_BASECASE_LIMBS {
        // SAFETY: n <= SSA_BNM1_BASECASE_LIMBS bounds the product and its
        // basecase workspace by small constants on both SSA pointer widths.
        let prod = unsafe { n.unchecked_mul(2) };
        return prod.saturating_add(Multiplication::required_sqr_scratch_for_parallelism(
            n,
            parallelism,
        ));
    }
    let h = n >> 1;
    let Some(ring_bits) = h.checked_mul(LIMB_BITS) else {
        return usize::MAX;
    };
    let plan = FftPlan::new_for_square(ring_bits);
    let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
        plan.required_sqr_scratch()
    } else {
        plan.transform_sqr_scratch(parallelism)
    };
    let Some(input_width) = h.checked_add(1) else {
        return usize::MAX;
    };
    SsaCrt::sqr_layout_len(h, ring_scratch, parallelism, input_width, h)
}
