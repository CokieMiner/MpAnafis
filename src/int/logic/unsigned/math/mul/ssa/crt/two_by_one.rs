//! The shared-operand `B^n - 1` recursion and its scratch layout.
//!
//! Each recursion level folds and transforms the common operand once.
//! The pair therefore requires three forward transforms and two inverses.

#![expect(
    unsafe_code,
    reason = "Checked shared-product layouts bound residue folds and disjoint recursive scratch partitions"
)]

use crate::parallel::ParallelExecutor;

use super::{
    FftPlan, LIMB_BITS, Limb, Multiplication, SSA_BASE_MODULUS_BITS, SSA_BNM1_BASECASE_LIMBS,
    SsaCrt, SsaPointwise, SsaTransform,
};

impl SsaCrt {
    /// Computes `dst_a = a * x mod (B^n - 1)` and `dst_b = b * x mod (B^n - 1)`
    /// sharing every decomposition of the common operand `x`.
    ///
    /// The split mirrors [`Self::mul_mod_bnm1_prepared`] level by level, but each
    /// `B^h + 1` ring product is the fused two-by-one transform and the
    /// `B^h - 1` half recurses with the same shared operand. Where two
    /// independent [`Self::mul_mod_bnm1_prepared`] calls forward-transform two
    /// folded copies of `x` at every level, this one transforms each residue
    /// of `x` exactly once, which is the same discount [`Self::sqr_mod_bnm1_prepared`] recovers
    /// by specialising `b = a`.
    ///
    /// All three operands and both destinations are exactly `n` limbs wide,
    /// and `n` must stay even down to [`SSA_BNM1_BASECASE_LIMBS`].
    #[expect(
        clippy::similar_names,
        clippy::many_single_char_names,
        reason = "Mathematical notation for CRT residues (xp, xm) and standard operand names (a, b, x, n, h)"
    )]
    pub fn mul_mod_bnm1_two_by_one<E: ParallelExecutor>(
        dst_a: &mut [Limb],
        dst_b: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        x: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        let n = a.len();
        debug_assert!(n > 0, "a Mersenne ring contains at least one data limb");
        debug_assert_eq!(n, b.len(), "mul_mod_bnm1_two_by_one widths must match");
        debug_assert_eq!(n, x.len(), "mul_mod_bnm1_two_by_one widths must match");
        debug_assert_eq!(
            n,
            dst_a.len(),
            "mul_mod_bnm1_two_by_one dst_a width must match"
        );
        debug_assert_eq!(
            n,
            dst_b.len(),
            "mul_mod_bnm1_two_by_one dst_b width must match"
        );

        if n <= SSA_BNM1_BASECASE_LIMBS {
            // Each fold consumes the full product before the next one overwrites
            // it. Both products reuse one complete output and tower workspace.
            // SAFETY: n is bounded by the compile-time basecase cutoff, and
            // the caller's arena covers the product and its tower workspace.
            let (prod, mul_scratch) = unsafe { scratch.split_at_mut_unchecked(n.unchecked_mul(2)) };

            Multiplication::mul_limbs_with_slice_scratch(a, x, prod, mul_scratch);
            Self::fold_bnm1_product(dst_a, prod);
            Multiplication::mul_limbs_with_slice_scratch(b, x, prod, mul_scratch);
            Self::fold_bnm1_product(dst_b, prod);
            return;
        }

        debug_assert!(
            n.is_multiple_of(2),
            "recursive mul_mod_bnm1_two_by_one width must be even"
        );
        let h = n >> 1;
        // SAFETY: two_by_one_layout_len checked both h-limb Mersenne residues
        // and their shared child arena; h=n/2 bounds the guarded width h+1.
        let (cl, xm_a, xm_b, rest4) = unsafe {
            let cl = h.unchecked_add(1);
            let (xm_a, rest3) = scratch.split_at_mut_unchecked(h);
            let (xm_b, rest4) = rest3.split_at_mut_unchecked(h);
            (cl, xm_a, xm_b, rest4)
        };

        // 1. Compute xp_a = a * x mod (B^h + 1) and xp_b = b * x mod (B^h + 1)
        //    with one forward transform of x's residue.
        {
            // SAFETY: h>=1 and both destinations have 2h limbs, so their
            // disjoint guarded prefixes each cover cl=h+1 initialized outputs.
            let (xp_a, xp_b) =
                unsafe { (dst_a.get_unchecked_mut(..cl), dst_b.get_unchecked_mut(..cl)) };
            // SAFETY: the checked Fermat half reserves three cl-limb inputs
            // before its executor-sized transform arena; all spans are disjoint.
            let (a_padded, b_padded, x_padded, ring_scratch) = unsafe {
                let (a_padded, rest5) = rest4.split_at_mut_unchecked(cl);
                let (b_padded, rest6) = rest5.split_at_mut_unchecked(cl);
                let (x_padded, ring_scratch) = rest6.split_at_mut_unchecked(cl);
                (a_padded, b_padded, x_padded, ring_scratch)
            };

            Self::stage_padded_difference(a_padded, a);
            Self::stage_padded_difference(b_padded, b);
            Self::stage_padded_difference(x_padded, x);

            // SAFETY: h is below the validated top-level CRT width, whose
            // bit capacity is representable on the target usize.
            let modulus_bits = unsafe { h.unchecked_mul(LIMB_BITS) };
            // SAFETY: padded differences are canonical h+1-limb coefficients.
            // The validated layout supplies two disjoint h+1-limb outputs and
            // the executor-sized shared-product arena for this Fermat ring.
            unsafe {
                SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                    xp_a,
                    xp_b,
                    a_padded,
                    b_padded,
                    x_padded,
                    modulus_bits,
                    None,
                    false,
                    None,
                    executor,
                    ring_scratch,
                );
            }
        }

        // 2. Compute xm_a and xm_b mod (B^h - 1) from the folded residues,
        //    recursing with x still shared between both products.
        {
            // SAFETY: the checked Mersenne half reserves three h-limb inputs
            // and its child arena, reusing the now-dead Fermat staging storage.
            let (a_folded, b_folded, x_folded, xm_scratch) = unsafe {
                let (a_folded, rest5) = rest4.split_at_mut_unchecked(h);
                let (b_folded, rest6) = rest5.split_at_mut_unchecked(h);
                let (x_folded, xm_scratch) = rest6.split_at_mut_unchecked(h);
                (a_folded, b_folded, x_folded, xm_scratch)
            };

            Self::stage_folded_sum(a_folded, a);
            Self::stage_folded_sum(b_folded, b);
            Self::stage_folded_sum(x_folded, x);

            Self::mul_mod_bnm1_two_by_one(
                xm_a, xm_b, a_folded, b_folded, x_folded, xm_scratch, executor,
            );
        }

        // SAFETY: the Fermat phase initialized both destination prefixes, and
        // each complete h-limb Mersenne residue is disjoint from its 2h output.
        unsafe {
            Self::merge_crt_halves_in_place(dst_a, xm_a);
            Self::merge_crt_halves_in_place(dst_b, xm_b);
        }
    }

    /// Scratch required by one [`Self::mul_mod_bnm1_two_by_one`] call on
    /// `n`-limb operands under an executor reporting `parallelism` scheduling
    /// lanes.
    ///
    /// The layout mirrors the execution: two Mersenne residues, three staged
    /// operands per half, and a nested ring scratch sized for the same executor
    /// width at every level.
    pub fn mul_mod_bnm1_two_by_one_scratch_len_for_parallelism(
        n: usize,
        parallelism: usize,
    ) -> usize {
        if n <= SSA_BNM1_BASECASE_LIMBS {
            // SAFETY: n<=SSA_BNM1_BASECASE_LIMBS bounds this product by a
            // small compile-time constant on both SSA pointer widths.
            let products = unsafe { n.unchecked_mul(2) };
            return products.saturating_add(Multiplication::required_scratch(n, n));
        }
        let h = n >> 1;
        let Some(ring_bits) = h.checked_mul(LIMB_BITS) else {
            return usize::MAX;
        };
        let plan = FftPlan::new_for_pair(ring_bits);
        let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
            SsaPointwise::fermat_basecase_scratch_len(ring_bits)
        } else {
            plan.transform_mul_two_by_one_scratch(parallelism)
        };
        let Some(input_width) = h.checked_add(1) else {
            return usize::MAX;
        };
        let Some(folded_width) = h.checked_mul(3) else {
            return usize::MAX;
        };
        Self::two_by_one_layout_len(h, ring_scratch, parallelism, input_width, folded_width)
    }

    /// Scratch layout for a two-by-one shared-operand CRT product.
    ///
    /// The top level and every recursive `B^n - 1` level stage the same three
    /// operands and two Mersenne residues. Guarded Fermat residues occupy each
    /// destination's prefix; a shorter top-level output reserves its own residue.
    /// `mersenne_input_width` sums only operands requiring an `h`-limb fold.
    pub fn two_by_one_layout_len(
        half_width: usize,
        ring_scratch: usize,
        parallelism: usize,
        fermat_input_width: usize,
        mersenne_input_width: usize,
    ) -> usize {
        let Some(residues) = half_width.checked_mul(2) else {
            return usize::MAX;
        };
        let Some(fermat_half) = fermat_input_width
            .checked_mul(3)
            .and_then(|width| width.checked_add(ring_scratch))
        else {
            return usize::MAX;
        };
        let mersenne_scratch =
            Self::mul_mod_bnm1_two_by_one_scratch_len_for_parallelism(half_width, parallelism);
        let Some(mersenne_half) = mersenne_input_width.checked_add(mersenne_scratch) else {
            return usize::MAX;
        };
        residues.saturating_add(fermat_half.max(mersenne_half))
    }
}
