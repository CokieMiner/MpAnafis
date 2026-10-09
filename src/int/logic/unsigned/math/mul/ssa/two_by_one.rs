//! Shared-operand SSA products and their scratch planning.

#![expect(
    unsafe_code,
    reason = "Admitted shared-product plans establish ring widths, guard conventions, and complete operation-sized arenas"
)]

use crate::parallel::ParallelExecutor;

use super::{
    FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, Ssa, SsaCrt, SsaOperation, SsaPlan,
    SsaTransform, TransformChoice,
};

impl Ssa {
    /// Scratch required by [`Self::try_mul_two_by_one_with_executor`] for an executor
    /// advertising `parallelism` scheduling lanes.
    ///
    /// Zero exactly when no fused geometry exists for these widths. The
    /// reservation covers a forced transform and a planned narrow-ring basecase.
    /// Above the direct-Fermat crossover it covers both CRT and direct strategies.
    #[must_use]
    pub fn mul_two_by_one_scratch_len_for_parallelism(
        len_a: usize,
        len_b: usize,
        len_x: usize,
        parallelism: usize,
    ) -> usize {
        // Each product is planned as a `Pair` operation against the shared
        // operand. The fused path exists only when the two searches agree,
        // since one forward transform of `x` serves both rings at one width.
        let Some(half_width) =
            SsaPlan::best_crt_half_width_for_operands(len_a, len_x, SsaOperation::Pair)
        else {
            return 0;
        };
        if SsaPlan::best_crt_half_width_for_operands(len_b, len_x, SsaOperation::Pair)
            != Some(half_width)
        {
            return 0;
        }
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        let ring_plan = FftPlan::new_for_pair(ring_bits);
        let transformed = ring_plan.transform_mul_two_by_one_scratch(parallelism);
        let ring_scratch = if ring_bits <= SSA_BASE_MODULUS_BITS {
            transformed.max(ring_plan.required_mul_scratch())
        } else {
            transformed
        };
        if ring_scratch == usize::MAX {
            return 0;
        }
        let Some(coefficient_len) = half_width.checked_add(1) else {
            return 0;
        };
        let Some((result_len_a, result_len_b)) =
            len_a.checked_add(len_x).zip(len_b.checked_add(len_x))
        else {
            return 0;
        };
        let fallback_a = if result_len_a < coefficient_len {
            coefficient_len
        } else {
            0
        };
        let fallback_b = if result_len_b < coefficient_len {
            coefficient_len
        } else {
            0
        };
        let fermat_input_len = if len_a <= half_width
            && len_b <= half_width
            && len_x <= half_width
            && ring_bits > SSA_BASE_MODULUS_BITS
        {
            0
        } else {
            coefficient_len
        };
        let Some(mersenne_input_len) = [len_a, len_b, len_x]
            .into_iter()
            .try_fold(0_usize, |total, len| {
                total.checked_add(if len == half_width { 0 } else { half_width })
            })
        else {
            return 0;
        };
        let Some(crt_scratch) = SsaCrt::two_by_one_layout_len(
            half_width,
            ring_scratch,
            parallelism,
            fermat_input_len,
            mersenne_input_len,
        )
        .checked_add(fallback_a)
        .and_then(|width| width.checked_add(fallback_b)) else {
            return 0;
        };
        if crt_scratch == usize::MAX {
            return 0;
        }
        let Some(direct_threshold) = Self::direct_fermat_threshold(parallelism) else {
            return crt_scratch;
        };
        if half_width < direct_threshold || !half_width.is_power_of_two() {
            return crt_scratch;
        }
        let Some(direct_bits) = ring_bits.checked_mul(2) else {
            return 0;
        };
        let direct_plan = FftPlan::new_for_pair(direct_bits);
        let direct_scratch = direct_plan.transform_mul_two_by_one_scratch(parallelism);
        if direct_scratch == usize::MAX {
            return 0;
        }
        crt_scratch.max(direct_scratch)
    }

    /// Multiply `a_limbs` and `b_limbs` by the shared `x_limbs` operand.
    ///
    /// The two products reuse the forward transform of `x_limbs`.
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::similar_names,
        reason = "Two-by-one SSA coordination across CRT or direct Fermat path"
    )]
    pub fn try_mul_two_by_one_with_executor<E: ParallelExecutor>(
        out_a: &mut [Limb],
        out_b: &mut [Limb],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        x_limbs: &[Limb],
        choice: TransformChoice,
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        if [a_limbs.len(), b_limbs.len(), x_limbs.len()]
            .into_iter()
            .any(|len| len.checked_mul(LIMB_BITS).is_none())
        {
            return false;
        }
        // SAFETY: the capacity-to-bit checks bound every length by
        // usize::MAX/LIMB_BITS; with LIMB_BITS>=16 both pairwise sums fit.
        let (result_len_a, result_len_b) = unsafe {
            (
                a_limbs.len().unchecked_add(x_limbs.len()),
                b_limbs.len().unchecked_add(x_limbs.len()),
            )
        };
        if out_a.len() < result_len_a || out_b.len() < result_len_b {
            return false;
        }
        // SAFETY: all three capacity-to-bit conversions succeeded above.
        let (sig_a, sig_b, sig_x) = unsafe {
            (
                SsaPlan::significant_bits_of_slice(a_limbs),
                SsaPlan::significant_bits_of_slice(b_limbs),
                SsaPlan::significant_bits_of_slice(x_limbs),
            )
        };

        if sig_x == 0 {
            out_a.fill(0);
            out_b.fill(0);
            return true;
        }
        if sig_a == 0 && sig_b == 0 {
            out_a.fill(0);
            out_b.fill(0);
            return true;
        }
        if sig_a == 0 {
            out_a.fill(0);
            return Self::try_mul_with_executor(out_b, b_limbs, x_limbs, choice, scratch, executor);
        }
        if sig_b == 0 {
            out_b.fill(0);
            return Self::try_mul_with_executor(out_a, a_limbs, x_limbs, choice, scratch, executor);
        }

        let parallelism = executor.parallelism().get();
        let Some(n) = SsaPlan::best_crt_half_width_for_operands(
            a_limbs.len(),
            x_limbs.len(),
            SsaOperation::Pair,
        ) else {
            return false;
        };
        if SsaPlan::best_crt_half_width_for_operands(
            b_limbs.len(),
            x_limbs.len(),
            SsaOperation::Pair,
        ) != Some(n)
        {
            return false;
        }
        let Some(ring_bits) = n.checked_mul(LIMB_BITS) else {
            return false;
        };

        let active_a_len = sig_a.div_ceil(LIMB_BITS);
        let active_b_len = sig_b.div_ceil(LIMB_BITS);
        let active_x_len = sig_x.div_ceil(LIMB_BITS);
        // SAFETY: each significant width was measured from its own initialized
        // operand, so these nonzero prefixes are within the original slices.
        let (active_a, active_b, active_x) = unsafe {
            (
                a_limbs.get_unchecked(..active_a_len),
                b_limbs.get_unchecked(..active_b_len),
                x_limbs.get_unchecked(..active_x_len),
            )
        };

        let direct_fermat_threshold = Self::direct_fermat_threshold(parallelism);
        let use_direct_fermat = choice.use_direct_fermat(n, direct_fermat_threshold);

        if use_direct_fermat {
            let Some(direct_bits) = ring_bits.checked_mul(2) else {
                return false;
            };
            let direct_plan = FftPlan::new_for_pair(direct_bits);
            let needed = direct_plan.transform_mul_two_by_one_scratch(parallelism);
            if needed == usize::MAX || scratch.len() < needed {
                return false;
            }
            // SAFETY: the boundary checked both result widths against their
            // destinations. The chosen half-width proves each product fits 2*n
            // data limbs, so the direct transform needs no guard or padding.
            let (direct_dst_a, direct_dst_b) = unsafe {
                (
                    out_a.get_unchecked_mut(..result_len_a),
                    out_b.get_unchecked_mut(..result_len_b),
                )
            };

            // SAFETY: the plan establishes every ring and scratch width.
            unsafe {
                SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                    direct_dst_a,
                    direct_dst_b,
                    active_a,
                    active_b,
                    active_x,
                    direct_bits,
                    Some((sig_a, sig_b, sig_x)),
                    choice.forces_transform(),
                    Some(&direct_plan),
                    executor,
                    scratch,
                );
            }
            // SAFETY: each prefix length is bounded by its destination.
            unsafe { out_a.get_unchecked_mut(result_len_a..) }.fill(0);
            // SAFETY: each prefix length is bounded by its destination.
            unsafe { out_b.get_unchecked_mut(result_len_b..) }.fill(0);
        } else {
            let ring_plan = FftPlan::new_for_pair(ring_bits);
            let ring_work = if ring_bits <= SSA_BASE_MODULUS_BITS && !choice.forces_transform() {
                ring_plan.required_mul_scratch()
            } else {
                ring_plan.transform_mul_two_by_one_scratch(parallelism)
            };
            // SAFETY: the checked ring width bounds n by usize::MAX/LIMB_BITS,
            // so its single guard fits on both supported SSA pointer widths.
            let coeff_len = unsafe { n.unchecked_add(1) };
            let fermat_input_len = if active_a.len() <= n
                && active_b.len() <= n
                && active_x.len() <= n
                && (ring_bits > SSA_BASE_MODULUS_BITS || choice.forces_transform())
            {
                0
            } else {
                coeff_len
            };
            let fallback_a = if result_len_a < coeff_len {
                coeff_len
            } else {
                0
            };
            let fallback_b = if result_len_b < coeff_len {
                coeff_len
            } else {
                0
            };
            let [left_folded_len, right_folded_len, x_folded_len] = [
                (a_limbs.len(), active_a_len),
                (b_limbs.len(), active_b_len),
                (x_limbs.len(), active_x_len),
            ]
            .map(|(declared, active)| if declared >= n && active <= n { 0 } else { n });
            let Some(mersenne_input_len) = [left_folded_len, right_folded_len, x_folded_len]
                .into_iter()
                .try_fold(0_usize, usize::checked_add)
            else {
                return false;
            };
            let Some(needed) = SsaCrt::two_by_one_layout_len(
                n,
                ring_work,
                parallelism,
                fermat_input_len,
                mersenne_input_len,
            )
            .checked_add(fallback_a)
            .and_then(|width| width.checked_add(fallback_b)) else {
                return false;
            };
            if needed == usize::MAX || scratch.len() < needed {
                return false;
            }

            // SAFETY: needed includes each short output's optional Fermat span,
            // two n-limb Mersenne residues, staging and the reusable child arena.
            let (xp_scratch_a, xp_scratch_b, xm_a, xm_b, shared_scratch) = unsafe {
                let (xp_a, after_xp_a) = scratch.split_at_mut_unchecked(fallback_a);
                let (xp_b, after_xp_b) = after_xp_a.split_at_mut_unchecked(fallback_b);
                let (xm_a, after_xm_a) = after_xp_b.split_at_mut_unchecked(n);
                let (xm_b, work) = after_xm_a.split_at_mut_unchecked(n);
                (xp_a, xp_b, xm_a, xm_b, work)
            };

            {
                // SAFETY: each zero fallback proves its output covers coeff_len;
                // otherwise its initialized scratch span has that exact width.
                // The two mutable outputs and all scratch partitions are disjoint.
                let (xp_a, xp_b) = unsafe {
                    (
                        if fallback_a == 0 {
                            out_a.get_unchecked_mut(..coeff_len)
                        } else {
                            &mut *xp_scratch_a
                        },
                        if fallback_b == 0 {
                            out_b.get_unchecked_mut(..coeff_len)
                        } else {
                            &mut *xp_scratch_b
                        },
                    )
                };
                // SAFETY: the CRT layout reserves three complete guarded inputs
                // and ring_work limbs after its output slots.
                let (left_padded, right_padded, x_padded, ring_scratch) = unsafe {
                    let (left, after_left) =
                        shared_scratch.split_at_mut_unchecked(fermat_input_len);
                    let (right, after_right) = after_left.split_at_mut_unchecked(fermat_input_len);
                    let (x, work) = after_right.split_at_mut_unchecked(fermat_input_len);
                    (left, right, x, work)
                };

                if fermat_input_len == 0 {
                    // SAFETY: all operands fit the data width with implicit
                    // zero high limbs and guards; exact widths are supplied.
                    unsafe {
                        SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                            xp_a,
                            xp_b,
                            active_a,
                            active_b,
                            active_x,
                            ring_bits,
                            Some((sig_a, sig_b, sig_x)),
                            choice.forces_transform(),
                            Some(&ring_plan),
                            executor,
                            ring_scratch,
                        );
                    }
                } else {
                    SsaCrt::stage_padded_operand(left_padded, active_a, n);
                    SsaCrt::stage_padded_operand(right_padded, active_b, n);
                    SsaCrt::stage_padded_operand(x_padded, active_x, n);

                    // SAFETY: staged operands have complete guarded widths.
                    unsafe {
                        SsaTransform::fft_mul_two_by_one_mod_slices_with_executor(
                            xp_a,
                            xp_b,
                            left_padded,
                            right_padded,
                            x_padded,
                            ring_bits,
                            None,
                            choice.forces_transform(),
                            Some(&ring_plan),
                            executor,
                            ring_scratch,
                        );
                    }
                }
            }

            {
                // SAFETY: the second CRT phase reuses the staging slots after
                // the first phase ends, and the checked layout covers its work.
                let (left_folded, right_folded, x_folded, xm_scratch) = unsafe {
                    let (left, after_left) = shared_scratch.split_at_mut_unchecked(left_folded_len);
                    let (right, after_right) = after_left.split_at_mut_unchecked(right_folded_len);
                    let (x, work) = after_right.split_at_mut_unchecked(x_folded_len);
                    (left, right, x, work)
                };
                let src_a = if left_folded_len == 0 {
                    // SAFETY: zero staging proves n readable limbs and no
                    // nonzero source limbs above this complete residue.
                    unsafe { a_limbs.get_unchecked(..n) }
                } else {
                    SsaCrt::stage_folded_operand(left_folded, active_a, n);
                    &*left_folded
                };
                let src_b = if right_folded_len == 0 {
                    // SAFETY: the same complete-prefix proof holds for b.
                    unsafe { b_limbs.get_unchecked(..n) }
                } else {
                    SsaCrt::stage_folded_operand(right_folded, active_b, n);
                    &*right_folded
                };
                let src_x = if x_folded_len == 0 {
                    // SAFETY: the shared operand has a complete n-limb prefix
                    // and every omitted high source limb is zero.
                    unsafe { x_limbs.get_unchecked(..n) }
                } else {
                    SsaCrt::stage_folded_operand(x_folded, active_x, n);
                    &*x_folded
                };
                SsaCrt::mul_mod_bnm1_two_by_one(
                    xm_a, xm_b, src_a, src_b, src_x, xm_scratch, executor,
                );
            }

            if fallback_a == 0 {
                // SAFETY: the Fermat writer initialized out_a[..=n], and the
                // admitted output covers this complete guarded prefix.
                unsafe {
                    SsaCrt::merge_exact_product_in_place(out_a, xm_a);
                }
            } else {
                SsaCrt::merge_exact_product(out_a, xp_scratch_a, xm_a);
            }
            if fallback_b == 0 {
                // SAFETY: the Fermat writer initialized out_b[..=n], and the
                // admitted output covers this complete guarded prefix.
                unsafe {
                    SsaCrt::merge_exact_product_in_place(out_b, xm_b);
                }
            } else {
                SsaCrt::merge_exact_product(out_b, xp_scratch_b, xm_b);
            }
        }
        true
    }
}
