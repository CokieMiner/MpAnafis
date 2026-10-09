//! Operand-bound SSA product planning and infallible execution.

#![expect(
    unsafe_code,
    reason = "Immutable operand-bound plans validate significant widths, output initialization and disjoint CRT arenas"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{
    CrtMulPlan, FftPlan, LIMB_BITS, Limb, LimbOutput, MulTransformPlan, SSA_BASE_MODULUS_BITS, Ssa,
    SsaCrt, SsaOperation, SsaPlan, SsaRing, SsaTransform, TransformChoice,
};

/// Immutable operands, executable product geometry and reusable arena width.
#[derive(Debug)]
pub struct SsaMultiplicationPlan<'operands> {
    pub a_limbs: &'operands [Limb],
    pub b_limbs: &'operands [Limb],
    /// Exact destination width required by this prepared product.
    pub result_len: usize,
    pub parallelism: NonZeroUsize,
    pub product: Option<SsaProductPlan>,
    /// Exact reusable scratch width required by this prepared product.
    pub scratch_len: usize,
}

#[derive(Debug)]
enum SsaProductStrategy {
    CrtTwoModuli {
        n: usize,
        ring_bits: usize,
        fermat_input_len: usize,
        ring_plan: MulTransformPlan,
        /// Retained Fermat workspace width for concurrent execution.
        ring_scratch_len: usize,
        crt_plan: CrtMulPlan,
    },
    DirectFermat {
        direct_bits: usize,
        direct_plan: MulTransformPlan,
    },
}

#[derive(Debug)]
pub struct SsaProductPlan {
    sig_a: usize,
    sig_b: usize,
    active_a_len: usize,
    active_b_len: usize,
    strategy: SsaProductStrategy,
    force_transform: bool,
}

impl<'operands> SsaMultiplicationPlan<'operands> {
    /// Builds an exact plan for these immutable operands and one executor width.
    ///
    /// Returns `None` when product dimensions or workspace are unrepresentable.
    #[expect(
        clippy::too_many_lines,
        reason = "One checked construction binds immutable operands, ring geometry, strategy and both CRT arenas"
    )]
    pub fn try_new(
        a_limbs: &'operands [Limb],
        b_limbs: &'operands [Limb],
        choice: TransformChoice,
        parallelism: NonZeroUsize,
    ) -> Option<Self> {
        let result_len = a_limbs.len().checked_add(b_limbs.len())?;
        let _left_capacity_bits = a_limbs.len().checked_mul(LIMB_BITS)?;
        let _right_capacity_bits = b_limbs.len().checked_mul(LIMB_BITS)?;
        // SAFETY: both capacity-to-bit conversions succeeded above.
        let (sig_a, sig_b) = unsafe {
            (
                SsaPlan::significant_bits_of_slice(a_limbs),
                SsaPlan::significant_bits_of_slice(b_limbs),
            )
        };
        if sig_a == 0 || sig_b == 0 {
            return Some(Self {
                a_limbs,
                b_limbs,
                result_len,
                parallelism,
                product: None,
                scratch_len: 0,
            });
        }

        // Cost-selected geometries are not monotone in significant width.
        // Declared widths fix geometry; active widths determine input staging.
        let required_bits = result_len.checked_mul(LIMB_BITS)?;
        let n = SsaPlan::best_crt_half_width(required_bits, SsaOperation::Multiply)?;
        let direct_fermat_threshold = Ssa::direct_fermat_threshold(parallelism.get());
        let (left_active_len, right_active_len) =
            (sig_a.div_ceil(LIMB_BITS), sig_b.div_ceil(LIMB_BITS));
        let (strategy, scratch_len) = if choice.use_direct_fermat(n, direct_fermat_threshold) {
            let direct_bits = n.checked_mul(LIMB_BITS.checked_mul(2)?)?;
            let direct_plan = MulTransformPlan::new(FftPlan::new(direct_bits));
            let direct_transform_scratch = direct_plan.transform_mul_scratch(parallelism.get());
            if direct_transform_scratch == usize::MAX {
                return None;
            }
            (
                SsaProductStrategy::DirectFermat {
                    direct_bits,
                    direct_plan,
                },
                direct_transform_scratch,
            )
        } else {
            let ring_bits = n.checked_mul(LIMB_BITS)?;
            let coeff_len = n.checked_add(1)?;
            let transform = choice.forces_transform() || ring_bits > SSA_BASE_MODULUS_BITS;
            let fermat_input_len = if left_active_len <= n && right_active_len <= n && transform {
                0
            } else {
                coeff_len
            };
            // A complete n-limb prefix represents the Mersenne operand when
            // every higher source limb is zero. Other operands reserve n limbs.
            let mersenne_input_len = [
                (a_limbs.len(), left_active_len),
                (b_limbs.len(), right_active_len),
            ]
            .into_iter()
            .try_fold(0_usize, |total, (declared, active)| {
                total.checked_add(if declared >= n && active <= n { 0 } else { n })
            })?;
            let ring_plan = MulTransformPlan::new(FftPlan::new(ring_bits));
            let crt_plan = CrtMulPlan::new(n)?;
            let ring_scratch_len = if transform {
                ring_plan.transform_mul_scratch(parallelism.get())
            } else {
                ring_plan.required_mul_scratch()
            };
            // Concurrent halves retain both child arenas; sequential halves
            // reuse the larger one. An overflowing child rejects either layout.
            let crt_scratch_len = if parallelism.get() > 1 {
                SsaCrt::layout_len_concurrent(
                    n,
                    ring_scratch_len,
                    parallelism.get(),
                    fermat_input_len,
                    mersenne_input_len,
                )
            } else {
                SsaCrt::layout_len(
                    n,
                    ring_scratch_len,
                    parallelism.get(),
                    fermat_input_len,
                    mersenne_input_len,
                )
            };
            if crt_scratch_len == usize::MAX {
                return None;
            }
            let fermat_scratch_len = if result_len < coeff_len { coeff_len } else { 0 };
            let scratch_len = crt_scratch_len.checked_add(fermat_scratch_len)?;
            (
                SsaProductStrategy::CrtTwoModuli {
                    n,
                    ring_bits,
                    fermat_input_len,
                    ring_plan,
                    ring_scratch_len,
                    crt_plan,
                },
                scratch_len,
            )
        };
        Some(Self {
            a_limbs,
            b_limbs,
            result_len,
            parallelism,
            scratch_len,
            product: Some(SsaProductPlan {
                sig_a,
                sig_b,
                active_a_len: left_active_len,
                active_b_len: right_active_len,
                strategy,
                force_transform: choice.forces_transform(),
            }),
        })
    }
}

impl SsaMultiplicationPlan<'_> {
    /// Runs with a caller-owned workspace sized before timing.
    ///
    /// # Safety
    /// `dst` covers `result_len`, `scratch` covers `scratch_len`, and the
    /// executor advertises the parallelism used to construct this plan.
    #[expect(
        clippy::too_many_lines,
        reason = "Both CRT halves share one operand-bound geometry and sequential or concurrent arena"
    )]
    pub unsafe fn run_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [impl LimbOutput],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        debug_assert_eq!(
            executor.parallelism(),
            self.parallelism,
            "SSA executor width differs from its prepared product plan"
        );
        // SAFETY: the caller proves this exact prefix exists.
        let scratch_buf = unsafe { scratch.get_unchecked_mut(..self.scratch_len) };
        let Some(product) = self.product.as_ref() else {
            dst.fill(LimbOutput::from_limb(0));
            return;
        };
        let &SsaProductPlan {
            sig_a,
            sig_b,
            active_a_len,
            active_b_len,
            ref strategy,
            force_transform,
        } = product;
        // SAFETY: both active lengths were derived from these immutable operands.
        let (active_a, active_b) = unsafe {
            (
                self.a_limbs.get_unchecked(..active_a_len),
                self.b_limbs.get_unchecked(..active_b_len),
            )
        };
        match *strategy {
            SsaProductStrategy::DirectFermat {
                direct_bits,
                ref direct_plan,
            } => {
                let ml = SsaRing::mod_limbs(direct_bits);
                debug_assert!(
                    active_a.len() <= ml && active_b.len() <= ml,
                    "the exact product-width proof bounds each direct Fermat operand"
                );
                // SAFETY: the direct ring covers result_len data limbs and
                // both significant operands. The exact product needs no guard.
                unsafe {
                    SsaTransform::fft_mul_mod_slices_with_executor(
                        dst.get_unchecked_mut(..self.result_len),
                        active_a,
                        active_b,
                        direct_bits,
                        Some((sig_a, sig_b)),
                        force_transform,
                        direct_plan,
                        executor,
                        scratch_buf,
                    );
                    dst.get_unchecked_mut(self.result_len..)
                        .fill(LimbOutput::from_limb(0));
                }
            }
            SsaProductStrategy::CrtTwoModuli {
                n,
                ring_bits,
                fermat_input_len,
                ref ring_plan,
                ring_scratch_len,
                ref crt_plan,
            } => {
                let left_folded_len = if self.a_limbs.len() >= n && active_a_len <= n {
                    0
                } else {
                    n
                };
                let right_folded_len = if self.b_limbs.len() >= n && active_b_len <= n {
                    0
                } else {
                    n
                };
                // SAFETY: construction checked n*LIMB_BITS, with LIMB_BITS>=16.
                let coeff_len = unsafe { n.unchecked_add(1) };
                let fermat_scratch_len = if self.result_len < coeff_len {
                    coeff_len
                } else {
                    0
                };
                // SAFETY: the plan reserves the optional Fermat span and
                // n Mersenne limbs before the sequential or concurrent arena.
                let (fermat_buffer, xm, rest3) = unsafe {
                    let (xp, after_fermat) = scratch_buf.split_at_mut_unchecked(fermat_scratch_len);
                    let (xm, work) = after_fermat.split_at_mut_unchecked(n);
                    (xp, xm, work)
                };
                {
                    // SAFETY: zero fallback proves dst covers coeff_len;
                    // otherwise the planned Fermat buffer is initialized and
                    // exclusive. Its writer stores only valid native limb values.
                    let xp = unsafe {
                        if fermat_scratch_len == 0 {
                            dst.get_unchecked_mut(..coeff_len)
                        } else {
                            LimbOutput::from_initialized_mut(fermat_buffer)
                        }
                    };
                    if executor.parallelism().get() > 1 {
                        // SAFETY: construction checked the Fermat sum, both
                        // staging widths and both live child arenas. Exclusive
                        // contiguous partitions give the closures disjoint storage.
                        let (
                            left_padded,
                            right_padded,
                            ring_scratch,
                            left_folded,
                            right_folded,
                            xm_scratch,
                        ) = unsafe {
                            let fermat_half = fermat_input_len
                                .unchecked_mul(2)
                                .unchecked_add(ring_scratch_len);
                            let (fermat, mersenne) = rest3.split_at_mut_unchecked(fermat_half);
                            let (left_padded, after_left_padded) =
                                fermat.split_at_mut_unchecked(fermat_input_len);
                            let (right_padded, ring_scratch) =
                                after_left_padded.split_at_mut_unchecked(fermat_input_len);
                            let (left_folded, after_left_folded) =
                                mersenne.split_at_mut_unchecked(left_folded_len);
                            let (right_folded, xm_scratch) =
                                after_left_folded.split_at_mut_unchecked(right_folded_len);
                            (
                                left_padded,
                                right_padded,
                                ring_scratch,
                                left_folded,
                                right_folded,
                                xm_scratch,
                            )
                        };
                        let need_fermat_staging = fermat_input_len != 0;
                        if need_fermat_staging && left_folded_len != 0 && right_folded_len != 0 {
                            let ((), ()) = executor.join(
                                || {
                                    SsaCrt::stage_padded_and_folded_operand(
                                        left_padded,
                                        left_folded,
                                        active_a,
                                        n,
                                    );
                                },
                                || {
                                    SsaCrt::stage_padded_and_folded_operand(
                                        right_padded,
                                        right_folded,
                                        active_b,
                                        n,
                                    );
                                },
                            );
                        } else {
                            if need_fermat_staging {
                                SsaCrt::stage_padded_operand(left_padded, active_a, n);
                                SsaCrt::stage_padded_operand(right_padded, active_b, n);
                            }
                            if left_folded_len != 0 {
                                SsaCrt::stage_folded_operand(left_folded, active_a, n);
                            }
                            if right_folded_len != 0 {
                                SsaCrt::stage_folded_operand(right_folded, active_b, n);
                            }
                        }
                        let ((), ()) = executor.join(
                            || {
                                let (src_a, src_b, sigs) = if need_fermat_staging {
                                    (&*left_padded, &*right_padded, None)
                                } else {
                                    (active_a, active_b, Some((sig_a, sig_b)))
                                };
                                // SAFETY: the sources are complete staged residues
                                // or admitted normalized inputs with implicit zero
                                // guards. xp and the ring arena are disjoint.
                                unsafe {
                                    SsaTransform::fft_mul_mod_slices_with_executor(
                                        xp,
                                        src_a,
                                        src_b,
                                        ring_bits,
                                        sigs,
                                        force_transform,
                                        ring_plan,
                                        executor,
                                        ring_scratch,
                                    );
                                }
                            },
                            || {
                                let src_a = if left_folded_len != 0 {
                                    &*left_folded
                                } else {
                                    // SAFETY: zero staging proves n readable limbs
                                    // and no nonzero source limbs above the prefix.
                                    unsafe { self.a_limbs.get_unchecked(..n) }
                                };
                                let src_b = if right_folded_len != 0 {
                                    &*right_folded
                                } else {
                                    // SAFETY: the right source has the same complete
                                    // prefix bound and no nonzero high limbs.
                                    unsafe { self.b_limbs.get_unchecked(..n) }
                                };
                                // SAFETY: both sources contain n initialized limbs;
                                // the retained Mersenne tree has its disjoint arena.
                                unsafe {
                                    SsaCrt::mul_mod_bnm1_prepared(
                                        xm, src_a, src_b, xm_scratch, executor, crt_plan,
                                    );
                                }
                            },
                        );
                    } else {
                        // Sequential halves reuse the larger staging and child arena.
                        {
                            // SAFETY: the layout reserves both Fermat staging widths
                            // and the transform workspace in this reusable tail.
                            let (left_padded, right_padded, ring_scratch) = unsafe {
                                let (left, rest) = rest3.split_at_mut_unchecked(fermat_input_len);
                                let (right, work) = rest.split_at_mut_unchecked(fermat_input_len);
                                (left, right, work)
                            };
                            let (src_a, src_b, sigs) = if fermat_input_len == 0 {
                                (active_a, active_b, Some((sig_a, sig_b)))
                            } else {
                                SsaCrt::stage_padded_operand(left_padded, active_a, n);
                                SsaCrt::stage_padded_operand(right_padded, active_b, n);
                                (&*left_padded, &*right_padded, None)
                            };
                            // SAFETY: both normalized sources fit the ring with
                            // implicit guards, or staging supplies full residues.
                            // The output and retained child workspace are disjoint.
                            unsafe {
                                SsaTransform::fft_mul_mod_slices_with_executor(
                                    xp,
                                    src_a,
                                    src_b,
                                    ring_bits,
                                    sigs,
                                    force_transform,
                                    ring_plan,
                                    executor,
                                    ring_scratch,
                                );
                            }
                        }
                        {
                            // SAFETY: Fermat borrows have ended; the shared tail
                            // reserves each required fold and the Mersenne arena.
                            let (left_folded, right_folded, xm_scratch) = unsafe {
                                let (left, rest) = rest3.split_at_mut_unchecked(left_folded_len);
                                let (right, work) = rest.split_at_mut_unchecked(right_folded_len);
                                (left, right, work)
                            };
                            let src_a = if left_folded_len == 0 {
                                // SAFETY: n readable source limbs form the complete
                                // residue, and every higher source limb is zero.
                                unsafe { self.a_limbs.get_unchecked(..n) }
                            } else {
                                SsaCrt::stage_folded_operand(left_folded, active_a, n);
                                &*left_folded
                            };
                            let src_b = if right_folded_len == 0 {
                                // SAFETY: the right source has the same complete
                                // prefix bound and no nonzero high limbs.
                                unsafe { self.b_limbs.get_unchecked(..n) }
                            } else {
                                SsaCrt::stage_folded_operand(right_folded, active_b, n);
                                &*right_folded
                            };
                            // SAFETY: both sources contain n initialized limbs;
                            // the checked arena covers the retained Mersenne tree.
                            unsafe {
                                SsaCrt::mul_mod_bnm1_prepared(
                                    xm, src_a, src_b, xm_scratch, executor, crt_plan,
                                );
                            }
                        }
                    }
                }
                if fermat_scratch_len == 0 {
                    // SAFETY: the Fermat writer initialized the admitted
                    // n+1-limb prefix, disjoint from the complete xm residue.
                    unsafe {
                        SsaCrt::merge_exact_product_in_place(dst, xm);
                    }
                } else {
                    SsaCrt::merge_exact_product(dst, fermat_buffer, xm);
                }
            }
        }
    }
}
