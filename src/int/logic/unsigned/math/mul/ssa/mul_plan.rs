//! Operand-bound SSA multiplication planning and infallible execution.

#![expect(
    unsafe_code,
    reason = "Immutable operand-bound plans retain validated significant widths and disjoint sequential or concurrent CRT layouts"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{
    CrtMulPlan, FftPlan, LIMB_BITS, Limb, LimbOutput, MulTransformPlan, SSA_BASE_MODULUS_BITS, Ssa,
    SsaCrt, SsaOperation, SsaPlan, SsaRing, SsaTransform, TransformChoice,
};

/// Operand-bound SSA product plan with all fallible geometry work completed.
///
/// Borrowed operands preserve the significant widths and geometry across runs.
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
        coeff_len: usize,
        ring_plan: MulTransformPlan,
        /// Scratch the `B^n + 1` transform reserves, retained so concurrent CRT
        /// execution can size its Fermat half without replanning.
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
    /// Returns `None` only when the product dimensions cannot be represented or
    /// a pinned transform exponent is invalid for the resulting Fermat ring.
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
        // Match caller scratch sizing; significant widths only prune execution.
        let required_bits = result_len.checked_mul(LIMB_BITS)?;
        let n = SsaPlan::best_crt_half_width(required_bits, SsaOperation::Multiply)?;

        let direct_fermat_threshold = Ssa::direct_fermat_threshold(parallelism.get());
        let use_direct_fermat = choice.use_direct_fermat(n, direct_fermat_threshold);
        let (strategy, scratch_len) = if use_direct_fermat {
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
            let ring_plan = MulTransformPlan::new(FftPlan::new(ring_bits));
            let crt_plan = CrtMulPlan::new(n)?;
            let ring_scratch_len = if choice.forces_transform() || ring_bits > SSA_BASE_MODULUS_BITS
            {
                ring_plan.transform_mul_scratch(parallelism.get())
            } else {
                ring_plan.required_mul_scratch()
            };
            if ring_scratch_len == usize::MAX {
                return None;
            }

            // A parallel executor evaluates the two CRT halves concurrently,
            // which keeps both halves' staging and workspaces live at once.
            let scratch_len = if parallelism.get() > 1 {
                SsaCrt::layout_len_concurrent(n, ring_scratch_len, parallelism.get())
            } else {
                SsaCrt::layout_len(n, ring_scratch_len, parallelism.get())
            };
            if scratch_len == usize::MAX {
                return None;
            }
            (
                SsaProductStrategy::CrtTwoModuli {
                    n,
                    ring_bits,
                    coeff_len,
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
                active_a_len: sig_a.div_ceil(LIMB_BITS),
                active_b_len: sig_b.div_ceil(LIMB_BITS),
                strategy,
                force_transform: choice.forces_transform(),
            }),
        })
    }

    /// Runs with a caller-owned workspace sized before timing.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::result_len`] limbs, `scratch`
    /// must contain at least [`Self::scratch_len`] limbs, and the executor must
    /// advertise the parallelism used to construct this plan.
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
        // SAFETY: the plan derived both active lengths from these borrowed
        // operands, so each prefix lies within its immutable source slice.
        let active_a = unsafe { self.a_limbs.get_unchecked(..active_a_len) };
        // SAFETY: the identical operand-bound invariant holds for the right side.
        let active_b = unsafe { self.b_limbs.get_unchecked(..active_b_len) };

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
                let output_len = self.result_len;
                // SAFETY: every CRT candidate has 2n>=result_len; the direct
                // ring has exactly 2n data limbs. The caller covers result_len.
                let direct_dst = unsafe { dst.get_unchecked_mut(..output_len) };
                // SAFETY: both normalized active operands fit the direct ring;
                // their omitted high limbs and guard are zero. Their significant
                // widths prove the exact product fits this guard-free destination.
                unsafe {
                    SsaTransform::fft_mul_mod_slices_with_executor(
                        direct_dst,
                        active_a,
                        active_b,
                        direct_bits,
                        Some((sig_a, sig_b)),
                        force_transform,
                        direct_plan,
                        executor,
                        scratch_buf,
                    );
                }
                // SAFETY: output_len <= dst.len(); exact-product high limbs
                // beyond the direct ring are implicit zeros. Filling an empty
                // suffix is a no-op on the common exact-width path.
                unsafe { dst.get_unchecked_mut(output_len..) }.fill(LimbOutput::from_limb(0));
            }
            SsaProductStrategy::CrtTwoModuli {
                n,
                ring_bits,
                coeff_len,
                ref ring_plan,
                ring_scratch_len,
                ref crt_plan,
            } => {
                // SAFETY: the immutable CRT layout reserves coeff_len Fermat
                // limbs and n Mersenne limbs before both child arenas.
                let (xp, xm, rest3) = unsafe {
                    let (xp, after_fermat) = scratch_buf.split_at_mut_unchecked(coeff_len);
                    let (xm, work) = after_fermat.split_at_mut_unchecked(n);
                    (xp, xm, work)
                };

                if executor.parallelism().get() > 1 {
                    // The two CRT halves are independent, so a parallel
                    // executor evaluates them concurrently. Each half owns its
                    // staging operands and workspace simultaneously, which the
                    // concurrent layout sized at plan time.
                    // SAFETY: layout_len_concurrent checked this exact Fermat
                    // sum and both operand pairs before accepting the arena.
                    // Every partition is initialized and exclusively owned;
                    // the two executor closures receive disjoint child regions.
                    let (
                        left_padded,
                        right_padded,
                        ring_scratch,
                        left_folded,
                        right_folded,
                        xm_scratch,
                    ) = unsafe {
                        let fermat_half =
                            coeff_len.unchecked_mul(2).unchecked_add(ring_scratch_len);
                        let (fermat, mersenne) = rest3.split_at_mut_unchecked(fermat_half);
                        let (left_padded, after_left_padded) =
                            fermat.split_at_mut_unchecked(coeff_len);
                        let (right_padded, ring_scratch) =
                            after_left_padded.split_at_mut_unchecked(coeff_len);
                        let (left_folded, after_left_folded) = mersenne.split_at_mut_unchecked(n);
                        let (right_folded, xm_scratch) =
                            after_left_folded.split_at_mut_unchecked(n);
                        (
                            left_padded,
                            right_padded,
                            ring_scratch,
                            left_folded,
                            right_folded,
                            xm_scratch,
                        )
                    };

                    let need_fermat_staging = active_a.len() > n
                        || active_b.len() > n
                        || (ring_bits <= SSA_BASE_MODULUS_BITS && !force_transform);
                    let need_mersenne_staging = active_a.len() != n || active_b.len() != n;

                    if need_fermat_staging && need_mersenne_staging {
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
                        if need_mersenne_staging {
                            SsaCrt::stage_folded_operand(left_folded, active_a, n);
                            SsaCrt::stage_folded_operand(right_folded, active_b, n);
                        }
                    }

                    let ((), ()) = executor.join(
                        || {
                            // 1. xp = a * b mod (B^n + 1).
                            let (src_a, src_b, sigs) = if need_fermat_staging {
                                (&*left_padded, &*right_padded, None)
                            } else {
                                (active_a, active_b, Some((sig_a, sig_b)))
                            };
                            // SAFETY: staged buffers have complete guarded widths, or normalized
                            // inputs fit within ml=n limbs with valid significant widths.
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
                            // 2. xm = a * b mod (B^n - 1).
                            let (src_a, src_b) = if need_mersenne_staging {
                                (&*left_folded, &*right_folded)
                            } else {
                                (active_a, active_b)
                            };
                            // SAFETY: staging establishes complete n-limb residues,
                            // and the retained child arena is disjoint.
                            unsafe {
                                SsaCrt::mul_mod_bnm1_prepared(
                                    xm, src_a, src_b, xm_scratch, executor, crt_plan,
                                );
                            }
                        },
                    );
                } else {
                    // Sequential: the two halves reuse one staging tail.
                    // 1. Compute xp = a * b mod (B^n + 1)
                    {
                        // SAFETY: the sequential layout reserves two guarded
                        // operands and the Fermat workspace in this shared tail.
                        let (left_padded, right_padded, ring_scratch) = unsafe {
                            let (left, rest) = rest3.split_at_mut_unchecked(coeff_len);
                            let (right, work) = rest.split_at_mut_unchecked(coeff_len);
                            (left, right, work)
                        };

                        if active_a.len() <= n
                            && active_b.len() <= n
                            && (ring_bits > SSA_BASE_MODULUS_BITS || force_transform)
                        {
                            // SAFETY: normalized operands are nonzero, fit within
                            // ml=n data limbs, and the transform treats their omitted
                            // guards as zero. The caller-owned matrices remain fully guarded.
                            unsafe {
                                SsaTransform::fft_mul_mod_slices_with_executor(
                                    xp,
                                    active_a,
                                    active_b,
                                    ring_bits,
                                    Some((sig_a, sig_b)),
                                    force_transform,
                                    ring_plan,
                                    executor,
                                    ring_scratch,
                                );
                            }
                        } else {
                            SsaCrt::stage_padded_operand(left_padded, active_a, n);
                            SsaCrt::stage_padded_operand(right_padded, active_b, n);

                            // SAFETY: staged buffers have their complete guarded widths.
                            unsafe {
                                SsaTransform::fft_mul_mod_slices_with_executor(
                                    xp,
                                    left_padded,
                                    right_padded,
                                    ring_bits,
                                    None,
                                    force_transform,
                                    ring_plan,
                                    executor,
                                    ring_scratch,
                                );
                            }
                        }
                    }

                    // 2. Compute xm = a * b mod (B^n - 1)
                    {
                        // SAFETY: the Fermat borrows have ended. The shared tail
                        // reserves both n-limb folds and the Mersenne child arena.
                        let (left_folded, right_folded, xm_scratch) = unsafe {
                            let (left, rest) = rest3.split_at_mut_unchecked(n);
                            let (right, work) = rest.split_at_mut_unchecked(n);
                            (left, right, work)
                        };

                        if active_a.len() == n && active_b.len() == n {
                            // The normalized equal-width case already is a pair of complete
                            // B^n-1 operands. `mul_mod_bnm1` only reads them, so routing the
                            // caller slices directly avoids two full-width staging copies.
                            // SAFETY: complete active operands and scratch match
                            // the retained Mersenne recursion.
                            unsafe {
                                SsaCrt::mul_mod_bnm1_prepared(
                                    xm, active_a, active_b, xm_scratch, executor, crt_plan,
                                );
                            }
                        } else {
                            SsaCrt::stage_folded_operand(left_folded, active_a, n);
                            SsaCrt::stage_folded_operand(right_folded, active_b, n);

                            // SAFETY: staging and arena partitioning establish
                            // every operand and scratch bound in the retained tree.
                            unsafe {
                                SsaCrt::mul_mod_bnm1_prepared(
                                    xm,
                                    left_folded,
                                    right_folded,
                                    xm_scratch,
                                    executor,
                                    crt_plan,
                                );
                            }
                        }
                    }
                }

                // 3. Reconstruct dst = X_p + k * B^n + k.
                SsaCrt::merge_exact_product(dst, xp, xm);
            }
        }
    }
}
