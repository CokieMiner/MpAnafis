//! Operand-bound SSA square planning for repeated infallible execution.

#![expect(
    unsafe_code,
    reason = "Immutable operand-bound plans validate square widths, coefficient staging, and reusable CRT workspace partitions"
)]

use crate::parallel::ParallelExecutor;

use super::{
    CrtSquarePlan, FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SquareTransformPlan, SsaCrt,
    SsaOperation, SsaPlan, SsaTransform, TransformChoice,
};

/// Operand-bound SSA square plan with all fallible geometry work completed.
///
/// The borrowed operand preserves its significant width and geometry across runs.
#[derive(Debug)]
pub struct SsaSquaringPlan<'operand> {
    pub a_limbs: &'operand [Limb],
    /// Exact destination width required by this prepared square.
    pub result_len: usize,
    pub parallelism: usize,
    pub square: Option<SsaSquareGeometry>,
    /// Exact reusable scratch width required by this prepared square.
    pub scratch_len: usize,
}

#[derive(Debug)]
pub enum SsaSquareGeometry {
    Crt {
        active_a_len: usize,
        n: usize,
        ring_bits: usize,
        fermat_input_len: usize,
        ring_plan: SquareTransformPlan,
        force_transform: bool,
        crt_plan: CrtSquarePlan,
    },
    DirectFermat {
        active_a_len: usize,
        ring_bits: usize,
        ring_plan: SquareTransformPlan,
    },
}

impl<'operand> SsaSquaringPlan<'operand> {
    /// Builds an exact plan for an immutable operand and one executor width.
    ///
    /// Returns `None` only when the square dimensions cannot be represented or
    /// a transform workspace size overflows.
    pub fn try_new(
        a_limbs: &'operand [Limb],
        choice: TransformChoice,
        executor_parallelism: usize,
    ) -> Option<Self> {
        let result_len = a_limbs.len().checked_mul(2)?;
        let _capacity_bits = a_limbs.len().checked_mul(LIMB_BITS)?;
        let parallelism = executor_parallelism.max(1);
        // SAFETY: the operand capacity-to-bit conversion succeeded above.
        let sig_a = unsafe { SsaPlan::significant_bits_of_slice(a_limbs) };
        if sig_a == 0 {
            return Some(Self {
                a_limbs,
                result_len,
                parallelism,
                square: None,
                scratch_len: 0,
            });
        }

        // Match capacity-only scratch queries. Short significant prefixes
        // prune execution without selecting a different arena.
        let required_bits = result_len.checked_mul(LIMB_BITS)?;
        let n = SsaPlan::best_crt_half_width(required_bits, SsaOperation::Square)?;
        if choice.use_direct_square(n) {
            let ring_bits = n.checked_mul(LIMB_BITS)?.checked_mul(2)?;
            let ring_plan = SquareTransformPlan::new(FftPlan::new_for_square(ring_bits));
            let scratch_len = ring_plan.transform_sqr_scratch(parallelism);
            if scratch_len == usize::MAX {
                return None;
            }
            return Some(Self {
                a_limbs,
                result_len,
                parallelism,
                scratch_len,
                square: Some(SsaSquareGeometry::DirectFermat {
                    active_a_len: sig_a.div_ceil(LIMB_BITS),
                    ring_bits,
                    ring_plan,
                }),
            });
        }
        let ring_bits = n.checked_mul(LIMB_BITS)?;
        let coeff_len = n.checked_add(1)?;
        let fermat_input_len = if choice.forces_transform() || ring_bits > SSA_BASE_MODULUS_BITS {
            0
        } else {
            coeff_len
        };
        let ring_plan = SquareTransformPlan::new(FftPlan::new_for_square(ring_bits));
        let crt_plan = CrtSquarePlan::new(n)?;
        let ring_scratch_len = if choice.forces_transform() || ring_bits > SSA_BASE_MODULUS_BITS {
            ring_plan.transform_sqr_scratch(parallelism)
        } else {
            ring_plan.required_sqr_scratch()
        };
        if ring_scratch_len == usize::MAX {
            return None;
        }
        let crt_scratch_len = SsaCrt::sqr_layout_len(
            n,
            ring_scratch_len,
            parallelism,
            fermat_input_len,
            if a_limbs.len() == n { 0 } else { n },
        );
        if crt_scratch_len == usize::MAX {
            return None;
        }
        let fermat_scratch_len = if result_len < coeff_len { coeff_len } else { 0 };
        let scratch_len = crt_scratch_len.checked_add(fermat_scratch_len)?;

        Some(Self {
            a_limbs,
            result_len,
            parallelism,
            scratch_len,
            square: Some(SsaSquareGeometry::Crt {
                active_a_len: sig_a.div_ceil(LIMB_BITS),
                n,
                ring_bits,
                fermat_input_len,
                ring_plan,
                force_transform: choice.forces_transform(),
                crt_plan,
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
        reason = "Direct and CRT square execution share the operand-bound geometry and reusable arena"
    )]
    pub unsafe fn run_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        debug_assert_eq!(
            executor.parallelism().get(),
            self.parallelism,
            "SSA executor width differs from its prepared square plan"
        );
        // SAFETY: the caller proves this exact prefix exists.
        let scratch_buf = unsafe { scratch.get_unchecked_mut(..self.scratch_len) };
        let Some(square) = self.square.as_ref() else {
            dst.fill(0);
            return;
        };
        let (active_a_len, n, ring_bits, fermat_input_len, ring_plan, force_transform, crt_plan) =
            match *square {
                SsaSquareGeometry::DirectFermat {
                    active_a_len,
                    ring_bits,
                    ref ring_plan,
                } => {
                    // Each CRT candidate satisfies 2n>=result_len, and the
                    // direct Fermat ring contains exactly 2n data limbs.
                    // SAFETY: the borrowed operand establishes active_a_len, and
                    // dst covers result_len. Twice the significant width is <=
                    // ring_bits, so the exact square fits a guard-free output.
                    unsafe {
                        SsaTransform::fft_sqr_mod_slices_with_executor(
                            dst.get_unchecked_mut(..self.result_len),
                            self.a_limbs.get_unchecked(..active_a_len),
                            ring_bits,
                            true,
                            ring_plan,
                            executor,
                            scratch_buf,
                        );
                        dst.get_unchecked_mut(self.result_len..).fill(0);
                    }
                    return;
                }
                SsaSquareGeometry::Crt {
                    active_a_len,
                    n,
                    ring_bits,
                    fermat_input_len,
                    ref ring_plan,
                    force_transform,
                    ref crt_plan,
                } => (
                    active_a_len,
                    n,
                    ring_bits,
                    fermat_input_len,
                    ring_plan,
                    force_transform,
                    crt_plan,
                ),
            };
        // SAFETY: the significant-bit count was derived from this immutable
        // operand, so its rounded-up active prefix is within the source slice.
        let active_a = unsafe { self.a_limbs.get_unchecked(..active_a_len) };
        let folded_len = if self.a_limbs.len() == n { 0 } else { n };

        // SAFETY: the plan validated n*LIMB_BITS; LIMB_BITS>=32 bounds n+1.
        let coeff_len = unsafe { n.unchecked_add(1) };
        let fermat_scratch_len = if self.result_len < coeff_len {
            coeff_len
        } else {
            0
        };
        // SAFETY: the plan reserves this optional short-output Fermat residue,
        // then n Mersenne limbs and the larger reusable child arena.
        let (fermat_buffer, xm, rest3) = unsafe {
            let (xp, after_fermat) = scratch_buf.split_at_mut_unchecked(fermat_scratch_len);
            let (xm, work) = after_fermat.split_at_mut_unchecked(n);
            (xp, xm, work)
        };

        // 1. Compute xp = a^2 mod (B^n + 1).
        {
            let xp = if fermat_scratch_len == 0 {
                // SAFETY: a zero fallback width proves dst covers coeff_len.
                unsafe { dst.get_unchecked_mut(..coeff_len) }
            } else {
                &mut *fermat_buffer
            };
            // SAFETY: the reusable tail reserves this complete Fermat operand
            // and the ring's workspace, disjoint from the live residues.
            let (padded, ring_scratch) = unsafe { rest3.split_at_mut_unchecked(fermat_input_len) };

            // Construction requests 2*a_limbs.len()*LIMB_BITS product bits.
            // Every CRT candidate has n>=a_limbs.len()>=active_a_len, so the
            // outer square never folds a high operand half.
            if fermat_input_len == 0 {
                // SAFETY: the normalized operand is nonzero, fits within
                // ml=n data limbs, and the transform treats its omitted guard as
                // zero.
                unsafe {
                    SsaTransform::fft_sqr_mod_slices_with_executor(
                        xp,
                        active_a,
                        ring_bits,
                        force_transform,
                        ring_plan,
                        executor,
                        ring_scratch,
                    );
                }
            } else {
                // SAFETY: active_a_len<=n and padded has n+1 limbs. Copy the
                // complete active operand and initialize only its omitted data
                // and guard before the complete-width child reads them. The
                // input, xp and planned ring workspace are disjoint.
                unsafe {
                    let (prefix, padding) = padded.split_at_mut_unchecked(active_a_len);
                    prefix.copy_from_slice(active_a);
                    padding.fill(0);
                    SsaTransform::fft_sqr_mod_slices_with_executor(
                        xp,
                        padded,
                        ring_bits,
                        force_transform,
                        ring_plan,
                        executor,
                        ring_scratch,
                    );
                }
            }
        }

        // 2. Compute xm = a^2 mod (B^n - 1).
        // SAFETY: Fermat borrows have ended; this tail reserves the n-limb
        // optional fold and retained Mersenne workspace. Constructor admission
        // proves active_a_len<=n. The complete operand is borrowed at equality or
        // initialized by copy/padding before the disjoint child reads it.
        unsafe {
            let (folded, xm_scratch) = rest3.split_at_mut_unchecked(folded_len);
            let input = if folded_len == 0 {
                // The declared width includes any initialized high zero limbs.
                self.a_limbs
            } else {
                let (prefix, padding) = folded.split_at_mut_unchecked(active_a_len);
                prefix.copy_from_slice(active_a);
                padding.fill(0);
                folded
            };
            SsaCrt::sqr_mod_bnm1_prepared(xm, input, xm_scratch, executor, crt_plan);
        }

        // 3. Reconstruct dst = X_p + k * B^n + k.
        if fermat_scratch_len == 0 {
            // SAFETY: the Fermat writer initialized the complete dst[..=n]
            // prefix and the destination covers the plan's guarded width.
            unsafe {
                SsaCrt::merge_exact_product_in_place(dst, xm);
            }
        } else {
            SsaCrt::merge_exact_product(dst, fermat_buffer, xm);
        }
    }
}
