//! Operand-bound SSA square planning for repeated infallible execution.

use crate::parallel::ParallelExecutor;

use super::{
    FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SsaCarry, SsaCrt, SsaPlan, SsaTransform,
    TransformChoice,
};

/// Operand-bound SSA square plan with all fallible geometry work completed.
///
/// The plan borrows the operand from which its significant width was derived.
/// Repeated execution therefore cannot substitute an operand whose CRT
/// half-width or transform geometry differs from the validated one.
#[derive(Debug)]
pub struct SsaSquaringPlan<'operand> {
    a_limbs: &'operand [Limb],
    result_len: usize,
    square: Option<SsaSquareGeometry>,
}

#[derive(Clone, Copy, Debug)]
struct SsaSquareGeometry {
    active_a_len: usize,
    n: usize,
    ring_bits: usize,
    coeff_len: usize,
    ring_plan: FftPlan,
    force_transform: bool,
    scratch_len: usize,
}

impl<'operand> SsaSquaringPlan<'operand> {
    /// Builds an exact plan for an immutable operand and one executor width.
    ///
    /// Returns `None` only when the square dimensions cannot be represented or
    /// a transform workspace size overflows.
    pub fn try_new(
        a_limbs: &'operand [Limb],
        choice: TransformChoice,
        parallelism: usize,
    ) -> Option<Self> {
        let result_len = a_limbs.len().checked_mul(2)?;
        let sig_a = SsaPlan::significant_bits_of_slice(a_limbs);
        if sig_a == 0 {
            return Some(Self {
                a_limbs,
                result_len,
                square: None,
            });
        }

        let required_bits = sig_a.checked_mul(2)?;
        let n = SsaPlan::crt_half_width(required_bits)?;
        let ring_bits = n.checked_mul(LIMB_BITS)?;
        let coeff_len = n.checked_add(1)?;
        let ring_plan = FftPlan::new(ring_bits);
        let slots = ring_plan.parallel_slots(parallelism.max(1));
        let ring_scratch_len = if choice.force || ring_bits > SSA_BASE_MODULUS_BITS {
            ring_plan.transform_sqr_scratch_for_slots(slots)
        } else {
            ring_plan.required_sqr_scratch()
        };
        if ring_scratch_len == usize::MAX {
            return None;
        }
        let scratch_len = SsaCrt::sqr_layout_len(n, ring_scratch_len);
        if scratch_len == usize::MAX {
            return None;
        }

        Some(Self {
            a_limbs,
            result_len,
            square: Some(SsaSquareGeometry {
                active_a_len: sig_a.div_ceil(LIMB_BITS),
                n,
                ring_bits,
                coeff_len,
                ring_plan,
                force_transform: choice.force,
                scratch_len,
            }),
        })
    }

    /// Exact destination width required by this prepared square.
    #[must_use]
    pub const fn destination_len(&self) -> usize {
        self.result_len
    }

    /// Exact reusable scratch width required by this prepared square.
    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        match self.square {
            Some(square) => square.scratch_len,
            None => 0,
        }
    }

    /// Runs with a caller-owned workspace sized before timing.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::destination_len`] limbs, `scratch`
    /// must contain at least [`Self::scratch_len`] limbs, and `executor` must
    /// advertise the parallelism used to construct this plan.
    pub unsafe fn run_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        // SAFETY: the caller proves this exact prefix exists.
        let planned = unsafe { scratch.get_unchecked_mut(..self.scratch_len()) };
        // SAFETY: all prepared execution invariants are inherited from this method.
        unsafe { self.run_prepared_with_scratch(dst, planned, executor) }
    }

    #[allow(
        unsafe_code,
        reason = "All slice accesses are bounded by the immutable prepared square plan"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "The two CRT residues and their reconstruction are one sequential prepared execution"
    )]
    unsafe fn run_prepared_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [Limb],
        scratch_buf: &mut [Limb],
        executor: &E,
    ) {
        let Some(square) = self.square else {
            dst.fill(0);
            return;
        };
        let SsaSquareGeometry {
            active_a_len,
            n,
            ring_bits,
            coeff_len,
            ring_plan,
            force_transform,
            ..
        } = square;
        // SAFETY: the significant-bit count was derived from this immutable
        // operand, so its rounded-up active prefix is within the source slice.
        let active_a = unsafe { self.a_limbs.get_unchecked(..active_a_len) };

        let (xp, rest1) = scratch_buf.split_at_mut(coeff_len);
        let (xm, rest2) = rest1.split_at_mut(n);
        let rest3 = rest2;

        // 1. Compute xp = a^2 mod (B^n + 1).
        {
            let (padded, ring_scratch) = rest3.split_at_mut(coeff_len);

            if active_a.len() == n && (ring_bits > SSA_BASE_MODULUS_BITS || force_transform) {
                // SAFETY: normalized full-width operand is nonzero, has exactly
                // ml=n data limbs, and the transform treats its omitted guard as
                // zero.
                unsafe {
                    SsaTransform::fft_sqr_mod_slices_with_executor(
                        xp,
                        active_a,
                        ring_bits,
                        force_transform,
                        Some(&ring_plan),
                        executor,
                        ring_scratch,
                    );
                }
            } else {
                let copy_a = active_a.len().min(n);
                // SAFETY: `copy_a <= n < padded.len() == n + 1`.
                let padded_prefix = unsafe { padded.get_unchecked_mut(..copy_a) };
                // SAFETY: `copy_a <= active_a.len()` by construction.
                let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
                padded_prefix.copy_from_slice(a_prefix);
                // SAFETY: `copy_a <= n` and the inclusive end is in `padded`.
                unsafe { padded.get_unchecked_mut(copy_a..=n) }.fill(0);

                // Above the half-width the operand folds negacyclically: B^n = -1.
                if active_a.len() > n {
                    // SAFETY: `padded.len() == n + 1`, so this is its data span.
                    let padded_data = unsafe { padded.get_unchecked_mut(..n) };
                    // SAFETY: the square-width proof bounds this tail by `n` limbs.
                    let a_tail = unsafe { active_a.get_unchecked(n..) };
                    let mut borrow = SsaCarry::sub_full_in_place(padded_data, a_tail);
                    if borrow > 0 {
                        // SAFETY: `padded.len() == n + 1`, so the data span is valid.
                        borrow = borrow.wrapping_sub(SsaCarry::add_full_in_place(
                            unsafe { padded.get_unchecked_mut(..n) },
                            &[1],
                        ));
                        // SAFETY: `n < padded.len()` by its guard-limb layout.
                        *unsafe { padded.get_unchecked_mut(n) } = 1_usize.wrapping_sub(borrow);
                    }
                }

                // SAFETY: the staged operand is guarded, disjoint from `xp`, and the
                // ring workspace was sized by this immutable plan.
                unsafe {
                    SsaTransform::fft_sqr_mod_slices_with_executor(
                        xp,
                        padded,
                        ring_bits,
                        force_transform,
                        Some(&ring_plan),
                        executor,
                        ring_scratch,
                    );
                }
            }
        }

        // 2. Compute xm = a^2 mod (B^n - 1).
        {
            let (folded, xm_scratch) = rest3.split_at_mut(n);

            if active_a.len() == n {
                // A normalized full-width operand is already a complete residue.
                SsaCrt::sqr_mod_bnm1(xm, active_a, xm_scratch, executor);
            } else {
                let copy_a = active_a.len().min(n);
                // SAFETY: `copy_a <= n == folded.len()`.
                let folded_prefix = unsafe { folded.get_unchecked_mut(..copy_a) };
                // SAFETY: `copy_a <= active_a.len()` by construction.
                let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
                folded_prefix.copy_from_slice(a_prefix);
                // SAFETY: the range is within the exact folded residue span.
                unsafe { folded.get_unchecked_mut(copy_a..n) }.fill(0);
                if active_a.len() > n {
                    // SAFETY: the square-width proof bounds this tail by `n` limbs.
                    let a_tail = unsafe { active_a.get_unchecked(n..) };
                    let mut carry = SsaCarry::add_full_in_place(folded, a_tail);
                    if carry > 0 {
                        carry = SsaCarry::add_full_in_place(folded, &[carry]);
                        if carry > 0 {
                            let _ = SsaCarry::add_full_in_place(folded, &[carry]);
                        }
                    }
                }

                SsaCrt::sqr_mod_bnm1(xm, folded, xm_scratch, executor);
            }
        }

        // 3. Reconstruct dst = X_p + k * B^n + k.
        SsaCrt::merge_exact_product(dst, xp, xm);
    }
}
