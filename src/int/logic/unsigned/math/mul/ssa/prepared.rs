//! Operand-bound SSA product planning for repeated infallible execution.

use core::num::NonZeroUsize;

use alloc::vec;

use crate::parallel::ParallelExecutor;

use super::{
    FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SsaCarry, SsaCrt, SsaPlan, SsaTransform,
    TransformChoice,
};

/// Operand-bound SSA product plan with all fallible geometry work completed.
///
/// The plan borrows the operands from which its significant widths were
/// derived. Repeated execution therefore cannot silently substitute a shape
/// whose CRT half-width or transform geometry differs from the validated one.
#[derive(Debug)]
pub struct SsaMultiplicationPlan<'operands> {
    a_limbs: &'operands [Limb],
    b_limbs: &'operands [Limb],
    result_len: usize,
    product: Option<SsaProductPlan>,
}

#[derive(Clone, Copy, Debug)]
struct SsaProductPlan {
    sig_a: usize,
    sig_b: usize,
    active_a_len: usize,
    active_b_len: usize,
    n: usize,
    ring_bits: usize,
    coeff_len: usize,
    ring_plan: FftPlan,
    force_transform: bool,
    scratch_len: usize,
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

        let sig_a = SsaPlan::significant_bits_of_slice(a_limbs);
        let sig_b = SsaPlan::significant_bits_of_slice(b_limbs);
        if sig_a == 0 || sig_b == 0 {
            return Some(Self {
                a_limbs,
                b_limbs,
                result_len,
                product: None,
            });
        }

        let required_bits = sig_a.checked_add(sig_b)?;
        let n = SsaPlan::crt_half_width(required_bits)?;
        let ring_bits = n.checked_mul(LIMB_BITS)?;
        let coeff_len = n.checked_add(1)?;
        let ring_plan = choice.exponent.map_or_else(
            || Some(FftPlan::new(ring_bits)),
            |exponent| FftPlan::try_forced(ring_bits, exponent),
        )?;
        let slots = ring_plan.parallel_slots(parallelism.get());
        let ring_scratch_len = if choice.force || ring_bits > SSA_BASE_MODULUS_BITS {
            ring_plan.transform_mul_scratch_for_slots(slots)
        } else {
            ring_plan.required_mul_scratch()
        };
        if ring_scratch_len == usize::MAX {
            return None;
        }

        let scratch_len = SsaCrt::layout_len(n, ring_scratch_len);
        if scratch_len == usize::MAX {
            return None;
        }

        Some(Self {
            a_limbs,
            b_limbs,
            result_len,
            product: Some(SsaProductPlan {
                sig_a,
                sig_b,
                active_a_len: sig_a.div_ceil(LIMB_BITS),
                active_b_len: sig_b.div_ceil(LIMB_BITS),
                n,
                ring_bits,
                coeff_len,
                ring_plan,
                force_transform: choice.force,
                scratch_len,
            }),
        })
    }

    /// Exact destination width required by this prepared product.
    #[must_use]
    pub const fn destination_len(&self) -> usize {
        self.result_len
    }

    /// Exact reusable scratch width required by this prepared product.
    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        match self.product {
            Some(product) => product.scratch_len,
            None => 0,
        }
    }

    /// Runs with a fresh zeroed workspace owned by this call.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::destination_len`] limbs and `executor`
    /// must advertise the same parallelism used to construct this plan.
    pub unsafe fn run_allocating<E: ParallelExecutor>(&self, dst: &mut [Limb], executor: &E) {
        let mut scratch = vec![0; self.scratch_len()];
        // SAFETY: this buffer has the exact planned scratch width; the caller
        // supplies the remaining destination and executor invariants.
        unsafe { self.run_prepared_with_scratch(dst, &mut scratch, executor) }
    }

    /// Runs with a caller-owned workspace sized before timing.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::destination_len`] limbs, `scratch`
    /// must contain at least [`Self::scratch_len`] limbs, and `executor` must
    /// advertise the same parallelism used to construct this plan.
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
        reason = "All slice accesses are bounded by the immutable prepared product plan"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Staging both CRT halves is one sequential construction with no reusable seam"
    )]
    #[allow(
        clippy::similar_names,
        reason = "Paired arithmetic operands conventionally use symmetric a and b names"
    )]
    unsafe fn run_prepared_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [Limb],
        scratch_buf: &mut [Limb],
        executor: &E,
    ) {
        let Some(product) = self.product else {
            dst.fill(0);
            return;
        };
        let SsaProductPlan {
            sig_a,
            sig_b,
            active_a_len,
            active_b_len,
            n,
            ring_bits,
            coeff_len,
            ring_plan,
            force_transform,
            ..
        } = product;
        // SAFETY: the plan derived both active lengths from these borrowed
        // operands, so each prefix lies within its immutable source slice.
        let active_a = unsafe { self.a_limbs.get_unchecked(..active_a_len) };
        // SAFETY: the identical operand-bound invariant holds for the right side.
        let active_b = unsafe { self.b_limbs.get_unchecked(..active_b_len) };

        let (xp, rest1) = scratch_buf.split_at_mut(coeff_len);
        let (xm, rest2) = rest1.split_at_mut(n);
        let rest3 = rest2;

        // 1. Compute xp = a * b mod (B^n + 1)
        {
            let (left_padded, rest4) = rest3.split_at_mut(coeff_len);
            let (right_padded, ring_scratch) = rest4.split_at_mut(coeff_len);

            if active_a.len() == n
                && active_b.len() == n
                && (ring_bits > SSA_BASE_MODULUS_BITS || force_transform)
            {
                // SAFETY: normalized full-width operands are nonzero, have exactly
                // ml=n data limbs, and the transform treats their omitted guards as
                // zero. The caller-owned matrices remain fully guarded.
                unsafe {
                    SsaTransform::fft_mul_mod_slices_with_executor(
                        xp,
                        active_a,
                        active_b,
                        ring_bits,
                        Some((sig_a, sig_b)),
                        force_transform,
                        Some(&ring_plan),
                        executor,
                        ring_scratch,
                    );
                }
            } else {
                let copy_a = active_a.len().min(n);
                // SAFETY: `copy_a <= n < left_padded.len() == n + 1`.
                let left_prefix = unsafe { left_padded.get_unchecked_mut(..copy_a) };
                // SAFETY: `copy_a` is also bounded by `active_a.len()`.
                let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
                left_prefix.copy_from_slice(a_prefix);
                // SAFETY: `copy_a <= n` and the inclusive end satisfies
                // `n < left_padded.len() == n + 1`.
                unsafe { left_padded.get_unchecked_mut(copy_a..=n) }.fill(0);

                let copy_b = active_b.len().min(n);
                // SAFETY: `copy_b <= n < right_padded.len() == n + 1`.
                let right_prefix = unsafe { right_padded.get_unchecked_mut(..copy_b) };
                // SAFETY: `copy_b` is also bounded by `active_b.len()`.
                let b_prefix = unsafe { active_b.get_unchecked(..copy_b) };
                right_prefix.copy_from_slice(b_prefix);
                // SAFETY: `copy_b <= n` and the inclusive end satisfies
                // `n < right_padded.len() == n + 1`.
                unsafe { right_padded.get_unchecked_mut(copy_b..=n) }.fill(0);

                if active_a.len() > n {
                    // SAFETY: `left_padded.len() == n + 1`, so its data prefix
                    // has exactly `n` limbs.
                    let left_data = unsafe { left_padded.get_unchecked_mut(..n) };
                    // SAFETY: this branch proves `n < active_a.len()`; after
                    // trimming high zeros above, the planner gives
                    // `active_a.len() <= 2n`, so this tail has at most `n` limbs.
                    let a_tail = unsafe { active_a.get_unchecked(n..) };
                    let mut borrow_a = SsaCarry::sub_full_in_place(left_data, a_tail);
                    if borrow_a > 0 {
                        // SAFETY: `left_padded.len() == n + 1`, so the data
                        // prefix has exactly `n` limbs.
                        borrow_a = borrow_a.wrapping_sub(SsaCarry::add_full_in_place(
                            unsafe { left_padded.get_unchecked_mut(..n) },
                            &[1],
                        ));
                        // SAFETY: `left_padded.len() == n + 1`, hence `n < len`.
                        *unsafe { left_padded.get_unchecked_mut(n) } =
                            1_usize.wrapping_sub(borrow_a);
                    }
                }
                if active_b.len() > n {
                    // SAFETY: `right_padded.len() == n + 1`, so its data prefix
                    // has exactly `n` limbs.
                    let right_data = unsafe { right_padded.get_unchecked_mut(..n) };
                    // SAFETY: this branch gives the lower bound, while the
                    // active-width/planner proof above gives `active_b.len() <= 2n`.
                    let b_tail = unsafe { active_b.get_unchecked(n..) };
                    let mut borrow_b = SsaCarry::sub_full_in_place(right_data, b_tail);
                    if borrow_b > 0 {
                        // SAFETY: `right_padded.len() == n + 1`, so the data
                        // prefix has exactly `n` limbs.
                        borrow_b = borrow_b.wrapping_sub(SsaCarry::add_full_in_place(
                            unsafe { right_padded.get_unchecked_mut(..n) },
                            &[1],
                        ));
                        // SAFETY: `right_padded.len() == n + 1`, hence `n < len`.
                        *unsafe { right_padded.get_unchecked_mut(n) } =
                            1_usize.wrapping_sub(borrow_b);
                    }
                }

                // SAFETY: staged buffers have their complete guarded widths.
                unsafe {
                    SsaTransform::fft_mul_mod_slices_with_executor(
                        xp,
                        left_padded,
                        right_padded,
                        ring_bits,
                        None,
                        force_transform,
                        Some(&ring_plan),
                        executor,
                        ring_scratch,
                    );
                }
            }
        }

        // 2. Compute xm = a * b mod (B^n - 1)
        {
            let (left_folded, rest4) = rest3.split_at_mut(n);
            let (right_folded, xm_scratch) = rest4.split_at_mut(n);

            if active_a.len() == n && active_b.len() == n {
                // The normalized equal-width case already is a pair of complete
                // B^n-1 operands. `mul_mod_bnm1` only reads them, so routing the
                // caller slices directly avoids two full-width staging copies.
                SsaCrt::mul_mod_bnm1(xm, active_a, active_b, xm_scratch, executor);
            } else {
                let copy_a = active_a.len().min(n);
                // SAFETY: `copy_a <= n == left_folded.len()`.
                let left_prefix = unsafe { left_folded.get_unchecked_mut(..copy_a) };
                // SAFETY: `copy_a <= active_a.len()` by its definition.
                let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
                left_prefix.copy_from_slice(a_prefix);
                // SAFETY: `copy_a <= n == left_folded.len()`; equality yields
                // an empty clear range.
                unsafe { left_folded.get_unchecked_mut(copy_a..n) }.fill(0);
                if active_a.len() > n {
                    // SAFETY: the branch proves the start is in range and the
                    // active-width proof gives a tail length at most `n`, which
                    // fits `left_folded` exactly.
                    let a_tail = unsafe { active_a.get_unchecked(n..) };
                    let mut carry_a = SsaCarry::add_full_in_place(left_folded, a_tail);
                    if carry_a > 0 {
                        carry_a = SsaCarry::add_full_in_place(left_folded, &[carry_a]);
                        if carry_a > 0 {
                            let _ = SsaCarry::add_full_in_place(left_folded, &[carry_a]);
                        }
                    }
                }

                let copy_b = active_b.len().min(n);
                // SAFETY: `copy_b <= n == right_folded.len()`.
                let right_prefix = unsafe { right_folded.get_unchecked_mut(..copy_b) };
                // SAFETY: `copy_b <= active_b.len()` by its definition.
                let b_prefix = unsafe { active_b.get_unchecked(..copy_b) };
                right_prefix.copy_from_slice(b_prefix);
                // SAFETY: `copy_b <= n == right_folded.len()`; equality yields
                // an empty clear range.
                unsafe { right_folded.get_unchecked_mut(copy_b..n) }.fill(0);
                if active_b.len() > n {
                    // SAFETY: the branch proves the start is in range and the
                    // active-width proof gives a tail length at most `n`.
                    let b_tail = unsafe { active_b.get_unchecked(n..) };
                    let mut carry_b = SsaCarry::add_full_in_place(right_folded, b_tail);
                    if carry_b > 0 {
                        carry_b = SsaCarry::add_full_in_place(right_folded, &[carry_b]);
                        if carry_b > 0 {
                            let _ = SsaCarry::add_full_in_place(right_folded, &[carry_b]);
                        }
                    }
                }

                SsaCrt::mul_mod_bnm1(xm, left_folded, right_folded, xm_scratch, executor);
            }
        }

        // 3. Reconstruct dst = X_p + k * B^n + k.
        SsaCrt::merge_exact_product(dst, xp, xm);
    }
}
