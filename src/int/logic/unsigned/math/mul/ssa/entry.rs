//! The [`Ssa`] namespace: the tower's product and square, the geometry choice
//! they take, and the answers derived from operand widths alone.
//!
//! Capability and scratch length both follow from the declared widths, which is
//! what lets the dispatcher decide a tier and size a buffer before any operand is
//! inspected. Each entry point stages its operands into the `B^n + 1` and
//! `B^n - 1` halves and hands both residues to the shared exact reconstruction in
//! [`crt`](super::crt).
//!
//! The tuner-only surface — forcing a transform the planner would decline, and the
//! full-width Fermat product that bypasses the CRT split — is not here: it lives
//! on the multiplication tuning namespace.
//!
//! The whole module is gated off 16-bit targets by its declaration in the parent,
//! because the CRT half-widths it computes cannot be represented there.

use alloc::vec;

use super::{
    FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SsaCarry, SsaCrt, SsaPlan, SsaTransform,
};

/// Namespace for recursive Schönhage-Strassen multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ssa;

/// How a top-level SSA product or square picks its transform geometry.
///
/// Production always passes [`Self::PLANNED`]. The tuner and the tier tests force
/// a transform on rings the planner would otherwise leave to the basecase, and can
/// pin the exponent, so a forced measurement runs through the *same*
/// implementation as production rather than a parallel copy of it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransformChoice {
    /// Transform even where the ring is narrow enough for the basecase.
    pub force: bool,
    /// Use this transform exponent instead of the planner's choice.
    pub exponent: Option<u32>,
}

impl TransformChoice {
    /// Let the planner decide both the geometry and whether to transform at all.
    pub const PLANNED: Self = Self {
        force: false,
        exponent: None,
    };

    /// Force the transform while leaving its geometry to the planner.
    #[cfg(any(test, feature = "_internal-tune"))]
    pub const FORCED: Self = Self {
        force: true,
        exponent: None,
    };

    /// Force one exact transform exponent for measurement.
    #[cfg(feature = "_internal-tune")]
    pub const fn forced_at(exponent: u32) -> Self {
        Self {
            force: true,
            exponent: Some(exponent),
        }
    }
}

impl Ssa {
    /// Multiply two limb slices with recursive Fermat-ring FFT multiplication.
    ///
    /// `scratch` of [`Ssa::mul_scratch_len`] limbs keeps the call allocation-free;
    /// `None` makes it allocate its own, which only the tuner and the tier tests do.
    /// Returns `false` when these widths have no representable CRT half-width, or
    /// when a pinned exponent yields no usable geometry.
    #[allow(
        unsafe_code,
        reason = "All slice accesses are bounded by exact mathematically constructed lengths"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Staging both CRT halves is one sequential construction with no reusable seam"
    )]
    #[allow(
        clippy::similar_names,
        reason = "Paired arithmetic operands conventionally use symmetric a and b names"
    )]
    pub fn try_mul(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        choice: TransformChoice,
        mut scratch: Option<&mut [Limb]>,
    ) -> bool {
        let TransformChoice {
            force: force_transform,
            exponent: transform_exponent_override,
        } = choice;
        if a_limbs.is_empty() || b_limbs.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(result_len) = a_limbs.len().checked_add(b_limbs.len()) else {
            return false;
        };
        if dst.len() < result_len {
            return false;
        }
        if a_limbs.len().checked_mul(LIMB_BITS).is_none()
            || b_limbs.len().checked_mul(LIMB_BITS).is_none()
        {
            return false;
        }

        let sig_a = SsaPlan::significant_bits_of_slice(a_limbs);
        let sig_b = SsaPlan::significant_bits_of_slice(b_limbs);
        if sig_a == 0 || sig_b == 0 {
            dst.fill(0);
            return true;
        }

        let Some(required_bits) = sig_a.checked_add(sig_b) else {
            return false;
        };
        let Some(n) = SsaPlan::crt_half_width(required_bits) else {
            return false;
        };
        let Some(ring_bits) = n.checked_mul(LIMB_BITS) else {
            return false;
        };
        let Some(coeff_len) = n.checked_add(1) else {
            return false;
        };

        // Ignore storage-only high zeros before deriving the CRT folds. The
        // selected half-width satisfies `sig_a + sig_b <= 2*n*LIMB_BITS`.
        // Since each significant width is at most that sum, each active operand
        // contains at most `2n` limbs and its tail above `n` fits an `n`-limb
        // residue on every supported limb width.
        let active_a_len = sig_a.div_ceil(LIMB_BITS);
        let active_b_len = sig_b.div_ceil(LIMB_BITS);
        // SAFETY: `significant_bits_of_slice` returns at most
        // `a_limbs.len() * LIMB_BITS`, so `active_a_len <= a_limbs.len()`.
        let active_a = unsafe { a_limbs.get_unchecked(..active_a_len) };
        // SAFETY: the identical significant-width bound holds for `b_limbs`.
        let active_b = unsafe { b_limbs.get_unchecked(..active_b_len) };

        // Size the buffer from the half-width this call actually uses, not from the
        // operands' limb widths. The caller's allocation is derived from the limb
        // widths, and `SsaCrt::layout_len` is monotone in the half-width, so a
        // leading-zero-heavy operand lands strictly inside what it reserved.
        //
        // A forced geometry is validated here too, against the ring the transform
        // really runs in — the CRT half-width, not the full product width.
        // Built once and threaded through to the transform, so the geometry the
        // scratch is sized from is literally the geometry that runs.
        let ring_plan = match transform_exponent_override {
            Some(transform_exponent) => {
                let Some(forced) = FftPlan::try_forced(ring_bits, transform_exponent) else {
                    return false;
                };
                forced
            }
            None => FftPlan::new(ring_bits),
        };
        let ring_scratch_len = if force_transform {
            ring_plan.transform_mul_scratch()
        } else {
            ring_plan.required_mul_scratch()
        };
        let total_needed = SsaCrt::layout_len(n, ring_scratch_len);

        let mut owned;
        let scratch_buf: &mut [Limb] = match scratch {
            // SAFETY: total_needed <= s.len()
            Some(ref mut s) if s.len() >= total_needed => unsafe {
                s.get_unchecked_mut(..total_needed)
            },
            _ => {
                debug_assert!(
                    transform_exponent_override.is_none(),
                    "forced-plan scratch is undersized; use Ssa::mul_scratch_len_for_plan"
                );
                owned = vec![0; total_needed];
                &mut owned
            }
        };

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
                    SsaTransform::fft_mul_mod_slices(
                        xp,
                        active_a,
                        active_b,
                        ring_bits,
                        Some((sig_a, sig_b)),
                        force_transform,
                        Some(&ring_plan),
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
                    SsaTransform::fft_mul_mod_slices(
                        xp,
                        left_padded,
                        right_padded,
                        ring_bits,
                        None,
                        force_transform,
                        Some(&ring_plan),
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
                SsaCrt::mul_mod_bnm1(xm, active_a, active_b, xm_scratch);
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

                SsaCrt::mul_mod_bnm1(xm, left_folded, right_folded, xm_scratch);
            }
        }

        // 3. Reconstruct dst = X_p + k * B^n + k.
        SsaCrt::merge_exact_product(dst, xp, xm);

        true
    }

    /// Whether [`Self::try_mul`] can compute a product of these operand widths.
    ///
    /// This is a *capability* predicate, not a crossover: it reports whether the
    /// construction exists for these widths, and says nothing about whether it is
    /// the fastest tier for them. The dispatcher applies `SSA_THRESHOLD` on top.
    ///
    /// Keeping the two separate is what lets the dispatcher name a tier instead of
    /// merely attempting one. Every rejection below mirrors a `false` return in
    /// [`Self::try_mul`], and `mul::dispatch::tests` sweeps the two against each
    /// other, so a plan naming this tier is a plan that runs.
    ///
    /// The bound is computed from the declared widths. A leading-zero-heavy operand
    /// only shortens the significant width, and [`crt_half_width`] is monotone in
    /// it, so an accepted pair stays accepted once the exact widths are known.
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        let Some(product_width) = len_a.checked_add(len_b) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// The squaring counterpart of [`Self::admits_mul`].
    pub fn admits_sqr(len: usize) -> bool {
        let Some(product_width) = len.checked_mul(2) else {
            return false;
        };
        admits_product_width(product_width)
    }

    /// Scratch required for one [`Self::try_mul`] call on operands of these limb
    /// widths, using the geometry the planner selects.
    ///
    /// Returns zero when the widths cannot describe a representable product, which
    /// is also the case the entry point declines.
    pub fn mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let Some(half_width) = SsaPlan::crt_half_width_for_operands(len_a, len_b) else {
            return 0;
        };
        // The top-level ring is always transformed, so it must be sized for the
        // transform layout even when it is narrow enough for the basecase.
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        let ring_scratch = SsaPlan::forced_scratch_len_for_ring(ring_bits);
        SsaCrt::layout_len(half_width, ring_scratch)
    }

    /// Scratch required to square a full-width operand of `len` limbs.
    ///
    /// The significant value can only shorten the selected half-width, and
    /// [`SsaCrt::sqr_layout_len`] is monotone in it, so sizing from the complete
    /// declared width is a valid reusable upper bound.
    ///
    /// Returns zero when the width cannot describe a representable square, which is
    /// also the case the entry point declines.
    pub fn sqr_scratch_len(len: usize) -> usize {
        let Some(required_bits) = len
            .checked_mul(2)
            .and_then(|width| width.checked_mul(LIMB_BITS))
        else {
            return 0;
        };
        let Some(half_width) = SsaPlan::crt_half_width(required_bits) else {
            return 0;
        };
        let Some(ring_bits) = half_width.checked_mul(LIMB_BITS) else {
            return 0;
        };
        // The top-level ring is always transformed, so it must be sized for the
        // transform layout even when it is narrow enough for the basecase.
        SsaCrt::sqr_layout_len(half_width, FftPlan::new(ring_bits).required_sqr_scratch())
    }

    /// Square a limb slice with recursive Fermat-ring FFT squaring.
    ///
    /// The CRT split is what makes a square cheaper than a general product. A single
    /// Fermat ring wide enough to hold the exact square would be simpler, but it
    /// discards that discount entirely, and sizing such a ring means rounding the
    /// product width up to the next power of two — which on any width that is not
    /// already one inflates the ring by the rounding ratio, worth 1.67x at five
    /// million limbs. Reaching [`crt_half_width`] avoids both.
    ///
    /// Takes the same [`TransformChoice`] and optional scratch as [`Ssa::try_mul`].
    /// [`TransformChoice::exponent`] has no effect here: the squaring transform plans
    /// its own ring rather than accepting one, so there is nowhere to pin it without
    /// the sizing and the running geometry disagreeing. No caller pins one.
    #[allow(
        unsafe_code,
        reason = "All slice accesses are bounded by exact mathematically constructed lengths"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Squaring mirrors the product orchestration step for step, keeping both callable from one entry"
    )]
    pub fn try_sqr(
        dst: &mut [Limb],
        a_limbs: &[Limb],
        choice: TransformChoice,
        mut scratch: Option<&mut [Limb]>,
    ) -> bool {
        let force_transform = choice.force;
        debug_assert!(
            choice.exponent.is_none(),
            "squaring has no exponent override; see this function's documentation"
        );
        if a_limbs.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(result_len) = a_limbs.len().checked_mul(2) else {
            return false;
        };
        if dst.len() < result_len {
            return false;
        }
        if a_limbs.len().checked_mul(LIMB_BITS).is_none() {
            return false;
        }
        let sig_a = SsaPlan::significant_bits_of_slice(a_limbs);
        if sig_a == 0 {
            dst.fill(0);
            return true;
        }
        let Some(required_bits) = sig_a.checked_mul(2) else {
            return false;
        };
        let Some(n) = SsaPlan::crt_half_width(required_bits) else {
            return false;
        };
        let Some(ring_bits) = n.checked_mul(LIMB_BITS) else {
            return false;
        };
        let Some(coeff_len) = n.checked_add(1) else {
            return false;
        };
        let active_a_len = sig_a.div_ceil(LIMB_BITS);
        // SAFETY: the significant-bit count is bounded by the supplied slice
        // width, so its rounded-up active limb count cannot exceed the slice.
        let active_a = unsafe { a_limbs.get_unchecked(..active_a_len) };

        let ring_plan = FftPlan::new(ring_bits);
        let ring_scratch_len = if force_transform {
            ring_plan.transform_sqr_scratch()
        } else {
            ring_plan.required_sqr_scratch()
        };
        let total_needed = SsaCrt::sqr_layout_len(n, ring_scratch_len);

        let mut owned;
        let scratch_buf: &mut [Limb] = match scratch {
            // SAFETY: total_needed <= s.len()
            Some(ref mut s) if s.len() >= total_needed => unsafe {
                s.get_unchecked_mut(..total_needed)
            },
            _ => {
                owned = vec![0; total_needed];
                &mut owned
            }
        };

        let (xp, rest1) = scratch_buf.split_at_mut(coeff_len);
        let (xm, rest2) = rest1.split_at_mut(n);
        let rest3 = rest2;

        // 1. Compute xp = a^2 mod (B^n + 1)
        {
            let (padded, ring_scratch) = rest3.split_at_mut(coeff_len);

            let copy_a = active_a.len().min(n);
            // SAFETY: `copy_a <= n < padded.len() == n + 1`.
            let padded_prefix = unsafe { padded.get_unchecked_mut(..copy_a) };
            // SAFETY: `copy_a <= active_a.len()` by its definition.
            let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
            padded_prefix.copy_from_slice(a_prefix);
            // SAFETY: `copy_a <= n` and the inclusive end satisfies
            // `n < padded.len() == n + 1`.
            unsafe { padded.get_unchecked_mut(copy_a..=n) }.fill(0);

            // Above the half-width the operand folds negacyclically: `B^n = -1`, so
            // the tail is subtracted rather than appended.
            if active_a.len() > n {
                // SAFETY: `padded.len() == n + 1`, so this is its exact data span.
                let padded_data = unsafe { padded.get_unchecked_mut(..n) };
                // SAFETY: the branch gives `n < active_a.len()` and
                // `required_bits == 2*sig_a <= 2*n*LIMB_BITS` proves the active
                // operand has at most `2n` limbs; its tail therefore has at most
                // `n` limbs and fits the data span.
                let a_tail = unsafe { active_a.get_unchecked(n..) };
                let mut borrow = SsaCarry::sub_full_in_place(padded_data, a_tail);
                if borrow > 0 {
                    // SAFETY: `padded.len() == n + 1`, so the data prefix has
                    // exactly `n` limbs.
                    borrow = borrow.wrapping_sub(SsaCarry::add_full_in_place(
                        unsafe { padded.get_unchecked_mut(..n) },
                        &[1],
                    ));
                    // SAFETY: `padded.len() == n + 1`, hence `n < len`.
                    *unsafe { padded.get_unchecked_mut(n) } = 1_usize.wrapping_sub(borrow);
                }
            }

            // SAFETY: the staged operand has its complete guarded width, is disjoint
            // from xp, and the ring scratch was sized from this very plan.
            unsafe {
                SsaTransform::fft_sqr_mod_slices(
                    xp,
                    padded,
                    ring_bits,
                    force_transform,
                    ring_scratch,
                );
            }
        }

        // 2. Compute xm = a^2 mod (B^n - 1)
        {
            let (folded, xm_scratch) = rest3.split_at_mut(n);

            if active_a.len() == n {
                // A normalized full-width operand already is a complete `B^n - 1`
                // residue, and `SsaCrt::sqr_mod_bnm1` only reads it, so the staging copy is
                // skipped entirely for the dominant shape.
                SsaCrt::sqr_mod_bnm1(xm, active_a, xm_scratch);
            } else {
                let copy_a = active_a.len().min(n);
                // SAFETY: `copy_a <= n == folded.len()`.
                let folded_prefix = unsafe { folded.get_unchecked_mut(..copy_a) };
                // SAFETY: `copy_a <= active_a.len()` by its definition.
                let a_prefix = unsafe { active_a.get_unchecked(..copy_a) };
                folded_prefix.copy_from_slice(a_prefix);
                // SAFETY: `copy_a <= n == folded.len()`; equality yields an
                // empty clear range.
                unsafe { folded.get_unchecked_mut(copy_a..n) }.fill(0);
                if active_a.len() > n {
                    // SAFETY: the branch proves the start is in range and the
                    // square-width proof above bounds this tail by `n` limbs.
                    let a_tail = unsafe { active_a.get_unchecked(n..) };
                    let mut carry = SsaCarry::add_full_in_place(folded, a_tail);
                    if carry > 0 {
                        carry = SsaCarry::add_full_in_place(folded, &[carry]);
                        if carry > 0 {
                            let _ = SsaCarry::add_full_in_place(folded, &[carry]);
                        }
                    }
                }

                SsaCrt::sqr_mod_bnm1(xm, folded, xm_scratch);
            }
        }

        // 3. Reconstruct dst = X_p + k * B^n + k.
        SsaCrt::merge_exact_product(dst, xp, xm);

        true
    }
}

/// Whether a product this many limbs wide has a representable CRT half-width.
fn admits_product_width(product_width: usize) -> bool {
    let Some(required_bits) = product_width.checked_mul(LIMB_BITS) else {
        return false;
    };
    let Some(half_width) = SsaPlan::crt_half_width(required_bits) else {
        return false;
    };
    half_width.checked_mul(LIMB_BITS).is_some()
}
