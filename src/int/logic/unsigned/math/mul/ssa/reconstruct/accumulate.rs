//! Coefficient-matrix reconstruction into a destination product.
//!
//! `SsaCoefficients::split` cuts an operand into radix-`2^chunk_bits` coefficients;
//! `SsaCoefficients::reconstruct` accumulates the inverse-transformed coefficients
//! back into a product. They are exact inverses either side of the transform,
//! so the chunk geometry that one assumes is the one the other undoes.
//!
//! After the inverse FFT and twiddle/scaling corrections, each coefficient
//! `c[i]` contributes `c[i] * B^(i * chunk_bits)` to the product, reduced
//! modulo `2^mod_bits + 1`. Coefficients that exceed the correction threshold
//! are treated as negative residues.
//!
//! The sweep separates the two signs, then folds:
//!
//! - `process_positive_coeff`: coefficients that keep their sign and are added.
//! - `process_negative_coeff`: coefficients read as negative residues, whose
//!   magnitude is recovered against the modulus and subtracted.
//! - `SsaCoefficients::fold_high_into_low`: the closing reduction of the accumulator by
//!   `2^n = -1`.

#![allow(
    unsafe_code,
    reason = "Direct limb-level accumulation for zero-allocation FFT reconstruct"
)]

// Re-exported for the submodules below so they reach the rest of the tier
// through `super::` rather than through a deep relative path.
use super::{
    Addition, ArchKernels, LIMB_BITS, Limb, SharedEval, SsaCarry, SsaCoefficients, SsaRing,
    SsaTransform,
};

/// The inverse twiddle and `1 / transform_len` scaling that follow the inverse
/// transform.
///
/// Carrying the three geometry values rather than the whole plan keeps this
/// module independent of the orchestration layer, so the correction has exactly
/// one definition and both drivers reach it the same way.
#[derive(Clone, Copy, Debug)]
pub struct InverseTwist {
    /// Fermat ring modulus bit width.
    pub inner_bits: usize,
    /// Base-two logarithm of the transform length.
    pub transform_log: usize,
    /// Forward twist increment per coefficient, in half-bit units.
    pub twist_step_half: usize,
}

impl InverseTwist {
    /// Total inverse shift for one coefficient, in half-bit units.
    ///
    /// Folds the inverse twiddle together with the `1 / transform_len` scaling,
    /// reduced modulo the ring period. Every term of the whole-bit derivation is
    /// doubled so an odd result denotes a remaining `sqrt(2)` factor.
    #[must_use]
    pub const fn shift_for(&self, index: usize) -> usize {
        let full_period = self.inner_bits.wrapping_mul(4);
        let scale_shift = self
            .inner_bits
            .wrapping_sub(self.transform_log)
            .wrapping_mul(2);
        let untwist = index.wrapping_mul(self.twist_step_half);
        let inverse_shift = SsaRing::reduce_mod_period(
            self.inner_bits
                .wrapping_mul(2)
                .wrapping_add(full_period)
                .wrapping_sub(untwist),
            full_period,
        );
        SsaRing::reduce_mod_period(inverse_shift.wrapping_add(scale_shift), full_period)
    }
}

/// Applies the inverse twiddle to one coefficient in place.
///
/// The coefficient is staged through `scratch` because a slot leaving the
/// inverse transform is only semi-normalized, which is what `fermat_shift_from`
/// accepts and the canonical in-place shift rejects.
///
/// # Safety
/// - `matrix` holds `transform_len` complete `SsaRing::coeff_limbs(inner_bits)` slots and
///   `index` is one of them.
/// - `scratch` is disjoint from `matrix` and holds at least two coefficients.
unsafe fn untwist_coefficient(
    matrix: &mut [Limb],
    index: usize,
    twist: &InverseTwist,
    scratch: &mut [Limb],
) {
    let inner_cl = SsaRing::coeff_limbs(twist.inner_bits);
    let total_shift = twist.shift_for(index);
    // SAFETY: the caller guarantees `index` addresses a complete slot.
    let slot = unsafe { SsaTransform::coeff_mut(matrix, index, inner_cl) };
    if total_shift != 0 {
        // SAFETY: the staging span is a disjoint complete coefficient.
        let stage = unsafe { scratch.get_unchecked_mut(..inner_cl) };
        stage.copy_from_slice(slot);
        // SAFETY: slot and stage are disjoint complete coefficients.
        unsafe {
            SsaRing::shift_from(slot, stage, total_shift.wrapping_shr(1), twist.inner_bits);
            if !total_shift.is_multiple_of(2) {
                // SAFETY: the staged copy is dead once the shift has read it, so
                // the whole two-coefficient buffer is free again here.
                SsaRing::mul_sqrt2(slot, twist.inner_bits, scratch);
            }
        }
    }
    // A zero shift still leaves the inverse transform semi-normalized.
    // SAFETY: slot is a complete coefficient and inner_bits matches.
    unsafe {
        SsaRing::normalize(slot, twist.inner_bits);
    }
}

impl SsaCoefficients {
    /// Accumulates inverse-transformed coefficients from `matrix` into the
    /// destination product buffer `dst`.
    ///
    /// After the inverse FFT and twiddle/scaling corrections, each coefficient
    /// `c[i]` contributes `c[i] * B^(i * chunk_bits)` to the product, where
    /// `B = 2` and the contribution is reduced modulo `2^mod_bits + 1`.
    ///
    /// Coefficients that exceed the correction threshold are treated as negative
    /// residues (subtracted instead of added).
    ///
    /// # Arguments
    /// - `matrix`: flat coefficient buffer (post-IFFT, post-scaling)
    /// - `transform_len`: number of coefficient slots
    /// - `chunk_bits`: radix chunk width
    /// - `inner_bits`: Fermat ring modulus bit width
    /// - `mod_bits`: outer modulus bit width for the final product
    /// - `dst`: destination product buffer
    /// - `scratch`: temporary buffer; must be large enough for both accumulator
    ///   (see internal computation) and work area (>= max shifted coeff width)
    #[allow(
        clippy::too_many_arguments,
        reason = "the fused sweep needs the chunk geometry, both output buffers, and the inverse twist"
    )]
    pub fn reconstruct(
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
        mod_bits: usize,
        dst: &mut [Limb],
        scratch: &mut [Limb],
        twist: Option<(InverseTwist, &mut [Limb])>,
    ) {
        let cl = SsaRing::coeff_limbs(inner_bits);
        let ml_inner = SsaRing::mod_limbs(inner_bits);
        let ml_outer = SsaRing::mod_limbs(mod_bits);
        let outer_cl = ml_outer.wrapping_add(1);

        // Maximum width needed for the accumulator: each coefficient (up to
        // `cl` limbs) can be shifted by up to `(transform_len-1)*chunk_bits`
        // bits. The total must be held before folding modulo 2^mod_bits+1.
        let max_limbs_contrib = cl.wrapping_add(
            (transform_len.wrapping_sub(1))
                .wrapping_mul(chunk_bits)
                .wrapping_div(LIMB_BITS)
                .wrapping_add(1),
        );

        // Split scratch into accumulator (first acc_limbs limbs) and work
        // area (remaining limbs) for process_* functions. These must not overlap.
        let acc_limbs = max_limbs_contrib.max(outer_cl);
        // SAFETY: caller guarantees scratch.len() >= acc_limbs.
        let (acc, work) = scratch.split_at_mut(acc_limbs);
        acc.fill(0);
        // Bias the signed reconstruction by one outer modulus. Processing all
        // positive coefficients before all negative magnitudes then cannot
        // underflow: the completed signed negacyclic product is `low - high`,
        // strictly between -2^mod_bits and 2^mod_bits, so adding 2^mod_bits+1
        // keeps every suffix of the subtraction phase positive. The bias folds to
        // zero at the end.
        // SAFETY: acc_limbs >= outer_cl = ml_outer + 1.
        unsafe {
            *acc.get_unchecked_mut(0) = 1;
            *acc.get_unchecked_mut(ml_outer) = 1;
        }

        #[allow(
            clippy::as_conversions,
            reason = "a usize trailing-zero count is at most Limb::BITS and fits every matching usize"
        )]
        let transform_log = transform_len.trailing_zeros() as usize;
        let coefficient_bound_bits = chunk_bits.wrapping_mul(2).wrapping_add(transform_log);
        debug_assert!(
            coefficient_bound_bits < inner_bits,
            "the centered coefficient bound must leave a sign-separation bit"
        );

        let mut twist_stage = twist;
        for negative_pass in [false, true] {
            for idx in 0..transform_len {
                // The inverse twiddle is applied on first touch instead of in a
                // separate sweep, so the coefficient matrix is read once rather than
                // read, rewritten, and read again. The second pass therefore sees
                // slots that the first pass already corrected.
                if !negative_pass
                    && let Some((ref stage_twist, ref mut stage_scratch)) = twist_stage
                {
                    // SAFETY: idx < transform_len and the caller guarantees the
                    // staging buffer holds two disjoint complete coefficients.
                    unsafe {
                        untwist_coefficient(matrix, idx, stage_twist, stage_scratch);
                    }
                }
                // SAFETY: idx < transform_len, matrix has transform_len * cl limbs.
                let coeff_slice = unsafe { SsaTransform::coeff(matrix, idx, cl) };

                // Exact convolution coefficients have magnitude below
                // 2^coefficient_bound_bits, which is below 2^(inner_bits-1).
                // Therefore canonical residues with the top data bit set are
                // precisely negative coefficients; the guard-only value is -1.
                // SAFETY: 0 < ml_inner < cl <= coeff_slice.len().
                let top_data = unsafe { *coeff_slice.get_unchecked(ml_inner.wrapping_sub(1)) };
                // SAFETY: ml_inner < cl <= coeff_slice.len().
                let guard = unsafe { *coeff_slice.get_unchecked(ml_inner) };
                let is_negative = guard != 0 || top_data.leading_zeros() == 0;
                if is_negative != negative_pass {
                    continue;
                }

                let shift_bits = idx.wrapping_mul(chunk_bits);
                let shift_limbs = shift_bits.wrapping_div(LIMB_BITS);
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "shift_bits % LIMB_BITS < LIMB_BITS; fits u32"
                )]
                let shift_sub_bits = shift_bits.wrapping_rem(LIMB_BITS) as u32;

                if is_negative {
                    // SAFETY: work.len() >= cl (guaranteed by caller), acc.len() >= needed.
                    unsafe {
                        process_negative_coeff(
                            coeff_slice,
                            shift_limbs,
                            shift_sub_bits,
                            cl,
                            ml_inner,
                            acc,
                            work,
                        );
                    }
                } else {
                    // Only the positive branch consumes an active length, and only
                    // it can see an all-zero coefficient: a negative classification
                    // requires a nonzero guard or a set top data bit. Scanning here
                    // rather than before the branch keeps the backward sweep off
                    // the negative pass entirely, which is one full pass over the
                    // coefficient matrix per multiplication.
                    let active = SharedEval::active_len(coeff_slice);
                    if active == 0 {
                        continue;
                    }
                    // SAFETY: work.len() >= needed, acc.len() >= needed.
                    unsafe {
                        process_positive_coeff(
                            coeff_slice,
                            active,
                            shift_limbs,
                            shift_sub_bits,
                            acc,
                            work,
                        );
                    }
                }
            }
        }

        // Fold high limbs of accumulator back using 2^mod_bits = -1.
        // SAFETY: acc.len() > outer_cl or equal, work.len() >= acc.len() - ml_outer.
        unsafe {
            Self::fold_high_into_low(acc, ml_outer, outer_cl, work);
        }

        // Copy the folded canonical result to dst.
        let copy_count = outer_cl.min(dst.len());
        // SAFETY: copy_count <= acc.len() and copy_count <= dst.len().
        unsafe {
            dst.get_unchecked_mut(..copy_count)
                .copy_from_slice(acc.get_unchecked(..copy_count));
        }
    }
}

// ── Positive coefficients ─────────────────────────────────────────────────────

/// Processes a positive (non-negative-residue) coefficient: shift and add to dst.
///
/// # Safety
/// - `coeff_slice` has length >= `active_len`.
/// - `scratch.len() >= active_len + 1`.
/// - `dst.len() >= shift_limbs + active_len + 1`.
unsafe fn process_positive_coeff(
    coeff_slice: &[Limb],
    active_len: usize,
    shift_limbs: usize,
    shift_sub_bits: u32,
    dst: &mut [Limb],
    scratch: &mut [Limb],
) {
    if shift_sub_bits == 0 {
        let end = shift_limbs.wrapping_add(active_len);
        debug_assert!(
            end <= dst.len(),
            "the accumulator is sized to hold every shifted coefficient"
        );
        if end > dst.len() {
            return;
        }
        // SAFETY: end <= dst.len() and active_len <= coeff_slice.len().
        let carry = Addition::add_slice_in_place(
            unsafe { dst.get_unchecked_mut(shift_limbs..end) },
            unsafe { coeff_slice.get_unchecked(..active_len) },
        );
        if carry != 0 {
            // SAFETY: end <= dst.len().
            let _ = SsaCarry::propagate_carry(unsafe { dst.get_unchecked_mut(end..) });
        }
        return;
    }

    let work_len = active_len.wrapping_add(1);
    let destination_end = shift_limbs.wrapping_add(work_len);

    debug_assert!(
        work_len <= scratch.len() && destination_end <= dst.len(),
        "reconstruction scratch and accumulator are sized from the same plan"
    );
    if work_len > scratch.len() || destination_end > dst.len() {
        return;
    }

    // Shift only the coefficient-local span. Its output is then added at the
    // final limb offset directly; zeroing and revisiting the `shift_limbs`
    // prefix would contribute nothing and made reconstruction work grow with
    // the coefficient's position rather than its width.
    // SAFETY: work_len <= scratch.len() and active_len < work_len.
    let work = unsafe { scratch.get_unchecked_mut(..work_len) };
    // SAFETY: active_len <= coeff_slice.len() and active_len < work.len().
    unsafe { work.get_unchecked_mut(..active_len) }
        .copy_from_slice(unsafe { coeff_slice.get_unchecked(..active_len) });
    // SAFETY: the complete active coefficient prefix is initialized and the
    // architecture shift supports exact in-place spans.
    let shifted_carry =
        unsafe { ArchKernels::lshift_unchecked(work.as_mut_ptr(), active_len, shift_sub_bits) };
    // SAFETY: active_len < work_len.
    unsafe {
        *work.get_unchecked_mut(active_len) = shifted_carry;
    }
    let apply_len = active_len.wrapping_add(usize::from(shifted_carry != 0));
    let apply_end = shift_limbs.wrapping_add(apply_len);
    // SAFETY: apply_end <= destination_end <= dst.len(), and apply_len <= work.len().
    let carry = Addition::add_slice_in_place(
        unsafe { dst.get_unchecked_mut(shift_limbs..apply_end) },
        unsafe { work.get_unchecked(..apply_len) },
    );
    if carry != 0 {
        let mut idx = apply_end;
        let mut pending = carry;
        while pending != 0 && idx < dst.len() {
            // SAFETY: idx < dst.len(), guaranteed by the loop condition.
            let limb = unsafe { dst.get_unchecked_mut(idx) };
            let (sum, overflow) = limb.overflowing_add(pending);
            pending = Limb::from(overflow);
            *limb = sum;
            idx = idx.wrapping_add(1);
        }
    }
}

// ── Negative-residue coefficients ─────────────────────────────────────────────

/// Processes a negative (negative-residue) coefficient: compute magnitude, shift,
/// subtract from dst.
///
/// # Safety
/// - `coeff_slice` has at least `cl` limbs.
/// - `scratch.len() >= cl + 1`.
/// - `dst.len() >= outer_cl`.
#[allow(
    clippy::too_many_arguments,
    reason = "all parameters are needed for Fermat-ring coefficient reconstruction"
)]
unsafe fn process_negative_coeff(
    coeff_slice: &[Limb],
    shift_limbs: usize,
    shift_sub_bits: u32,
    cl: usize,
    ml_inner: usize,
    dst: &mut [Limb],
    scratch: &mut [Limb],
) {
    // SAFETY: caller guarantees scratch.len() >= cl.
    let mag_slice = unsafe { scratch.get_unchecked_mut(..cl) };
    mag_slice.fill(0);

    // SAFETY: 0 < cl and ml_inner < cl.
    unsafe {
        *mag_slice.get_unchecked_mut(0) = 1;
        *mag_slice.get_unchecked_mut(ml_inner) = 1;
    }

    // SAFETY: caller guarantees both slices have at least cl limbs.
    let mag_borrow =
        Addition::sub_slice_in_place(unsafe { mag_slice.get_unchecked_mut(..cl) }, unsafe {
            coeff_slice.get_unchecked(..cl)
        });
    debug_assert_eq!(mag_borrow, 0, "modulus >= canonical residue");

    let mag_active = mag_slice
        .iter()
        .rposition(|l| *l != 0)
        .map_or(0, |pos| pos.wrapping_add(1));

    if mag_active == 0 {
        return;
    }

    if shift_sub_bits == 0 {
        let end = shift_limbs.wrapping_add(mag_active);
        debug_assert!(
            end <= dst.len(),
            "the accumulator is sized to hold every shifted magnitude"
        );
        if end > dst.len() {
            return;
        }
        // SAFETY: end <= dst.len() and mag_active <= cl <= mag_slice.len().
        let borrow = Addition::sub_slice_in_place(
            unsafe { dst.get_unchecked_mut(shift_limbs..end) },
            unsafe { mag_slice.get_unchecked(..mag_active) },
        );
        if borrow != 0 {
            // SAFETY: end <= dst.len().
            let _ = SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(end..) });
        }
        return;
    }

    let mag_needed = mag_active.wrapping_add(1);
    let destination_end = shift_limbs.wrapping_add(mag_needed);
    debug_assert!(
        mag_needed <= scratch.len() && destination_end <= dst.len(),
        "reconstruction buffers are sized to hold every shifted magnitude"
    );
    if mag_needed > scratch.len() || destination_end > dst.len() {
        return;
    }

    // Keep the magnitude at scratch offset zero and shift it in place. Applying
    // it at `dst[shift_limbs..]` is algebraically the same as moving it right
    // by `shift_limbs` zero limbs first, without clearing or subtracting that
    // position-dependent prefix.
    // SAFETY: mag_needed <= scratch.len() and mag_active < mag_needed.
    let mag_work = unsafe { scratch.get_unchecked_mut(..mag_needed) };
    // SAFETY: the magnitude occupies its initialized active prefix and the
    // architecture shift supports exact in-place spans.
    let shifted_carry =
        unsafe { ArchKernels::lshift_unchecked(mag_work.as_mut_ptr(), mag_active, shift_sub_bits) };
    // SAFETY: mag_active < mag_needed.
    unsafe {
        *mag_work.get_unchecked_mut(mag_active) = shifted_carry;
    }
    let sub_len = mag_active.wrapping_add(usize::from(shifted_carry != 0));
    let sub_end = shift_limbs.wrapping_add(sub_len);
    // SAFETY: sub_end <= destination_end <= dst.len() and sub_len <= mag_work.len().
    let sub_borrow = Addition::sub_slice_in_place(
        unsafe { dst.get_unchecked_mut(shift_limbs..sub_end) },
        unsafe { mag_work.get_unchecked(..sub_len) },
    );
    if sub_borrow != 0 {
        let mut idx = sub_end;
        let mut pending = sub_borrow;
        while pending != 0 && idx < dst.len() {
            // SAFETY: idx < dst.len(), guaranteed by the loop condition.
            let (difference, underflow) =
                unsafe { dst.get_unchecked_mut(idx) }.overflowing_sub(pending);
            pending = Limb::from(underflow);
            // SAFETY: idx < dst.len().
            unsafe {
                *dst.get_unchecked_mut(idx) = difference;
            }
            idx = idx.wrapping_add(1);
        }
    }
}
