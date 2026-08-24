//! Coefficient-matrix reconstruction into a destination product.
//!
//! `SsaCoefficients::split_twisted` cuts an operand into radix-`2^chunk_bits`
//! coefficients;
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

use crate::parallel::ParallelExecutor;

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

impl SsaCoefficients {
    /// Accumulates inverse-transformed coefficients from `matrix` into the
    /// destination product buffer `dst`.
    ///
    /// After the inverse FFT, each coefficient `c[i]` contributes
    /// `c[i] * B^(i * chunk_bits)` to the product, where `B = 2` and the
    /// contribution is reduced modulo `2^mod_bits + 1`. The inverse twiddle and
    /// scaling use a separate parallel sweep when there is enough work and
    /// staging scratch to fork. Otherwise each correction is fused into the
    /// serial accumulation so the coefficient stays hot in cache.
    ///
    /// Coefficients that exceed the correction threshold are treated as negative
    /// residues (subtracted instead of added).
    ///
    /// # Arguments
    /// - `matrix`: flat coefficient buffer (post-IFFT)
    /// - `transform_len`: number of coefficient slots
    /// - `chunk_bits`: radix chunk width
    /// - `inner_bits`: Fermat ring modulus bit width
    /// - `mod_bits`: outer modulus bit width for the final product
    /// - `dst`: destination product buffer
    /// - `scratch`: temporary buffer; must be large enough for both accumulator
    ///   (see internal computation) and work area (>= max shifted coeff width)
    /// - `twist`: the inverse correction and its staging arena
    /// - `executor`: forks the untwist sweep
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the sweep carries the twist, accumulator, work area, and executor through whole-product reconstruction"
    )]
    pub fn reconstruct<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
        mod_bits: usize,
        dst: &mut [Limb],
        scratch: &mut [Limb],
        twist: Option<(InverseTwist, &mut [Limb])>,
        executor: &E,
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

        let mut fused_twist = None;
        if let Some((stage_twist, stage_scratch)) = twist {
            let parallel_scratch = cl.checked_mul(4).is_some_and(|needed| {
                SsaTransform::has_parallel_work(transform_len, needed, executor)
                    && stage_scratch.len() >= needed
            });
            if parallel_scratch {
                // SAFETY: the matrix holds transform_len complete coefficients
                // and the staging arena holds the disjoint coefficients every
                // untwist fork needs.
                unsafe {
                    untwist_all(matrix, transform_len, &stage_twist, stage_scratch, executor);
                }
            } else {
                // A sweep that cannot fork only loses the accumulator's temporal
                // locality. Retain the original first-touch fusion instead.
                fused_twist = Some((stage_twist, stage_scratch));
            }
        }
        for idx in 0..transform_len {
            if let Some((stage_twist, stage_scratch)) = fused_twist.as_mut() {
                // SAFETY: idx addresses one complete coefficient and the staging
                // arena is disjoint with at least two coefficient slots.
                unsafe {
                    untwist_coefficient(matrix, idx, idx, stage_twist, stage_scratch);
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

// ── Inverse twist ──────────────────────────────────────────────────────────────

/// Applies the inverse twiddle to one coefficient in place.
///
/// A slot leaving the inverse transform is only semi-normalized, which the
/// in-place shift accepts directly; the staging buffer serves only its
/// discarded-high pass and the `sqrt(2)` factor.
///
/// # Safety
/// - `matrix` holds complete `SsaRing::coeff_limbs(inner_bits)` slots and
///   `slot` addresses one of them.
/// - `index` is the coefficient's absolute transform position, from which the
///   twist derives its shift.
/// - `scratch` is disjoint from `matrix` and holds at least two coefficients.
unsafe fn untwist_coefficient(
    matrix: &mut [Limb],
    slot: usize,
    index: usize,
    twist: &InverseTwist,
    scratch: &mut [Limb],
) {
    let inner_cl = SsaRing::coeff_limbs(twist.inner_bits);
    let total_shift = twist.shift_for(index);
    // SAFETY: the caller guarantees `slot` addresses a complete slot.
    let coefficient = unsafe { SsaTransform::coeff_mut(matrix, slot, inner_cl) };
    if total_shift != 0 {
        // SAFETY: the staging span is disjoint from the matrix slot and holds
        // the coefficient-width arena the shift and sqrt(2) factor need.
        unsafe {
            SsaRing::shift_in_place(
                coefficient,
                total_shift.wrapping_shr(1),
                twist.inner_bits,
                scratch,
            );
            if !total_shift.is_multiple_of(2) {
                // SAFETY: the whole two-coefficient buffer is free again here.
                SsaRing::mul_sqrt2(coefficient, twist.inner_bits, scratch);
            }
        }
    }
    // A zero shift still leaves the inverse transform semi-normalized.
    // SAFETY: coefficient is a complete slot and inner_bits matches.
    unsafe {
        SsaRing::normalize(coefficient, twist.inner_bits);
    }
}

/// Applies the inverse twist and scaling to every coefficient, forking over
/// coefficient ranges when the executor and the staging arena allow it.
///
/// The untwist normally rides on the accumulation sweep's first touch to retain
/// coefficient locality. This separate sweep is selected only when it can
/// actually fork: every coefficient's shift is independent, and the existing
/// transform-sized twiddle arena feeds one staging pair per child range.
///
/// # Safety
/// `matrix` holds `transform_len` complete coefficients for `twist.inner_bits`,
/// and `scratch` is disjoint from it with at least two complete coefficients.
unsafe fn untwist_all<E: ParallelExecutor>(
    matrix: &mut [Limb],
    transform_len: usize,
    twist: &InverseTwist,
    scratch: &mut [Limb],
    executor: &E,
) {
    if transform_len == 0 {
        return;
    }
    // SAFETY: transform_len is nonzero and the arena satisfies the two
    // coefficients every leaf needs.
    unsafe {
        untwist_range(matrix, 0, transform_len, twist, scratch, executor);
    }
}

/// Recursive coefficient-range fork of [`untwist_all`].
///
/// # Safety
/// `matrix` holds `count` complete coefficients starting at absolute position
/// `base`, and `scratch` holds at least two complete coefficients.
unsafe fn untwist_range<E: ParallelExecutor>(
    matrix: &mut [Limb],
    base: usize,
    count: usize,
    twist: &InverseTwist,
    scratch: &mut [Limb],
    executor: &E,
) {
    debug_assert!(count > 0, "an untwist range is never empty");
    let inner_cl = SsaRing::coeff_limbs(twist.inner_bits);
    // One fork needs the in-place shift's staging coefficient plus the sqrt(2)
    // factor's two-coefficient arena, so four coefficients are the smallest
    // arena two forks can share.
    if !SsaTransform::has_parallel_work(count, inner_cl.wrapping_mul(4), executor)
        || scratch.len() < inner_cl.wrapping_mul(4)
    {
        for slot in 0..count {
            // SAFETY: slot addresses a complete coefficient of this range and
            // the staging span holds the two disjoint coefficients the shift
            // and sqrt(2) factor need.
            unsafe {
                untwist_coefficient(matrix, slot, base.wrapping_add(slot), twist, scratch);
            }
        }
        return;
    }

    let half = count.div_euclid(2);
    // SAFETY: count exceeds the grain, so both halves are non-empty whole
    // numbers of complete coefficients.
    let (left_matrix, right_matrix) =
        unsafe { matrix.split_at_mut_unchecked(half.wrapping_mul(inner_cl)) };
    // SAFETY: the arena covers at least four coefficients, so each half retains
    // the two every leaf needs.
    let (left_scratch, right_scratch) = scratch.split_at_mut(scratch.len().div_euclid(2));
    let ((), ()) = executor.join(
        // SAFETY: left_matrix and left_scratch are disjoint complete spans.
        || unsafe {
            untwist_range(left_matrix, base, half, twist, left_scratch, executor);
        },
        // SAFETY: right_matrix and right_scratch are disjoint complete spans.
        || unsafe {
            untwist_range(
                right_matrix,
                base.wrapping_add(half),
                count.wrapping_sub(half),
                twist,
                right_scratch,
                executor,
            );
        },
    );
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
        // SAFETY: apply_end <= destination_end <= dst.len().
        let _ = SsaCarry::propagate_carry(unsafe { dst.get_unchecked_mut(apply_end..) });
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
    // The magnitude is `(2^inner_bits + 1) - coefficient`. A canonical
    // coefficient with a set guard is exactly `2^n`, whose magnitude is one;
    // every other negative residue has its top data bit set, so the bitwise
    // complement plus two reproduces the modulus difference in one fused pass
    // with no prebuilt modulus buffer. The same pass tracks the top nonzero
    // limb, so no separate active-length scan re-reads the magnitude.
    // SAFETY: 0 < cl and ml_inner < cl.
    let mag_active = unsafe {
        if *coeff_slice.get_unchecked(ml_inner) != 0 {
            mag_slice.fill(0);
            *mag_slice.get_unchecked_mut(0) = 1;
            1
        } else {
            let mut carry = Limb::from(2_usize);
            let mut active = 0_usize;
            for index in 0..ml_inner {
                // SAFETY: index < ml_inner < cl <= coeff_slice.len() and the
                // matching mag_slice index is in range.
                let complement = !*coeff_slice.get_unchecked(index);
                let (sum, escaped) = complement.overflowing_add(carry);
                *mag_slice.get_unchecked_mut(index) = sum;
                if sum != 0 {
                    active = index.wrapping_add(1);
                }
                carry = Limb::from(escaped);
            }
            debug_assert_eq!(carry, 0, "the magnitude stays below 2^inner_bits");
            // SAFETY: ml_inner < cl <= mag_slice.len().
            *mag_slice.get_unchecked_mut(ml_inner) = 0;
            active
        }
    };
    debug_assert!(mag_active > 0, "a negative residue has a nonzero magnitude");

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
        // SAFETY: sub_end <= destination_end <= dst.len().
        let _ = SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(sub_end..) });
    }
}
