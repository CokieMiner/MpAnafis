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
//! The sweep accumulates signed coefficients in index order, then folds:
//!
//! - `process_positive_coeff`: coefficients that keep their sign and are added.
//! - `shift_sub_magnitude_run`: coefficients read as negative residues, whose
//!   magnitude is recovered against the modulus and subtracted.
//! - `SsaCoefficients::fold_high_into_low`: the closing reduction of the accumulator by
//!   `2^n = -1`.

#![expect(
    unsafe_code,
    reason = "Direct limb-level accumulation for zero-allocation FFT reconstruct"
)]

use core::{num::NonZeroUsize, ptr::copy_nonoverlapping};

use crate::parallel::ParallelExecutor;

use super::{
    LIMB_BITS, Limb, LimbOutput, ReconstructionBlocks, SharedEval, SsaCoefficients, SsaRing,
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
    /// - `scratch`: accumulator plus parallel block storage when selected
    /// - `twist`: the inverse correction and its staging arena
    /// - `executor`: forks the untwist sweep
    ///
    /// # Safety
    /// The dimensions and scratch are derived from one validated `FftPlan`.
    /// The matrix contains a complete-coefficient prefix of the inverse negacyclic convolution of
    /// ordinary operands below `2^mod_bits`, with the supplied inverse twist
    /// still pending, or canonical exact coefficients when `twist` is absent.
    /// Every omitted coefficient is proved zero; no uncomputed inverse outputs
    /// appear in the supplied prefix. Their magnitude bound
    /// `2*chunk_bits + log2(transform_len)` is strictly
    /// below `inner_bits`. The destination holds a complete outer coefficient
    /// or an exact-product prefix whose omitted high limbs are proved zero.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the sweep carries the twist, accumulator, work area, and executor through whole-product reconstruction"
    )]
    pub unsafe fn reconstruct<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: NonZeroUsize,
        inner_bits: usize,
        mod_bits: usize,
        dst: &mut [impl LimbOutput],
        scratch: &mut [Limb],
        twist: Option<(InverseTwist, &mut [Limb])>,
        executor: &E,
    ) {
        let cl = SsaRing::coeff_limbs(inner_bits).get();
        let coefficient_count = matrix.len().div_euclid(cl);
        debug_assert!(
            matrix.len().is_multiple_of(cl) && coefficient_count <= transform_len,
            "reconstruction consumes only complete established inverse coefficients"
        );
        let ml_inner = SsaRing::mod_limbs(inner_bits);
        let ml_outer = SsaRing::mod_limbs(mod_bits);
        // SAFETY: both rings have representable 4*bits. The established prefix
        // has count<=K and K*chunk_bits=mod_bits; the planner checked the outer
        // accumulator plus a complete inner coefficient and carry limb.
        let (outer_cl, max_limbs_contrib) = unsafe {
            let shifted_limbs = coefficient_count
                .saturating_sub(1)
                .unchecked_mul(chunk_bits.get())
                .div_euclid(LIMB_BITS);
            (
                ml_outer.unchecked_add(1),
                cl.unchecked_add(shifted_limbs).unchecked_add(1),
            )
        };

        // Parallel blocks need private accumulators beyond the outer one.
        // Serial reconstruction constructs magnitude digits without workspace.
        let blocks = ReconstructionBlocks::new(
            transform_len,
            coefficient_count,
            chunk_bits,
            inner_bits,
            executor.parallelism().get(),
        );
        // Each block contributes unsigned digits through outer+bound, followed
        // by a signed overlap carry at that position. Retain those temporary
        // high digits until the complete block prefix has been resolved.
        let acc_limbs = if blocks.is_some() {
            // SAFETY: the caller's checked reconstruction arena reserves the
            // outer coefficient, inner coefficient and one block carry limb.
            unsafe { outer_cl.unchecked_add(cl).unchecked_add(1) }
        } else {
            max_limbs_contrib.max(outer_cl)
        };
        // SAFETY: caller guarantees scratch.len() >= acc_limbs.
        let (acc, work) = unsafe { scratch.split_at_mut_unchecked(acc_limbs) };
        acc.fill(0);
        // Let q = 2^mod_bits. Splitting the polynomial product at degree K
        // gives P + q*H, with P >= 0 and 0 <= H < q because the ordinary
        // operand product is below q^2. P includes unpropagated carries.
        // Negacyclic coefficients subtract the wrapped high contributions.
        // Their total negative magnitude is at most H: cancellation against
        // positive contributions can only reduce it. Thus every index-order
        // prefix is >= -H > -q, independently of the order of its signs.
        // Bias q+1 prevents underflow at every prefix and folds to zero.
        // For radix R = 2^chunk_bits, the positive contributions are at most
        // sum((i+1)*(R-1)^2*R^i, i=0..K-1) = (K*(R-1)-1)*q+1.
        // Thus every biased prefix is <= K*(R-1)*q+2 < q^2: K >= 2 and
        // R >= 2 give R^K >= 1+K*(R-1)+(R-1)^2. The final fold therefore
        // has no nonzero limbs above the outer radix squared.
        // SAFETY: acc_limbs >= outer_cl = ml_outer + 1.
        unsafe {
            *acc.get_unchecked_mut(0) = 1;
            *acc.get_unchecked_mut(ml_outer) = 1;
        }

        #[expect(
            clippy::as_conversions,
            reason = "a usize trailing-zero count is at most Limb::BITS and fits every matching usize"
        )]
        let transform_log = transform_len.trailing_zeros() as usize;
        // SAFETY: geometry construction checked 2*chunk_bits+log(K)<inner_bits.
        let coefficient_bound_bits = unsafe {
            chunk_bits
                .get()
                .unchecked_mul(2)
                .unchecked_add(transform_log)
        };
        debug_assert!(
            coefficient_bound_bits < inner_bits,
            "the centered coefficient bound must leave a sign-separation bit"
        );
        // SAFETY: chunk_bits>=1 makes 2*chunk_bits+log(K)>=2. Its positive
        // ceiling limb count is bounded by the already admitted inner ring.
        let magnitude_limbs =
            unsafe { NonZeroUsize::new_unchecked(coefficient_bound_bits.div_ceil(LIMB_BITS)) };

        let mut fused_twist = None;
        if let Some((stage_twist, stage_scratch)) = twist {
            // SAFETY: inner_bits*4 fits and cl=inner_bits/LIMB_BITS+1 with
            // LIMB_BITS>=16, so four coefficients also fit usize.
            let needed = unsafe { cl.unchecked_mul(4) };
            let parallel_scratch = SsaTransform::has_parallel_work(
                coefficient_count,
                needed,
                executor.parallelism().get(),
            ) && stage_scratch.len() >= needed;
            if parallel_scratch {
                // SAFETY: the matrix holds transform_len complete coefficients
                // and the staging arena holds the disjoint coefficients every
                // untwist fork needs. has_parallel_work proves at least two
                // coefficients, so the range is nonempty.
                unsafe {
                    untwist_range(
                        matrix,
                        0,
                        coefficient_count,
                        &stage_twist,
                        stage_scratch,
                        executor,
                    );
                }
            } else {
                // A sweep that cannot fork only loses the accumulator's temporal
                // locality. Retain the original first-touch fusion instead.
                fused_twist = Some((stage_twist, stage_scratch));
            }
        }
        if let Some(layout) = blocks {
            if let Some((stage_twist, stage_scratch)) = fused_twist {
                // SAFETY: the complete matrix and private twist arena satisfy
                // the same coefficient contracts as the sequential sweep.
                // Block admission proves more than one active coefficient.
                unsafe {
                    untwist_range(
                        matrix,
                        0,
                        coefficient_count,
                        &stage_twist,
                        stage_scratch,
                        executor,
                    );
                }
            }
            // SAFETY: the planner reserves the outer accumulator and complete
            // block arena. Untwisting establishes canonical bounded coefficients.
            unsafe {
                layout.run(matrix, acc, work, executor);
            }
        } else {
            let mut inverse_shift = fused_twist
                .as_ref()
                .map_or(0, |(stage_twist, _)| stage_twist.shift_for(0));
            for idx in 0..coefficient_count {
                if let Some((stage_twist, stage_scratch)) = fused_twist.as_mut() {
                    // SAFETY: idx addresses one complete coefficient and the staging
                    // arena is disjoint with at least two coefficient slots.
                    unsafe {
                        untwist_coefficient(matrix, idx, inverse_shift, stage_twist, stage_scratch);
                    }
                    // K*step = 2n and log2(K) < n give a final shift
                    // 2n-2log2(K) > 0 even after the last coefficient. The
                    // descending recurrence needs neither multiply nor reduction.
                    // A one-frequency TFT scales by one and starts at zero;
                    // its final, unused update remains zero. Every longer
                    // prefix follows the exact positive descending recurrence.
                    inverse_shift = inverse_shift.saturating_sub(stage_twist.twist_step_half);
                }
                // SAFETY: idx < transform_len, matrix has transform_len * cl limbs.
                let coeff_slice = unsafe { SsaTransform::coeff(matrix, idx, cl) };

                // Exact convolution coefficients have magnitude below
                // 2^coefficient_bound_bits, which is below 2^(inner_bits-1).
                // Therefore canonical residues with the top data bit set are
                // precisely negative coefficients; the guard-only value is -1.
                // SAFETY: 0 < ml_inner < cl <= coeff_slice.len().
                let top_data = unsafe { *coeff_slice.get_unchecked(ml_inner.unchecked_sub(1)) };
                // SAFETY: ml_inner < cl <= coeff_slice.len().
                let guard = unsafe { *coeff_slice.get_unchecked(ml_inner) };
                let is_negative = guard != 0 || top_data >> (Limb::BITS - 1) != 0;

                // SAFETY: idx<count<=K, and K*chunk_bits=mod_bits fits usize.
                let shift_bits = unsafe { idx.unchecked_mul(chunk_bits.get()) };
                let shift_limbs = shift_bits.div_euclid(LIMB_BITS);
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "shift_bits % LIMB_BITS < LIMB_BITS; fits u32"
                )]
                let shift_sub_bits = shift_bits.rem_euclid(LIMB_BITS) as u32;

                if is_negative {
                    // SAFETY: the coefficient satisfies the centered bound;
                    // the biased accumulator contains its complete shifted span.
                    unsafe {
                        Self::shift_sub_magnitude_run(
                            acc,
                            shift_limbs,
                            coeff_slice,
                            magnitude_limbs,
                            ml_inner,
                            shift_sub_bits,
                        );
                    }
                } else {
                    // SAFETY: the strict coefficient bound is below inner_bits,
                    // so magnitude_limbs <= ml_inner < coeff_slice.len().
                    let magnitude = unsafe { coeff_slice.get_unchecked(..magnitude_limbs.get()) };
                    let Some(active) = NonZeroUsize::new(SharedEval::active_len(magnitude)) else {
                        continue;
                    };
                    // SAFETY: acc covers the shifted active span and its top limb.
                    unsafe {
                        Self::process_positive_coeff(
                            coeff_slice,
                            active,
                            shift_limbs,
                            shift_sub_bits,
                            acc,
                        );
                    }
                }
            }
        }
        // Fold high limbs of accumulator back using 2^mod_bits = -1.
        // SAFETY: acc includes the outer guard; the biased prefix proof bounds
        // the complete accumulator below the square of the outer radix.
        unsafe {
            Self::fold_high_into_low(acc, ml_outer);
        }

        // The output's first writer copies the folded result; no destination
        // limb is read and no initialization pass precedes this exact copy.
        let copy_count = outer_cl.min(dst.len());
        // SAFETY: copy_count <= acc.len() and copy_count <= dst.len().
        unsafe {
            copy_nonoverlapping(acc.as_ptr(), dst.as_mut_ptr().cast(), copy_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse twist
// ---------------------------------------------------------------------------

/// Applies inverse twist and scaling to disjoint coefficient ranges.
/// The separate sweep serves parallel reconstruction; serial accumulation
/// fuses each correction with its coefficient's first read.
///
/// # Safety
/// `count > 0`; `matrix` holds `count` complete coefficients starting at absolute position
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
    let inner_cl = SsaRing::coeff_limbs(twist.inner_bits).get();
    // One fork needs the in-place shift's staging coefficient plus the sqrt(2)
    // factor's two-coefficient arena, so four coefficients are the smallest
    // arena two forks can share. The budget halves on each fork so siblings
    // share the pool rather than each assuming the full width.
    // SAFETY: the validated ring has representable 4*inner_bits, so four
    // guarded limb coefficients fit usize for both SSA pointer widths.
    let fork_width = unsafe { inner_cl.unchecked_mul(4) };
    if !SsaTransform::has_parallel_work(count, fork_width, executor.parallelism().get())
        || scratch.len() < fork_width
    {
        let mut inverse_shift = twist.shift_for(base);
        for slot in 0..count {
            // SAFETY: slot addresses a complete coefficient of this range and
            // the staging span holds the two disjoint coefficients the shift
            // and sqrt(2) factor need.
            unsafe {
                untwist_coefficient(matrix, slot, inverse_shift, twist, scratch);
            }
            // base+count <= K and K*step = 2n bound every correction below
            // 4n. Subtracting one step stays positive throughout the range.
            // A one-frequency TFT's sole exponent is zero, and its unused
            // update stays zero; longer prefixes subtract exactly throughout.
            inverse_shift = inverse_shift.saturating_sub(twist.twist_step_half);
        }
        return;
    }

    let half = count.div_euclid(2);
    // SAFETY: count exceeds the grain, so both halves are non-empty whole
    // numbers of complete coefficients.
    let (left_matrix, right_matrix) =
        unsafe { matrix.split_at_mut_unchecked(half.unchecked_mul(inner_cl)) };
    // SAFETY: the arena covers at least four coefficients, so each half retains
    // the two every leaf needs.
    // SAFETY: halving a slice's length always selects a valid partition.
    let (left_scratch, right_scratch) =
        unsafe { scratch.split_at_mut_unchecked(scratch.len() >> 1) };
    let ((), ()) = executor.join(
        // SAFETY: left_matrix and left_scratch are disjoint complete spans.
        || unsafe {
            untwist_range(left_matrix, base, half, twist, left_scratch, executor);
        },
        // SAFETY: right_matrix and right_scratch are disjoint complete spans.
        || unsafe {
            untwist_range(
                right_matrix,
                base.unchecked_add(half),
                count.unchecked_sub(half),
                twist,
                right_scratch,
                executor,
            );
        },
    );
}

/// Applies the inverse twiddle to one coefficient in place.
///
/// A slot leaving the inverse transform is only semi-normalized, which the
/// in-place shift accepts directly; the staging buffer serves only its
/// discarded-high pass and the `sqrt(2)` factor.
///
/// # Safety
/// - `matrix` holds complete `SsaRing::coeff_limbs(inner_bits)` slots and
///   `slot` addresses one of them.
/// - `total_shift < 4*inner_bits` is the inverse half-bit exponent for this slot.
/// - `scratch` is disjoint from `matrix` and holds at least two coefficients.
unsafe fn untwist_coefficient(
    matrix: &mut [Limb],
    slot: usize,
    total_shift: usize,
    twist: &InverseTwist,
    scratch: &mut [Limb],
) {
    let inner_cl = SsaRing::coeff_limbs(twist.inner_bits).get();
    // SAFETY: the caller guarantees `slot` addresses a complete slot.
    let coefficient = unsafe { SsaTransform::coeff_mut(matrix, slot, inner_cl) };
    // SAFETY: the staging span is disjoint from the matrix slot and holds
    // the coefficient-width arena the shift and sqrt(2) factor need. The
    // caller supplies a reduced half-bit exponent, so halving gives < 2n.
    unsafe {
        if total_shift.is_multiple_of(2) {
            SsaRing::shift_in_place(coefficient, total_shift >> 1, twist.inner_bits, scratch);
        } else {
            SsaRing::shift_sqrt2(coefficient, total_shift >> 1, twist.inner_bits, scratch);
        }
    }
    // A zero shift still leaves the inverse transform semi-normalized.
    // SAFETY: coefficient is a complete slot and inner_bits matches.
    unsafe {
        let _ = SsaRing::normalize(coefficient, twist.inner_bits);
    }
}

impl InverseTwist {
    /// Total inverse shift for one coefficient, in half-bit units.
    ///
    /// Folds the inverse twiddle together with the `1 / transform_len` scaling,
    /// reduced modulo the ring period. Every term of the whole-bit derivation is
    /// doubled so an odd result denotes a remaining `sqrt(2)` factor.
    #[must_use]
    const fn shift_for(&self, index: usize) -> usize {
        // theta^(-index) / K has half-bit exponent -(index*step + 2log K).
        // index < K and K*step = 2n bound the first term below 2n; the
        // geometry's log K < n bounds the second below 2n. Their sum is
        // therefore below the representable 4n period without a remainder.
        // SAFETY: geometry construction checks 4n; index<K and K*step=2n
        // bound index*step<2n, while 0<=log(inverse_len)<n bounds its scale<2n.
        let (full_period, correction) = unsafe {
            (
                self.inner_bits.unchecked_mul(4),
                index
                    .unchecked_mul(self.twist_step_half)
                    .unchecked_add(self.transform_log.unchecked_mul(2)),
            )
        };
        debug_assert!(
            correction < full_period,
            "the inverse exponent has one-period magnitude"
        );
        if correction == 0 {
            // A one-frequency TFT has inverse_len=1 and index=0.
            0
        } else {
            // SAFETY: 0<correction<full_period gives the exact reduced exponent.
            unsafe { full_period.unchecked_sub(correction) }
        }
    }
}
