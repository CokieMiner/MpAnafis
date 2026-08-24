//! Operand-to-coefficient splitting, including the fused whole-bit pre-twist.
//!
//! Declares the [`SsaCoefficients`] namespace, because splitting is where an operand
//! first becomes a coefficient matrix; [`accumulate`](super::accumulate) contributes
//! the reverse direction to the same namespace.

use super::{Addition, LIMB_BITS, Limb, SsaCarry, SsaRing, SsaTransform};

/// Namespace for the transform's coefficient matrix: cutting an operand into it,
/// and accumulating a product back out of it.
///
/// The two directions are exact inverses either side of the transform, so the
/// chunk geometry one assumes is the one the other undoes. Keeping them on one
/// namespace is what makes that pairing visible at the call sites.
pub struct SsaCoefficients;

/// Values derived once from a transform's coefficient layout.
#[derive(Clone, Copy)]
struct SplitLayout {
    cl: usize,
    chunk_bits: usize,
    copy_count: usize,
    mask_index: usize,
    needs_mask: bool,
    mask: Limb,
}

impl SplitLayout {
    fn new(chunk_bits: usize, inner_bits: usize) -> Self {
        let cl = SsaRing::coeff_limbs(inner_bits);
        let chunk_limbs = chunk_bits.wrapping_div(LIMB_BITS);
        let chunk_remainder = chunk_bits.wrapping_rem(LIMB_BITS);
        let has_partial_limb = chunk_remainder != 0;
        let result_limbs = chunk_limbs.wrapping_add(usize::from(has_partial_limb));
        let copy_count = result_limbs.min(cl);
        let mask_index = result_limbs.wrapping_sub(1);
        let needs_mask = has_partial_limb && result_limbs > 0 && mask_index < cl;
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS - chunk_remainder is below 64 and always fits u32"
        )]
        let mask = Limb::MAX.wrapping_shr(LIMB_BITS.wrapping_sub(chunk_remainder) as u32);

        Self {
            cl,
            chunk_bits,
            copy_count,
            mask_index,
            needs_mask,
            mask,
        }
    }
}

impl SsaCoefficients {
    /// Splits an operand and applies the pre-twist in the same sweep.
    ///
    /// The ordinary path first writes every narrow chunk into a zeroed matrix and
    /// then reads and rewrites the whole matrix to apply its twist. Each source
    /// chunk is instead staged in one cache-hot coefficient and shifted directly
    /// into its final slot. The twist exponent is tracked in half-bit units
    /// modulo `4n`, so an odd step applies its `sqrt(2)` factor per chunk
    /// through [`SsaRing::mul_sqrt2`] and no geometry needs a separate
    /// whole-matrix twist pass.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint buffer of at least two complete coefficients.
    pub unsafe fn split_twisted(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
        twist_step_half: usize,
        scratch: &mut [Limb],
    ) {
        let layout = SplitLayout::new(chunk_bits, inner_bits);
        debug_assert!(
            scratch.len() >= layout.cl.saturating_mul(2),
            "split scratch must hold the staging coefficient and the sqrt(2) factor"
        );
        // Half-bit exponents live modulo 4n: sqrt(2) has order 4n because its
        // square, 2, has order 2n. This is the same domain the inverse twist
        // correction accumulates in.
        let half_period = inner_bits.wrapping_mul(4);
        let mut shift = 0_usize;

        let src_bits = src.len().saturating_mul(LIMB_BITS);
        let active_chunks = if layout.chunk_bits == 0 {
            0
        } else {
            src_bits.div_ceil(layout.chunk_bits).min(transform_len)
        };

        for index in 0..active_chunks {
            // The staging borrow covers the first scratch coefficient and ends
            // at the copy below, before the sqrt(2) factor reclaims the arena.
            // SAFETY: the contract guarantees two complete scratch coefficients.
            let stage = unsafe { scratch.get_unchecked_mut(..layout.cl) };
            // SAFETY: stage is one complete coefficient; extract_chunk fully
            // defines it without requiring a pre-zeroed buffer.
            unsafe {
                extract_chunk(src, stage, index, layout);
            }
            // SAFETY: the matrix contains transform_len complete slots.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, layout.cl) };
            let whole_shift = shift.wrapping_shr(1);
            if whole_shift == 0 {
                slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(slot, stage, whole_shift, inner_bits);
                }
            }
            if !shift.is_multiple_of(2) {
                // SAFETY: slot is canonical after the whole-bit shift, and the
                // full scratch holds the two-coefficient arena the factor needs.
                unsafe {
                    SsaRing::mul_sqrt2(slot, inner_bits, scratch);
                }
            }
            shift = SsaRing::reduce_mod_period(shift.wrapping_add(twist_step_half), half_period);
        }

        let active_limbs = active_chunks.wrapping_mul(layout.cl);
        if active_limbs < matrix.len() {
            // SAFETY: active_limbs <= matrix.len() by construction.
            unsafe {
                matrix.get_unchecked_mut(active_limbs..).fill(0);
            }
        }
    }

    /// Splits an operand, applies the pre-twist, and computes the first DIF
    /// butterfly stage across both matrix halves in a single pass.
    ///
    /// When the active operand digits occupy at most the lower half of the transform
    /// length (`transform_len / 2`), the upper matrix half starts at zero. The first
    /// DIF butterfly stage is therefore `(low, high) = (low, low * w^j)`.
    ///
    /// This method extracts chunk `j`, computes `low = chunk * theta^j`, and computes
    /// `high = low * w^j = chunk * (theta^j * w^j)`, writing directly to both halves
    /// of the matrix in one streaming pass. This avoids writing zeros to the high half
    /// and eliminates a complete DRAM read-and-rewrite pass of the matrix. Twist
    /// exponents are tracked in half-bit units modulo `4n`, so odd steps fold their
    /// `sqrt(2)` factor into the same streaming pass.
    ///
    /// Returns `false` only when `transform_len < 2` or `matrix` is undersized.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint buffer of at least two complete coefficients.
    #[allow(
        clippy::too_many_arguments,
        reason = "Internal FFT staging requires explicit operand, matrix, geometry, and scratch buffers"
    )]
    pub unsafe fn split_twisted_and_stage1_dif(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
        twist_step_half: usize,
        omega_shift: usize,
        scratch: &mut [Limb],
    ) -> bool {
        if transform_len < 2 {
            return false;
        }

        let layout = SplitLayout::new(chunk_bits, inner_bits);
        debug_assert!(
            scratch.len() >= layout.cl.saturating_mul(2),
            "split scratch must hold the staging coefficient and the sqrt(2) factor"
        );
        // Half-bit twist exponents live modulo 4n; the whole-bit transform root
        // omega contributes two half-bit units per step.
        let half_period = inner_bits.wrapping_mul(4);
        let whole_period = inner_bits.wrapping_mul(2);
        let half_len = transform_len >> 1;
        let half_matrix_len = half_len.wrapping_mul(layout.cl);

        if matrix.len() < transform_len.wrapping_mul(layout.cl) {
            return false;
        }

        let (low_matrix, high_matrix) = matrix.split_at_mut(half_matrix_len);

        let mut low_shift = 0_usize;
        let mut twiddle_shift = 0_usize;

        let src_bits = src.len().saturating_mul(LIMB_BITS);
        let active_chunks = if layout.chunk_bits == 0 {
            0
        } else {
            src_bits.div_ceil(layout.chunk_bits).min(half_len)
        };

        for (index, (low_slot, high_slot)) in low_matrix
            .chunks_exact_mut(layout.cl)
            .zip(high_matrix.chunks_exact_mut(layout.cl))
            .take(active_chunks)
            .enumerate()
        {
            // The staging borrow covers the first scratch coefficient and ends
            // at the two copies below, before the sqrt(2) factors reclaim the
            // arena.
            // SAFETY: the contract guarantees two complete scratch coefficients.
            let stage = unsafe { scratch.get_unchecked_mut(..layout.cl) };
            // SAFETY: stage is one complete coefficient; extract_chunk fully
            // defines it without requiring a pre-zeroed buffer.
            unsafe {
                extract_chunk(src, stage, index, layout);
            }

            // low = chunk * theta^index.
            let low_whole = low_shift.wrapping_shr(1);
            if low_whole == 0 {
                low_slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and low_slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(low_slot, stage, low_whole, inner_bits);
                }
            }

            // high = chunk * theta^index * omega^index; omega is a whole-bit
            // shift, so it adds two half-bit units per step.
            let high_half = SsaRing::reduce_mod_period(
                low_shift.wrapping_add(twiddle_shift.wrapping_mul(2)),
                half_period,
            );
            let high_whole = high_half.wrapping_shr(1);
            if high_whole == 0 {
                high_slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and high_slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(high_slot, stage, high_whole, inner_bits);
                }
            }

            if !low_shift.is_multiple_of(2) {
                // SAFETY: low_slot is canonical after the whole-bit shift, and
                // the full scratch holds the two-coefficient arena.
                unsafe {
                    SsaRing::mul_sqrt2(low_slot, inner_bits, scratch);
                }
            }
            if !high_half.is_multiple_of(2) {
                // SAFETY: high_slot is canonical after the whole-bit shift, and
                // the full scratch holds the two-coefficient arena.
                unsafe {
                    SsaRing::mul_sqrt2(high_slot, inner_bits, scratch);
                }
            }

            low_shift =
                SsaRing::reduce_mod_period(low_shift.wrapping_add(twist_step_half), half_period);
            twiddle_shift =
                SsaRing::reduce_mod_period(twiddle_shift.wrapping_add(omega_shift), whole_period);
        }

        let active_limbs = active_chunks.wrapping_mul(layout.cl);
        if active_limbs < half_matrix_len {
            // SAFETY: active_limbs <= half_matrix_len by construction.
            unsafe {
                low_matrix.get_unchecked_mut(active_limbs..).fill(0);
                high_matrix.get_unchecked_mut(active_limbs..).fill(0);
            }
        }
        true
    }

    /// Folds the high accumulator limbs into the low half using `2^mod_bits = -1`.
    ///
    /// # Safety
    /// - `dst.len() >= outer_cl`.
    /// - `scratch.len() >= dst.len() - ml_outer`.
    pub unsafe fn fold_high_into_low(
        dst: &mut [Limb],
        ml_outer: usize,
        outer_cl: usize,
        scratch: &mut [Limb],
    ) {
        if dst.len() <= outer_cl {
            return;
        }

        let high_start = ml_outer;
        let high_end = dst.len();
        let high_len = high_end.wrapping_sub(high_start);
        if high_len == 0 {
            return;
        }

        // SAFETY: high_len <= scratch.len(), guaranteed by caller.
        let high_copy = unsafe { scratch.get_unchecked_mut(..high_len) };
        // SAFETY: high_start < high_end <= dst.len(), guaranteed by caller.
        high_copy.copy_from_slice(unsafe { dst.get_unchecked(high_start..high_end) });
        // SAFETY: high_start < high_end <= dst.len().
        unsafe { dst.get_unchecked_mut(high_start..high_end) }.fill(0);

        let sub_count = high_len.min(ml_outer);
        // SAFETY: sub_count <= ml_outer <= outer_cl <= dst.len().
        let low = unsafe { dst.get_unchecked_mut(..sub_count) };
        // SAFETY: sub_count <= high_len == high_copy.len().
        let high = unsafe { high_copy.get_unchecked(..sub_count) };
        let borrow = Addition::sub_slice_in_place(low, high);

        // SAFETY: sub_count <= ml_outer <= dst.len().
        let final_borrow = borrow != 0
            && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(sub_count..ml_outer) });

        if final_borrow {
            // SAFETY: ml_outer < outer_cl <= dst.len().
            unsafe {
                SsaCarry::correct_wrapped_shift_difference(dst, ml_outer);
            }
        }
    }
}

/// Extract one source chunk into a complete coefficient.
///
/// Fully defines `slot[..layout.cl]` without requiring a pre-zeroed buffer.
///
/// # Safety
///
/// `slot.len() == layout.cl`.
unsafe fn extract_chunk(src: &[Limb], slot: &mut [Limb], index: usize, layout: SplitLayout) {
    let bit_start = index.wrapping_mul(layout.chunk_bits);
    let start_limb = bit_start.wrapping_div(LIMB_BITS);
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "bit_start modulo LIMB_BITS is at most 63 and fits u32"
    )]
    let start_bit = bit_start.wrapping_rem(LIMB_BITS) as u32;

    if start_bit == 0 {
        if start_limb < src.len() {
            let available = src.len().wrapping_sub(start_limb).min(layout.copy_count);
            // SAFETY: the calculated source and destination prefixes are in bounds.
            unsafe {
                slot.get_unchecked_mut(..available).copy_from_slice(
                    src.get_unchecked(start_limb..start_limb.wrapping_add(available)),
                );
            }
            if available < layout.copy_count {
                // SAFETY: available < copy_count <= layout.cl == slot.len().
                unsafe {
                    slot.get_unchecked_mut(available..layout.copy_count).fill(0);
                }
            }
        } else {
            // SAFETY: copy_count <= layout.cl == slot.len().
            unsafe {
                slot.get_unchecked_mut(..layout.copy_count).fill(0);
            }
        }
    } else {
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64 and fits u32"
        )]
        let shift_up = (LIMB_BITS as u32).wrapping_sub(start_bit);
        // Only the final chunk can overhang the source, so the readable count
        // splits the sweep into a fully unchecked paired body, at most one
        // low-only limb at the overhang, and a zero tail.
        let available = src.len().saturating_sub(start_limb);
        let paired = available.saturating_sub(1).min(layout.copy_count);
        for offset in 0..paired {
            let source_index = start_limb.wrapping_add(offset);
            // SAFETY: offset < paired <= available - 1 proves both source
            // limbs lie below src.len(), and offset < copy_count <= slot.len().
            unsafe {
                let low = *src.get_unchecked(source_index);
                let high = *src.get_unchecked(source_index.wrapping_add(1));
                *slot.get_unchecked_mut(offset) =
                    low.wrapping_shr(start_bit) | high.wrapping_shl(shift_up);
            }
        }
        let mut written = paired;
        if written < layout.copy_count && written < available {
            // The last readable limb has no limb above it, so its high bits
            // are zero.
            // SAFETY: written < available proves the index is below src.len(),
            // and written < copy_count <= slot.len().
            unsafe {
                let low = *src.get_unchecked(start_limb.wrapping_add(written));
                *slot.get_unchecked_mut(written) = low.wrapping_shr(start_bit);
            }
            written = written.wrapping_add(1);
        }
        // SAFETY: written <= copy_count <= layout.cl == slot.len().
        unsafe {
            slot.get_unchecked_mut(written..layout.copy_count).fill(0);
        }
    }

    if layout.copy_count < layout.cl {
        // SAFETY: copy_count < layout.cl == slot.len().
        unsafe {
            slot.get_unchecked_mut(layout.copy_count..).fill(0);
        }
    }

    if layout.needs_mask {
        // SAFETY: needs_mask proves mask_index < layout.cl == slot.len().
        unsafe {
            *slot.get_unchecked_mut(layout.mask_index) &= layout.mask;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/tiers/split.rs"]
mod tests;
