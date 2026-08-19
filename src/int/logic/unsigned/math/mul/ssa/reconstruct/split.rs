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
    /// Extracts `chunk_bits`-wide bit ranges from `src` into coefficient slots of
    /// `matrix`, zero-extending each slot to `SsaRing::coeff_limbs(inner_bits)`.
    pub fn split(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
    ) {
        let layout = SplitLayout::new(chunk_bits, inner_bits);

        // Every coefficient suffix is zero extension. One flat clear lets the
        // allocator use a wide memset rather than issuing one fill per slot.
        matrix.fill(0);
        for index in 0..transform_len {
            // SAFETY: the caller supplies transform_len complete coefficient slots.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, layout.cl) };
            // SAFETY: `slot` is one complete zeroed coefficient.
            unsafe {
                extract_chunk(src, slot, index, layout);
            }
        }
    }

    /// Splits an operand and applies an even half-bit pre-twist in the same sweep.
    ///
    /// The ordinary path first writes every narrow chunk into a zeroed matrix and
    /// then reads and rewrites the whole matrix to apply its twist. When the twist
    /// step is an even number of half bits, each source chunk is instead staged in
    /// one cache-hot coefficient and shifted directly into its final slot. This is
    /// the same decomposition identity with one RAM-sized matrix pass removed.
    ///
    /// Returns `false` for an odd half-bit step; that geometry needs a `sqrt(2)`
    /// operation and must use [`Self::split`] followed by the general twist.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint coefficient-sized buffer.
    pub unsafe fn split_twisted(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: usize,
        inner_bits: usize,
        twist_step_half: usize,
        scratch: &mut [Limb],
    ) -> bool {
        if !twist_step_half.is_multiple_of(2) {
            return false;
        }

        let layout = SplitLayout::new(chunk_bits, inner_bits);
        debug_assert!(scratch.len() >= layout.cl, "split scratch is undersized");
        let period = inner_bits.wrapping_mul(2);
        let whole_step = twist_step_half.wrapping_shr(1);
        // SAFETY: this function's contract guarantees one complete scratch coefficient.
        let stage = unsafe { scratch.get_unchecked_mut(..layout.cl) };
        let mut shift = 0_usize;

        for index in 0..transform_len {
            stage.fill(0);
            // SAFETY: stage is one complete zeroed coefficient.
            unsafe {
                extract_chunk(src, stage, index, layout);
            }
            // SAFETY: the matrix contains transform_len complete slots.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, layout.cl) };
            if shift == 0 {
                slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(slot, stage, shift, inner_bits);
                }
            }
            shift = SsaRing::reduce_mod_period(shift.wrapping_add(whole_step), period);
        }
        true
    }

    /// Splits an operand, applies an even half-bit pre-twist, and computes the
    /// first DIF butterfly stage across both matrix halves in a single pass.
    ///
    /// When the active operand digits occupy at most the lower half of the transform
    /// length (`transform_len / 2`), the upper matrix half starts at zero. The first
    /// DIF butterfly stage is therefore `(low, high) = (low, low * w^j)`.
    ///
    /// This method extracts chunk `j`, computes `low = chunk * theta^j`, and computes
    /// `high = low * w^j = chunk * (theta^j * w^j)`, writing directly to both halves
    /// of the matrix in one streaming pass. This avoids writing zeros to the high half
    /// and eliminates a complete DRAM read-and-rewrite pass of the matrix.
    ///
    /// Returns `false` when `twist_step_half` is odd, `transform_len < 2`, or `matrix`
    /// is undersized.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint coefficient-sized buffer.
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
        if !twist_step_half.is_multiple_of(2) || transform_len < 2 {
            return false;
        }

        let layout = SplitLayout::new(chunk_bits, inner_bits);
        debug_assert!(scratch.len() >= layout.cl, "split scratch is undersized");
        let period = inner_bits.wrapping_mul(2);
        let whole_step = twist_step_half.wrapping_shr(1);
        let half_len = transform_len >> 1;
        let half_matrix_len = half_len.wrapping_mul(layout.cl);

        if matrix.len() < transform_len.wrapping_mul(layout.cl) {
            return false;
        }

        let (low_matrix, high_matrix) = matrix.split_at_mut(half_matrix_len);

        // SAFETY: this function's contract guarantees one complete scratch coefficient.
        let stage = unsafe { scratch.get_unchecked_mut(..layout.cl) };
        let mut low_shift = 0_usize;
        let mut twiddle_shift = 0_usize;

        for (index, (low_slot, high_slot)) in low_matrix
            .chunks_exact_mut(layout.cl)
            .zip(high_matrix.chunks_exact_mut(layout.cl))
            .enumerate()
        {
            stage.fill(0);
            // SAFETY: stage is one complete zeroed coefficient.
            unsafe {
                extract_chunk(src, stage, index, layout);
            }

            if low_shift == 0 {
                low_slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and low_slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(low_slot, stage, low_shift, inner_bits);
                }
            }

            let high_shift =
                SsaRing::reduce_mod_period(low_shift.wrapping_add(twiddle_shift), period);
            if high_shift == 0 {
                high_slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and high_slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(high_slot, stage, high_shift, inner_bits);
                }
            }

            low_shift = SsaRing::reduce_mod_period(low_shift.wrapping_add(whole_step), period);
            twiddle_shift =
                SsaRing::reduce_mod_period(twiddle_shift.wrapping_add(omega_shift), period);
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

/// Extract one source chunk into an already-zeroed complete coefficient.
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
        }
    } else {
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64 and fits u32"
        )]
        let shift_up = (LIMB_BITS as u32).wrapping_sub(start_bit);
        for offset in 0..layout.copy_count {
            let source_index = start_limb.wrapping_add(offset);
            let low = src.get(source_index).copied().unwrap_or(0);
            let high = src.get(source_index.wrapping_add(1)).copied().unwrap_or(0);
            // SAFETY: offset < copy_count <= layout.cl == slot.len().
            unsafe {
                *slot.get_unchecked_mut(offset) =
                    low.wrapping_shr(start_bit) | high.wrapping_shl(shift_up);
            }
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
