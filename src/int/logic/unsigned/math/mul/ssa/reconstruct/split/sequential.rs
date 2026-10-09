//! Sequential operand splitting with fused whole-bit and half-bit pre-twists.
//!
//! Declares the [`SsaCoefficients`] namespace, because splitting is where an operand
//! first becomes a coefficient matrix. [`SsaCoefficients::reconstruct`] provides
//! the reverse direction on the same namespace.

#![expect(
    unsafe_code,
    reason = "Checked split geometry bounds operand windows, initialized coefficient spans, and reduced pre-twists"
)]

use core::num::NonZeroUsize;

use super::{LIMB_BITS, Limb, RingPeriods, SsaRing, SsaTransform};

/// Namespace for the transform's coefficient matrix: cutting an operand into it,
/// and accumulating a product back out of it.
///
/// The two directions are exact inverses either side of the transform, so the
/// chunk geometry one assumes is the one the other undoes. Keeping them on one
/// namespace is what makes that pairing visible at the call sites.
pub struct SsaCoefficients;

/// Values derived once from a transform's coefficient layout.
#[derive(Clone, Copy)]
pub struct SplitLayout {
    cl: NonZeroUsize,
    chunk_bits: NonZeroUsize,
    copy_count: usize,
    mask_index: usize,
    needs_mask: bool,
    mask: Limb,
}

impl SplitLayout {
    pub fn new(chunk_bits: NonZeroUsize, inner_bits: usize, cl: NonZeroUsize) -> Self {
        debug_assert!(
            chunk_bits.get() < inner_bits,
            "validated transform chunks are positive and below their coefficient width"
        );
        debug_assert_eq!(
            cl.get(),
            SsaRing::coeff_limbs(inner_bits).get(),
            "the plan binds the complete coefficient width to its ring"
        );
        let chunk_remainder = chunk_bits.get().rem_euclid(LIMB_BITS);
        let has_partial_limb = chunk_remainder != 0;
        let copy_count = chunk_bits.get().div_ceil(LIMB_BITS);
        // SAFETY: chunk_bits>0 gives copy_count>0; chunk_bits<inner_bits
        // gives copy_count<=mod_limbs(inner_bits)<cl on both SSA widths.
        let mask_index = unsafe { copy_count.unchecked_sub(1) };
        // SAFETY: a remainder modulo LIMB_BITS is strictly smaller than LIMB_BITS.
        let mask_shift = unsafe { LIMB_BITS.unchecked_sub(chunk_remainder) };
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "mask_shift <= LIMB_BITS <= 64, so it always fits u32"
        )]
        let mask = Limb::MAX.wrapping_shr(mask_shift as u32);

        Self {
            cl,
            chunk_bits,
            copy_count,
            mask_index,
            needs_mask: has_partial_limb,
            mask,
        }
    }

    /// Width of the complete coefficient whose chunk and zero suffix this
    /// immutable layout binds together.
    pub const fn coefficient_len(self) -> NonZeroUsize {
        self.cl
    }

    /// Initializes the invariant zero suffix beyond every extracted chunk.
    ///
    /// # Safety
    /// `stage` is a complete coefficient of the validated layout.
    pub unsafe fn initialize_padding(self, stage: &mut [Limb]) {
        // SAFETY: construction proves copy_count<cl<=stage.len().
        unsafe {
            stage.get_unchecked_mut(self.copy_count..).fill(0);
        }
    }

    /// Adds two reduced exponents without requiring representable `2*period`.
    /// The validated twist callers establish `x,y<period`; the plan carries
    /// the period's positivity through its type.
    pub const fn add_reduced(x: usize, y: usize, period: NonZeroUsize) -> usize {
        // SAFETY: y<period gives an exact positive distance. If x is below
        // that distance, x+y<period; otherwise x-distance is nonnegative.
        unsafe {
            let distance = period.get().unchecked_sub(y);
            if x >= distance {
                x.unchecked_sub(distance)
            } else {
                x.unchecked_add(y)
            }
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
    /// through [`SsaRing::shift_sqrt2`] and no geometry needs a separate
    /// whole-matrix twist pass.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint buffer of at least two complete coefficients.
    /// The positive chunk width and periods come from a validated plan;
    /// `periods.half == 2*periods.whole == 4*inner_bits`, with positive,
    /// limb-aligned `inner_bits` and `cl == coeff_limbs(inner_bits)`.
    /// `transform_len*chunk_bits` and
    /// `src.len()*LIMB_BITS` fit usize.
    #[expect(
        clippy::too_many_arguments,
        reason = "Splitting consumes the plan's typed coefficient width, chunk and periods alongside disjoint buffers"
    )]
    pub unsafe fn split_twisted(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        chunk_bits: NonZeroUsize,
        cl: NonZeroUsize,
        periods: RingPeriods,
        twist_step_half: usize,
        scratch: &mut [Limb],
    ) {
        let inner_bits = periods.whole.get() >> 1;
        let layout = SplitLayout::new(chunk_bits, inner_bits, cl);
        debug_assert!(
            scratch.len() >= cl.get().saturating_mul(2),
            "split scratch must hold the staging coefficient and the sqrt(2) factor"
        );
        // Half-bit exponents live modulo 4n: sqrt(2) has order 4n because its
        // square, 2, has order 2n. This is the same domain the inverse twist
        // correction accumulates in.
        let half_period = periods.half;
        let twist_step = SsaRing::reduce_mod_period(twist_step_half, half_period);
        let mut shift = 0_usize;

        // SAFETY: the caller validates the source's representable bit capacity.
        let src_bits = unsafe { src.len().unchecked_mul(LIMB_BITS) };
        let active_chunks = src_bits
            .div_ceil(layout.chunk_bits.get())
            .min(transform_len);

        // Every chunk reuses the same disjoint staging and factor coefficients.
        // SAFETY: the contract guarantees two complete scratch coefficients.
        let (stage, factor) = unsafe { scratch.split_at_mut_unchecked(cl.get()) };
        // SAFETY: the scratch split supplies one complete staging coefficient.
        unsafe {
            layout.initialize_padding(stage);
        }
        for index in 0..active_chunks {
            // SAFETY: stage is one complete coefficient; extract_chunk fully
            // defines it without requiring a pre-zeroed buffer.
            unsafe {
                Self::extract_chunk(src, stage, index, layout);
            }
            // SAFETY: the matrix contains transform_len complete slots.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl.get()) };
            let whole_shift = shift >> 1;
            if !shift.is_multiple_of(2) {
                // SAFETY: the complete source, destination, and factor arena
                // are disjoint; shift/2 < 2n and the extracted guard is zero.
                unsafe {
                    SsaRing::shift_sqrt2_from(slot, stage, whole_shift, inner_bits, factor);
                }
            } else if whole_shift == 0 {
                slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(slot, stage, whole_shift, inner_bits);
                }
            }
            shift = SplitLayout::add_reduced(shift, twist_step, half_period);
            debug_assert!(shift < half_period.get(), "split twists stay reduced");
        }

        // SAFETY: active_chunks<=transform_len and the caller owns its full matrix.
        let active_limbs = unsafe { active_chunks.unchecked_mul(cl.get()) };
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
    /// length (`transform_len / 2`), the upper matrix half is implicitly zero. The first
    /// DIF butterfly stage is therefore `(low, high) = (low, low * w^j)`.
    ///
    /// This method extracts chunk `j`, computes `low = chunk * theta^j`, and computes
    /// `high = low * w^j = chunk * (theta^j * w^j)`, writing directly to both halves
    /// of the matrix in one streaming pass. This avoids writing zeros to the high half
    /// and eliminates a complete DRAM read-and-rewrite pass of the matrix. Twist
    /// exponents are tracked in half-bit units modulo `4n`, so odd steps fold their
    /// `sqrt(2)` factor into the same streaming pass.
    /// Only `active_chunks` slots in each half are written. Their remaining
    /// physical slots are arbitrary; the sparse DIF children never read them.
    ///
    /// Returns `false` only when `transform_len < 2`.
    ///
    /// # Safety
    ///
    /// `matrix` contains at least `transform_len * SsaRing::coeff_limbs(inner_bits)` limbs,
    /// and `scratch` is a disjoint buffer of at least two complete coefficients.
    /// The chunk and ring geometry and source bit capacity satisfy `split_twisted`.
    /// `active_chunks <= transform_len/2` bounds the source polynomial support.
    #[expect(
        clippy::too_many_arguments,
        reason = "Internal FFT staging requires explicit operand, matrix, geometry, and scratch buffers"
    )]
    pub unsafe fn split_twisted_and_stage1_dif(
        src: &[Limb],
        matrix: &mut [Limb],
        transform_len: usize,
        active_chunks: usize,
        chunk_bits: NonZeroUsize,
        cl: NonZeroUsize,
        periods: RingPeriods,
        twist_step_half: usize,
        omega_shift: usize,
        scratch: &mut [Limb],
    ) -> bool {
        if transform_len < 2 {
            return false;
        }

        let inner_bits = periods.whole.get() >> 1;
        let layout = SplitLayout::new(chunk_bits, inner_bits, cl);
        debug_assert!(
            scratch.len() >= cl.get().saturating_mul(2),
            "split scratch must hold the staging coefficient and the sqrt(2) factor"
        );
        // Half-bit twist exponents live modulo 4n; the whole-bit transform root
        // omega contributes two half-bit units per step.
        let RingPeriods {
            half: half_period,
            whole: whole_period,
        } = periods;
        let twist_step = SsaRing::reduce_mod_period(twist_step_half, half_period);
        let root_step = SsaRing::reduce_mod_period(omega_shift, whole_period);
        let half_len = transform_len >> 1;
        // SAFETY: the caller provides transform_len*cl limbs, with a positive
        // power-of-two transform_len; its half matrix is a valid partition.
        let (low_matrix, high_matrix) =
            unsafe { matrix.split_at_mut_unchecked(half_len.unchecked_mul(cl.get())) };

        let mut low_shift = 0_usize;
        let mut twiddle_shift = 0_usize;

        debug_assert!(
            active_chunks <= half_len,
            "the declared input support fits each first-stage child"
        );

        // SAFETY: the two complete scratch coefficients are disjoint and each
        // twist reads stage without modifying its invariant zero suffix.
        let (stage, factor) = unsafe { scratch.split_at_mut_unchecked(cl.get()) };
        // SAFETY: the scratch split supplies one complete staging coefficient.
        unsafe {
            layout.initialize_padding(stage);
        }
        for (index, (low_slot, high_slot)) in low_matrix
            .chunks_exact_mut(cl.get())
            .zip(high_matrix.chunks_exact_mut(cl.get()))
            .take(active_chunks)
            .enumerate()
        {
            // SAFETY: stage is one complete coefficient; extract_chunk fully
            // defines it without requiring a pre-zeroed buffer.
            unsafe {
                Self::extract_chunk(src, stage, index, layout);
            }

            // low = chunk * theta^index.
            let low_whole = low_shift >> 1;
            if !low_shift.is_multiple_of(2) {
                // SAFETY: the complete source, destination, and factor arena
                // are disjoint; low_shift/2 < 2n and the source guard is zero.
                unsafe {
                    SsaRing::shift_sqrt2_from(low_slot, stage, low_whole, inner_bits, factor);
                }
            } else if low_whole == 0 {
                low_slot.copy_from_slice(stage);
            } else {
                // SAFETY: stage is canonical, and low_slot and stage are disjoint
                // complete coefficients in the same Fermat ring.
                unsafe {
                    SsaRing::shift_from(low_slot, stage, low_whole, inner_bits);
                }
            }

            // Both outputs share the same half-bit parity. Reuse the completed
            // low twist, including its square-root factor, for high = low*w^j.
            if twiddle_shift == 0 {
                high_slot.copy_from_slice(low_slot);
            } else {
                // SAFETY: the complete slots are disjoint and the low slot is
                // semi-normalized after its twist; twiddle_shift < 2*inner_bits.
                unsafe {
                    SsaRing::shift_from(high_slot, low_slot, twiddle_shift, inner_bits);
                }
            }

            // Reduced increments need at most one subtraction, including
            // caller-forced twist steps that wrap within the matrix.
            low_shift = SplitLayout::add_reduced(low_shift, twist_step, half_period);
            twiddle_shift = SplitLayout::add_reduced(twiddle_shift, root_step, whole_period);
        }

        true
    }

    /// Extract one source chunk into a complete coefficient.
    ///
    /// Fully defines the chunk prefix while retaining its initialized zero suffix.
    ///
    /// # Safety
    ///
    /// `slot.len() == layout.cl`; `initialize_padding` establishes the zero suffix
    /// beyond `copy_count`, and consumers preserve it between extractions.
    /// `index*layout.chunk_bits` is within the plan's representable outer width.
    pub unsafe fn extract_chunk(
        src: &[Limb],
        slot: &mut [Limb],
        index: usize,
        layout: SplitLayout,
    ) {
        // SAFETY: the caller supplies an index within the validated outer bit width.
        let bit_start = unsafe { index.unchecked_mul(layout.chunk_bits.get()) };
        let start_limb = bit_start.div_euclid(LIMB_BITS);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "bit_start modulo LIMB_BITS is at most 63 and fits u32"
        )]
        let start_bit = bit_start.rem_euclid(LIMB_BITS) as u32;

        if start_bit == 0 {
            if start_limb < src.len() {
                // SAFETY: this branch establishes start_limb<src.len().
                let available =
                    unsafe { src.len().unchecked_sub(start_limb) }.min(layout.copy_count);
                // SAFETY: the calculated source and destination prefixes are in bounds.
                unsafe {
                    slot.get_unchecked_mut(..available).copy_from_slice(
                        src.get_unchecked(start_limb..start_limb.unchecked_add(available)),
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
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS is at most 64 and fits u32"
            )]
            // SAFETY: this branch has 0<start_bit<LIMB_BITS<=64.
            let shift_up = unsafe { (LIMB_BITS as u32).unchecked_sub(start_bit) };
            // Only the final chunk can overhang the source, so the readable count
            // splits the sweep into a fully unchecked paired body, at most one
            // low-only limb at the overhang, and a zero tail.
            let available = src.len().saturating_sub(start_limb);
            let paired = available.saturating_sub(1).min(layout.copy_count);
            for offset in 0..paired {
                // SAFETY: offset < paired <= available - 1 proves both source
                // limbs lie below src.len(), and offset < copy_count <= slot.len().
                unsafe {
                    let source_index = start_limb.unchecked_add(offset);
                    let low = *src.get_unchecked(source_index);
                    let high = *src.get_unchecked(source_index.unchecked_add(1));
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
                    let low = *src.get_unchecked(start_limb.unchecked_add(written));
                    *slot.get_unchecked_mut(written) = low.wrapping_shr(start_bit);
                }
                // SAFETY: written<copy_count, so its next value<=copy_count.
                written = unsafe { written.unchecked_add(1) };
            }
            // SAFETY: written <= copy_count <= layout.cl == slot.len().
            unsafe {
                slot.get_unchecked_mut(written..layout.copy_count).fill(0);
            }
        }

        if layout.needs_mask {
            // SAFETY: needs_mask proves mask_index < layout.cl == slot.len().
            unsafe {
                *slot.get_unchecked_mut(layout.mask_index) &= layout.mask;
            }
        }
    }
}
