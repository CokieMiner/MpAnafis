//! Shifted limb accumulation under the centered coefficient magnitude bound.

#![expect(
    unsafe_code,
    reason = "Centered coefficient bounds and checked reconstruction spans establish each shifted carry and borrow window"
)]

use core::num::NonZeroUsize;

use super::{Addition, Limb, SsaCarry, SsaCoefficients};

impl SsaCoefficients {
    /// Adds a nonnegative coefficient at its whole- and sub-limb offset.
    ///
    /// # Safety
    /// `0 < active_len <= coeff_slice.len()`, `shift_sub_bits < Limb::BITS`, and
    /// `dst.len() >= shift_limbs + active_len + 1`. The initialized slices are
    /// disjoint, and the complete biased prefix sum fits the accumulator.
    pub unsafe fn process_positive_coeff(
        coeff_slice: &[Limb],
        active_len: NonZeroUsize,
        shift_limbs: usize,
        shift_sub_bits: u32,
        dst: &mut [Limb],
    ) {
        // SAFETY: the supplied accumulator includes shift_limbs+active_len
        // and its top limb, so this exact endpoint is representable.
        let end = unsafe { shift_limbs.unchecked_add(active_len.get()) };
        debug_assert!(
            end < dst.len(),
            "the accumulator holds the coefficient and its top limb"
        );
        if shift_sub_bits == 0 {
            // SAFETY: end < dst.len() and active_len <= coeff_slice.len(); the
            // source and initialized destination are disjoint by contract.
            let carry = Addition::add_slice_in_place(
                unsafe { dst.get_unchecked_mut(shift_limbs..end) },
                unsafe { coeff_slice.get_unchecked(..active_len.get()) },
            );
            if carry != 0 {
                // SAFETY: the complete biased sum fits dst. A carry from its
                // prefix therefore leaves a nonmaximum limb in this tail.
                unsafe { SsaCarry::absorb_carry(dst.get_unchecked_mut(end..)) }
            }
            return;
        }

        // Shifting directly into the accumulator avoids writing and rereading
        // a coefficient-sized temporary. The biased prefix bound reserves
        // enough tail for every carry.
        // SAFETY: the contract supplies disjoint initialized spans, a nonzero
        // sub-limb shift, and a complete destination including the top limb.
        unsafe {
            Self::shift_add_run(dst, shift_limbs, coeff_slice, active_len, shift_sub_bits);
        }
    }

    /// Adds `coeff[..active] << bit_shift` into `dst[shift..]` in one pass.
    ///
    /// Each source limb feeds the destination directly. A carry escaping the
    /// accumulator is impossible under the reconstruction's biased prefix bound.
    ///
    /// # Safety
    /// `active > 0`, `0 < bit_shift < Limb::BITS`, `coeff.len() >= active`, and
    /// `dst.len() >= shift + active + 1`. The source and destination are disjoint.
    /// The accumulator contains the complete biased prefix sum, including carry space.
    pub unsafe fn shift_add_run(
        dst: &mut [Limb],
        shift: usize,
        coeff: &[Limb],
        active: NonZeroUsize,
        bit_shift: u32,
    ) {
        let active_len = active.get();
        debug_assert!(
            bit_shift > 0 && bit_shift < Limb::BITS,
            "a fused shift carries a nonzero sub-limb shift"
        );
        debug_assert!(
            shift
                .checked_add(active_len)
                .is_some_and(|top| top < dst.len()),
            "the accumulator holds the shifted span and its top limb"
        );
        // SAFETY: this leaf requires 0<bit_shift<Limb::BITS.
        let right_shift = unsafe { Limb::BITS.unchecked_sub(bit_shift) };
        let mut shift_carry = 0;
        let mut carry = false;
        for index in 0..active_len {
            // SAFETY: index < active <= coeff.len(), and shift + index addresses
            // an initialized destination limb below the asserted top.
            let source = unsafe { *coeff.get_unchecked(index) };
            // SAFETY: 0<bit_shift<L gives 0<right_shift<L. Truncating the
            // shifted limb is the required radix-B digit, not shift-count wrapping.
            let shifted = unsafe { source.unchecked_shl(bit_shift) } | shift_carry;
            // SAFETY: the same sub-limb shift proves right_shift<L.
            shift_carry = unsafe { source.unchecked_shr(right_shift) };
            // SAFETY: the same destination bound as the source read.
            let limb = unsafe { dst.get_unchecked_mut(shift.unchecked_add(index)) };
            let (sum, first) = limb.overflowing_add(shifted);
            let (result, second) = sum.overflowing_add(Limb::from(carry));
            *limb = result;
            carry = first || second;
        }
        // SAFETY: the caller reserves the shifted span and its top limb.
        let top = unsafe { shift.unchecked_add(active_len) };
        if shift_carry == 0 && !carry {
            return;
        }
        // SAFETY: the top limb is inside the asserted initialized span.
        let limb = unsafe { dst.get_unchecked_mut(top) };
        // SAFETY: 0<bit_shift<L implies shift_carry<2^bit_shift<=B/2;
        // carry is binary, so their sum is at most B/2 on every limb width.
        let top_addend = unsafe { shift_carry.unchecked_add(Limb::from(carry)) };
        let (result, overflow) = limb.overflowing_add(top_addend);
        *limb = result;
        if overflow {
            // SAFETY: the complete biased sum fits dst, so this escaped top
            // carry must reach a nonmaximum initialized limb in the tail.
            unsafe { SsaCarry::absorb_carry(dst.get_unchecked_mut(top.unchecked_add(1)..)) }
        }
    }

    /// Subtracts `(2^inner_bits + 1 - coeff) << bit_shift` from `dst[shift..]`.
    ///
    /// Magnitude construction, shifting, and subtraction share one pass through
    /// the coefficient's bound prefix, with no intermediate magnitude buffer.
    ///
    /// # Safety
    /// The disjoint coefficient has at least `ml_inner + 1` limbs and is canonical
    /// negative with magnitude below `2^(bound * LIMB_BITS)`. `0 < bound <= ml_inner`,
    /// `bit_shift < Limb::BITS`, and `dst.len() >= shift + bound + 1`.
    /// The biased accumulator remains nonnegative after this complete subtraction.
    pub unsafe fn shift_sub_magnitude_run(
        dst: &mut [Limb],
        shift: usize,
        coeff: &[Limb],
        bound: NonZeroUsize,
        ml_inner: usize,
        bit_shift: u32,
    ) {
        let bound_len = bound.get();
        debug_assert!(bit_shift < Limb::BITS, "the magnitude shift fits one limb");
        debug_assert!(
            shift
                .checked_add(bound_len)
                .is_some_and(|top| top < dst.len()),
            "the accumulator holds the bound prefix and its top limb"
        );
        // SAFETY: ml_inner < coeff.len() and the complete coefficient is initialized.
        if unsafe { *coeff.get_unchecked(ml_inner) } != 0 {
            // A canonical guard-only residue has magnitude one.
            // SAFETY: the caller supplies bit_shift<Limb::BITS.
            let shifted = unsafe { Limb::from(1_usize).unchecked_shl(bit_shift) };
            // SAFETY: shift addresses an initialized limb below the asserted top.
            let limb = unsafe { dst.get_unchecked_mut(shift) };
            let (result, borrow) = limb.overflowing_sub(shifted);
            *limb = result;
            if borrow {
                // SAFETY: the complete biased subtraction is nonnegative;
                // this escaping low borrow requires a nonzero tail limb.
                unsafe {
                    SsaCarry::absorb_borrow(dst.get_unchecked_mut(shift.unchecked_add(1)..));
                }
            }
            return;
        }
        if bit_shift == 0 {
            // SAFETY: the guard case has returned. The coefficient is an ordinary
            // canonical negative residue with the caller's strict magnitude bound,
            // and the disjoint accumulator contains the entire bound and carry tail.
            unsafe {
                subtract_aligned_magnitude(dst, shift, coeff, bound);
            }
            return;
        }
        // SAFETY: the zero-shift branch returned, leaving 0<bit_shift<Limb::BITS.
        let right_shift = unsafe { Limb::BITS.unchecked_sub(bit_shift) };
        let mut carry_add = Limb::from(2_usize);
        let mut shift_carry = 0;
        let mut borrow = false;
        for index in 0..bound_len {
            // SAFETY: index < bound <= ml_inner < coeff.len(), and shift + index
            // addresses an initialized destination limb below the asserted top.
            let complement = !unsafe { *coeff.get_unchecked(index) };
            let (sum, escaped) = complement.overflowing_add(carry_add);
            carry_add = Limb::from(escaped);
            // SAFETY: this branch has 0<bit_shift<L and right_shift=L-bit_shift.
            // Low discarded bits implement the required radix-B reduction.
            let shifted = unsafe { sum.unchecked_shl(bit_shift) } | shift_carry;
            // SAFETY: 0<bit_shift<L proves 0<right_shift<L.
            shift_carry = unsafe { sum.unchecked_shr(right_shift) };
            // SAFETY: the same destination bound as the coefficient read.
            let limb = unsafe { dst.get_unchecked_mut(shift.unchecked_add(index)) };
            let (difference, first) = limb.overflowing_sub(shifted);
            let (result, second) = difference.overflowing_sub(Limb::from(borrow));
            *limb = result;
            borrow = first || second;
        }
        debug_assert_eq!(carry_add, 0, "the magnitude stays within its proven bound");
        // SAFETY: the supplied accumulator contains the shifted bound and its top limb.
        let top = unsafe { shift.unchecked_add(bound_len) };
        if shift_carry == 0 && !borrow {
            return;
        }
        // SAFETY: the top limb is inside the asserted initialized span.
        let limb = unsafe { dst.get_unchecked_mut(top) };
        // SAFETY: this branch has 0<bit_shift<L, hence shift_carry<B/2.
        // Adding its binary borrow remains at most B/2 and fits every Limb.
        let top_subtrahend = unsafe { shift_carry.unchecked_add(Limb::from(borrow)) };
        let (result, escaped) = limb.overflowing_sub(top_subtrahend);
        *limb = result;
        if escaped {
            // SAFETY: the complete biased subtraction is nonnegative;
            // this escaping top borrow requires a nonzero initialized tail.
            unsafe { SsaCarry::absorb_borrow(dst.get_unchecked_mut(top.unchecked_add(1)..)) }
        }
    }

    /// Folds the high accumulator limbs into the low half using `2^mod_bits = -1`.
    ///
    /// Only the first `ml_outer + 1` limbs hold the folded residue; the consumed
    /// high workspace beyond that prefix is unspecified.
    ///
    /// # Safety
    /// `ml_outer>0`, `dst.len()>ml_outer`, and limbs beyond `2*ml_outer`
    /// are zero: the complete accumulator is below `B^(2*ml_outer)`.
    pub unsafe fn fold_high_into_low(dst: &mut [Limb], ml_outer: usize) {
        // SAFETY: the caller supplies the outer data and at least one guard.
        let (low, high) = unsafe { dst.split_at_mut_unchecked(ml_outer) };
        let sub_count = high.len().min(ml_outer);
        // SAFETY: sub_count<=high.len(), and sub_count<=low.len()=ml_outer.
        let ((high_value, high_padding), (low_value, low_tail)) = unsafe {
            (
                high.split_at_mut_unchecked(sub_count),
                low.split_at_mut_unchecked(sub_count),
            )
        };
        debug_assert!(
            high_padding.iter().all(|&limb| limb == 0),
            "the accumulator is below the square of the outer radix"
        );
        // Subtraction consumes the high accumulator's final read. Only its
        // first limb becomes the output guard; the remaining workspace is dead.
        let borrow = Addition::sub_slice_in_place(low_value, high_value);
        let final_borrow = borrow != 0 && SsaCarry::propagate_borrow(low_tail);
        if final_borrow {
            // SAFETY: dst includes the guard and its data subtraction borrowed once.
            unsafe {
                let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml_outer);
            }
        } else {
            // SAFETY: the complete output retains this guard; its previous
            // high-accumulator value has no reader after subtraction.
            unsafe {
                *dst.get_unchecked_mut(ml_outer) = 0;
            }
        }
    }
}

/// Subtracts a limb-aligned ordinary negative coefficient's magnitude.
///
/// Complement-plus-two constructs `(2^n+1)-coeff` modulo `B^bound`.
/// The strict magnitude bound makes those low digits the exact magnitude.
/// Each digit feeds the accumulator immediately, without a temporary write,
/// a second magnitude read, or a highest-nonzero-digit calculation.
///
/// # Safety
/// `coeff` is canonical negative with guard zero and magnitude strictly below
/// `B^bound`; `0 < bound <= coeff.len()`. The disjoint initialized destination
/// includes `shift + bound` and enough tail for the nonnegative biased result.
unsafe fn subtract_aligned_magnitude(
    dst: &mut [Limb],
    shift: usize,
    coeff: &[Limb],
    bound: NonZeroUsize,
) {
    let bound_len = bound.get();
    let mut carry = Limb::from(2_usize);
    let mut borrow = false;
    for index in 0..bound_len {
        // SAFETY: index < bound <= coeff.len(); the coefficient is initialized.
        let complement = !unsafe { *coeff.get_unchecked(index) };
        let (magnitude, escaped) = complement.overflowing_add(carry);
        carry = Limb::from(escaped);
        if magnitude == 0 && !borrow {
            continue;
        }
        // SAFETY: the complete shifted bound lies inside the initialized
        // accumulator, disjoint from the source coefficient.
        let limb = unsafe { dst.get_unchecked_mut(shift.unchecked_add(index)) };
        let (difference, first) = limb.overflowing_sub(magnitude);
        let (result, second) = difference.overflowing_sub(Limb::from(borrow));
        *limb = result;
        borrow = first || second;
    }
    debug_assert_eq!(carry, 0, "the magnitude stays within its proven bound");
    if borrow {
        // SAFETY: the complete biased subtraction is nonnegative. Its shifted
        // prefix borrowed, so the initialized tail contains a nonzero limb.
        unsafe { SsaCarry::absorb_borrow(dst.get_unchecked_mut(shift.unchecked_add(bound_len)..)) }
    }
}
