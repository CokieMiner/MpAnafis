//! Guarded fixed-width addition, subtraction, and paired evaluations.

#![expect(
    unsafe_code,
    reason = "Initialized guarded spans prove raw arithmetic bounds, disjointness, and bounded scalar carry states"
)]

use core::{
    cmp::{Ordering, min},
    ptr::copy_nonoverlapping,
};

use super::{AddMulKernel, Addition, ArchKernels, Limb, SharedEval};

impl SharedEval {
    #[expect(
        clippy::inline_always,
        reason = "Critical for recursive reconstruction hot paths"
    )]
    /// Add `src` into `dst` at a limb offset, and report the carry frontier.
    ///
    /// The return value is one past the carry frontier. Carries propagate
    /// through the retained guard; the complete nonnegative sum must fit.
    ///
    /// # Safety
    ///
    /// When `src` is nonempty, `shift_limbs <= dst.len()` and
    /// `src.len() <= dst.len() - shift_limbs`. These inequalities also prove
    /// `shift_limbs + src.len()` cannot overflow `usize`.
    #[inline(always)]
    pub unsafe fn fused_add_shifted_in_place(
        dst: &mut [Limb],
        src: &[Limb],
        shift_limbs: usize,
    ) -> usize {
        if src.is_empty() {
            return 0;
        }
        debug_assert!(
            shift_limbs <= dst.len() && src.len() <= dst.len().saturating_sub(shift_limbs),
            "shifted source exceeds reconstruction buffer"
        );
        // SAFETY: the unsafe function contract proves `shift_limbs <= dst.len()`;
        // the returned suffix has `dst.len() - shift_limbs >= src.len()` limbs.
        let carry =
            Addition::add_slice_in_place(unsafe { dst.get_unchecked_mut(shift_limbs..) }, src);
        // SAFETY: the contract bounds this sum by the valid destination length.
        let carry_start = unsafe { shift_limbs.unchecked_add(src.len()) };
        if carry != 0 {
            for index in carry_start..dst.len() {
                // SAFETY: index is produced by a range ending at dst.len().
                let limb = unsafe { dst.get_unchecked_mut(index) };
                let (sum, overflow) = limb.overflowing_add(1);
                *limb = sum;
                if !overflow {
                    // SAFETY: index < dst.len(), so index+1 <= dst.len().
                    return unsafe { index.unchecked_add(1) };
                }
            }
            debug_assert_eq!(carry, 0, "reconstruction buffer dropped a final carry");
            return dst.len();
        }
        carry_start
    }

    /// Add one polynomial part to a guarded fixed-width evaluation.
    ///
    /// Evaluation uses zero offset and discards the reconstruction frontier.
    ///
    /// # Safety
    ///
    /// If `src` is nonempty, `dst.len() >= src.len()`.
    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook evaluation")]
    #[inline(always)]
    pub unsafe fn add_part(dst: &mut [Limb], src: &[Limb]) {
        // SAFETY: the caller guarantees the source fits at offset zero.
        let _ = unsafe { Self::fused_add_shifted_in_place(dst, src, 0) };
    }

    /// Add one interpolated coefficient at a radix-limb offset.
    ///
    /// The nonnegative coefficient, shifted by `shift`, is a term of the
    /// complete product and therefore fits the destination. Zero coefficients
    /// access neither the coefficient nor the destination at that shift.
    pub fn add_coefficient_in_place(dst: &mut [Limb], coefficient: &[Limb], shift: usize) {
        let active = Self::active_len(coefficient);
        if active == 0 {
            return;
        }
        debug_assert!(
            shift <= dst.len() && active <= dst.len().saturating_sub(shift),
            "interpolated coefficient exceeds the full product width"
        );
        // SAFETY: active_len returns a prefix length at most coefficient.len().
        let (active_span, _) = unsafe { coefficient.split_at_unchecked(active) };
        // SAFETY: the coefficient is a nonnegative term of the complete
        // product, so its active span fits wholly at the caller's radix shift.
        let _ = unsafe { Self::fused_add_shifted_in_place(dst, active_span, shift) };
    }

    /// Replace an even/odd evaluation pair with their sum and absolute difference.
    ///
    /// The points are `E+O` and `E-O`. Return whether `O>E`, and select the
    /// nonnegative subtraction orientation before the fused limb pass.
    pub fn sum_and_absolute_difference(even_sum: &mut [Limb], odd_difference: &mut [Limb]) -> bool {
        debug_assert_eq!(
            even_sum.len(),
            odd_difference.len(),
            "paired evaluation widths must match"
        );
        if even_sum.iter().rev().cmp(odd_difference.iter().rev()) == Ordering::Less {
            Self::apply_sum_and_difference::<true>(even_sum, odd_difference);
            true
        } else {
            Self::apply_sum_and_difference::<false>(even_sum, odd_difference);
            false
        }
    }

    /// Computes the paired sum and absolute difference with a known sign.
    ///
    /// `odd_is_larger` must equal the comparison `O>E` for these buffers.
    pub fn apply_sum_and_absolute_difference(
        even_sum: &mut [Limb],
        odd_difference: &mut [Limb],
        odd_is_larger: bool,
    ) {
        debug_assert_eq!(
            even_sum.len(),
            odd_difference.len(),
            "paired evaluation widths must match"
        );
        if odd_is_larger {
            Self::apply_sum_and_difference::<true>(even_sum, odd_difference);
        } else {
            Self::apply_sum_and_difference::<false>(even_sum, odd_difference);
        }
    }

    /// Replace an already-summed evaluation with the absolute difference of its two
    /// halves.
    ///
    /// `even_sum` holds `E + O` and `odd` holds `O`. On return `even_sum` holds
    /// `|E - O|` and `odd` is clobbered. `odd_is_larger` carries the same meaning as
    /// in [`Self::apply_sum_and_absolute_difference`].
    pub fn overwrite_sum_with_absolute_difference(
        even_sum: &mut [Limb],
        odd: &mut [Limb],
        odd_is_larger: bool,
    ) {
        debug_assert_eq!(
            even_sum.len(),
            odd.len(),
            "paired evaluation widths must match"
        );
        if odd_is_larger {
            // O - E = 2O - (E + O), so the doubled odd half absorbs the sum.
            Self::double_evaluation_in_place(odd);
            let borrow = Addition::sub_slice_in_place(odd, even_sum);
            debug_assert_eq!(
                borrow, 0,
                "negative evaluation magnitude must be nonnegative"
            );
            // SAFETY: paired evaluations have the same initialized width, and
            // their exclusive mutable borrows prove the complete spans disjoint.
            unsafe {
                copy_nonoverlapping(odd.as_ptr(), even_sum.as_mut_ptr(), odd.len());
            }
        } else {
            // E - O = (E + O) - 2O, in a single multiply-and-subtract pass.
            Self::sub_mul_word_in_place(even_sum, odd, 2);
        }
    }

    /// Double a fixed-width evaluation whose retained guard proves no overflow.
    pub fn double_evaluation_in_place(value: &mut [Limb]) {
        let mut carry = 0;
        for limb in value {
            let (doubled, overflow) = limb.overflowing_add(*limb);
            // SAFETY: doubling modulo the even radix B leaves an even word
            // at most B-2. The previous overflow carry is binary, so their
            // sum is at most B-1 and cannot produce another carry.
            *limb = unsafe { doubled.unchecked_add(carry) };
            carry = Limb::from(overflow);
        }
        debug_assert_eq!(carry, 0, "evaluation exceeded its retained guard limb");
    }

    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook evaluation")]
    /// Add `scalar * src` into `dst`, selecting the backend for this one call.
    ///
    /// Multiple-point evaluators pass a preselected kernel instead.
    #[inline(always)]
    pub fn add_mul_word_in_place(dst: &mut [Limb], src: &[Limb], scalar: Limb) {
        let kernel = ArchKernels::selected_add_mul_limbs_unchecked();
        Self::add_mul_word_with_kernel_in_place(dst, src, scalar, kernel);
    }

    /// Add a scalar product using a backend selected once by the outer algorithm.
    pub fn add_mul_word_with_kernel_in_place(
        dst: &mut [Limb],
        src: &[Limb],
        scalar: Limb,
        kernel: AddMulKernel,
    ) {
        if scalar == 0 || src.is_empty() {
            return;
        }
        debug_assert!(
            src.len() <= dst.len(),
            "scalar-product source exceeds evaluation destination"
        );
        // SAFETY: the guarded-evaluation contract proves both pointer spans
        // cover `src.len()` initialized limbs. Rust's borrows make them disjoint.
        let mut carry = unsafe { kernel(dst.as_mut_ptr(), src.as_ptr(), src.len(), scalar) };
        if carry != 0 {
            // SAFETY: the guarded-evaluation contract supplies src.len()
            // destination limbs; the carry starts in the disjoint suffix.
            let (_, suffix) = unsafe { dst.split_at_mut_unchecked(src.len()) };
            for limb in suffix {
                let (sum, overflow) = limb.overflowing_add(carry);
                *limb = sum;
                if !overflow {
                    break;
                }
                carry = 1;
            }
        }
    }

    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
    /// Subtract `scalar * src` from `dst`, borrowing through the guard.
    ///
    /// The interpolation counterpart of [`Self::add_mul_word_in_place`]. The kernel
    /// reports the escaping product limb and the subtraction's borrow separately;
    /// both are owed to the limbs above `src`, so they are summed before being
    /// propagated.
    #[inline(always)]
    pub fn sub_mul_word_in_place(dst: &mut [Limb], src: &[Limb], scalar: Limb) {
        if scalar == 0 || src.is_empty() {
            return;
        }
        debug_assert!(
            src.len() <= dst.len(),
            "scalar-product source exceeds interpolation destination"
        );
        // SAFETY: the interpolation-buffer contract proves both pointer spans
        // cover `src.len()` initialized limbs. Rust's borrows make them disjoint.
        let (carry, initial_borrow) = unsafe {
            ArchKernels::sub_mul_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src.len(), scalar)
        };
        // SAFETY: scalar > 0, and a scalar product's escaping limb is at most
        // scalar-1. initial_borrow is binary, so their sum is at most scalar
        // <= Limb::MAX, including scalar == Limb::MAX.
        let mut borrow = unsafe { initial_borrow.unchecked_add(carry) };
        if borrow != 0 {
            // SAFETY: the interpolation layout supplies src.len() destination
            // limbs; the product borrow propagates through the retained suffix.
            let (_, suffix) = unsafe { dst.split_at_mut_unchecked(src.len()) };
            for limb in suffix {
                let (difference, underflow) = limb.overflowing_sub(borrow);
                *limb = difference;
                if !underflow {
                    break;
                }
                borrow = 1;
            }
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "Critical for fixed-width interpolation"
    )]
    /// Subtract `src` from `dst` where the two need not share a width.
    ///
    /// Interpolation subtracts values whose active widths differ, so the shared
    /// prefix is subtracted and any borrow is then propagated alone through the
    /// remainder of `dst`. A final borrow is left to the guard, matching arithmetic
    /// modulo `B^n` on two's-complement intermediates.
    #[inline(always)]
    pub fn sub_full_slices_in_place(dst: &mut [Limb], src: &[Limb]) {
        let shared_len = min(dst.len(), src.len());
        // SAFETY: shared_len is the minimum of both slice lengths.
        let initial_borrow =
            Addition::sub_slice_in_place(dst, unsafe { src.get_unchecked(..shared_len) });
        if initial_borrow != 0 {
            // SAFETY: shared_len is the minimum of both input lengths, hence
            // no greater than dst.len(). The suffix begins after that prefix.
            let (_, suffix) = unsafe { dst.split_at_mut_unchecked(shared_len) };
            let mut borrow = initial_borrow;
            for limb in suffix {
                let (difference, underflow) = limb.overflowing_sub(borrow);
                *limb = difference;
                if !underflow {
                    break;
                }
                borrow = 1;
            }
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "Critical for fixed-width interpolation"
    )]
    /// Subtract two values from `dst` in a single pass.
    ///
    /// Interpolation repeatedly owes one accumulator two subtractions. Running them
    /// together carries two independent borrow chains over one traversal, which
    /// reads and writes `dst` once instead of twice; the chains stay independent
    /// because each tracks its own operand. Widths may differ, so whatever extends
    /// past the shared prefix is finished by
    /// [`Self::sub_full_slices_with_borrow_in_place`] carrying that chain's borrow in.
    #[inline(always)]
    pub fn sub_two_full_slices_in_place(dst: &mut [Limb], src1: &[Limb], src2: &[Limb]) {
        let shared_len = min(dst.len(), min(src1.len(), src2.len()));
        let (dst_shared, dst_suffix) = dst.split_at_mut(shared_len);
        let mut borrow1 = 0;
        let mut borrow2 = 0;
        for ((d, s1), s2) in dst_shared.iter_mut().zip(src1).zip(src2) {
            let (d1, b1) = d.overflowing_sub(*s1);
            let (d2, b2) = d1.overflowing_sub(borrow1);
            borrow1 = Limb::from(b1) | Limb::from(b2);

            let (d3, b3) = d2.overflowing_sub(*s2);
            let (d4, b4) = d3.overflowing_sub(borrow2);
            borrow2 = Limb::from(b3) | Limb::from(b4);

            *d = d4;
        }
        // SAFETY: shared_len is the minimum of dst, src1, and src2 lengths.
        let (src1_suffix, src2_suffix) = unsafe {
            (
                src1.get_unchecked(shared_len..),
                src2.get_unchecked(shared_len..),
            )
        };
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src1_suffix, borrow1);
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src2_suffix, borrow2);
    }

    #[expect(
        clippy::inline_always,
        reason = "Critical for fixed-width interpolation"
    )]
    /// Subtract three values from `dst` in a single pass.
    #[inline(always)]
    pub fn sub_three_full_slices_in_place(
        dst: &mut [Limb],
        src1: &[Limb],
        src2: &[Limb],
        src3: &[Limb],
    ) {
        let shared_len = min(dst.len(), min(src1.len(), min(src2.len(), src3.len())));
        let (dst_shared, dst_suffix) = dst.split_at_mut(shared_len);
        let mut borrow1 = 0;
        let mut borrow2 = 0;
        let mut borrow3 = 0;
        for (((d, s1), s2), s3) in dst_shared.iter_mut().zip(src1).zip(src2).zip(src3) {
            let (d1, b1) = d.overflowing_sub(*s1);
            let (d2, b2) = d1.overflowing_sub(borrow1);
            borrow1 = Limb::from(b1) | Limb::from(b2);

            let (d3, b3) = d2.overflowing_sub(*s2);
            let (d4, b4) = d3.overflowing_sub(borrow2);
            borrow2 = Limb::from(b3) | Limb::from(b4);

            let (d5, b5) = d4.overflowing_sub(*s3);
            let (d6, b6) = d5.overflowing_sub(borrow3);
            borrow3 = Limb::from(b5) | Limb::from(b6);

            *d = d6;
        }
        // SAFETY: shared_len is the minimum of all four slice lengths.
        let (src1_suffix, src2_suffix, src3_suffix) = unsafe {
            (
                src1.get_unchecked(shared_len..),
                src2.get_unchecked(shared_len..),
                src3.get_unchecked(shared_len..),
            )
        };
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src1_suffix, borrow1);
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src2_suffix, borrow2);
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src3_suffix, borrow3);
    }

    /// Replace `dst` with `positive-dst` modulo its fixed width.
    pub fn reverse_difference_in_place(dst: &mut [Limb], positive: &[Limb]) {
        debug_assert_eq!(
            dst.len(),
            positive.len(),
            "reverse-difference widths must match"
        );
        if dst.is_empty() {
            return;
        }
        // SAFETY: both slices cover the same nonzero length. Every architecture
        // backend loads src2[i] before writing dst[i], so src2 == dst is valid.
        // A final borrow is intentionally discarded for signed two's-complement
        // interpolation intermediates, exactly matching arithmetic modulo B^n.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst.as_mut_ptr(),
                positive.as_ptr(),
                dst.as_ptr(),
                dst.len(),
            )
        };
        let _ = borrow;
    }

    #[expect(
        clippy::inline_always,
        reason = "Critical for fixed-width interpolation"
    )]
    #[inline(always)]
    fn sub_full_slices_with_borrow_in_place(dst: &mut [Limb], src: &[Limb], mut borrow: Limb) {
        let shared_len = min(dst.len(), src.len());
        // SAFETY: shared_len <= dst.len() by its definition as the minimum.
        let (dst_shared, dst_suffix) = unsafe { dst.split_at_mut_unchecked(shared_len) };
        for (d, s) in dst_shared.iter_mut().zip(src) {
            let (difference1, underflow1) = d.overflowing_sub(*s);
            let (difference2, underflow2) = difference1.overflowing_sub(borrow);
            borrow = Limb::from(underflow1) | Limb::from(underflow2);
            *d = difference2;
        }
        if borrow != 0 {
            for limb in dst_suffix {
                let (difference, underflow) = limb.overflowing_sub(borrow);
                *limb = difference;
                if !underflow {
                    break;
                }
                borrow = 1;
            }
        }
    }

    fn apply_sum_and_difference<const ODD_MINUS_EVEN: bool>(
        even_sum: &mut [Limb],
        odd_difference: &mut [Limb],
    ) {
        // SAFETY: both evaluation buffers are disjoint and have equal widths. The
        // ordering check in `sum_and_absolute_difference` selected the orientation
        // whose mathematical difference is nonnegative.
        let (sum_carry, difference_borrow) = unsafe {
            if ODD_MINUS_EVEN {
                ArchKernels::add_reverse_sub_limbs_unchecked(
                    even_sum.as_mut_ptr(),
                    odd_difference.as_mut_ptr(),
                    even_sum.len(),
                )
            } else {
                ArchKernels::add_sub_limbs_unchecked(
                    even_sum.as_mut_ptr(),
                    odd_difference.as_mut_ptr(),
                    even_sum.len(),
                )
            }
        };
        debug_assert_eq!(sum_carry, 0, "positive evaluation exceeded its guard limb");
        debug_assert_eq!(difference_borrow, 0, "absolute difference underflowed");
    }
}
