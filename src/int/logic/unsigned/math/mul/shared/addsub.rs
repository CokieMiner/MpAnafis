//! Fixed-width buffer inspection and guarded addition and subtraction.
//!
//! The buffer inspection functions establish or measure the state that the
//! guarded arithmetic below then relies on. Every guarded routine writes into a
//! buffer that carries at least one guard limb, so a final carry or borrow is
//! propagated into the guard rather than returned.

use core::cmp::{Ordering, min};

use super::{Addition, ArchKernels, Limb};

/// Namespace for fixed-width evaluation, interpolation, and exact-division helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedEval;

// ─── Buffer inspection and preparation ───────────────────────────────────────

impl SharedEval {
    /// Index one past the highest nonzero limb.
    ///
    /// The single implementation of the backward significance scan; evaluation,
    /// interpolation, Karatsuba reconstruction, and the SSA reconstruction sweep
    /// all bound their work by it.
    #[allow(
        clippy::inline_always,
        reason = "Critical for recursive reconstruction hot paths"
    )]
    #[inline(always)]
    pub fn active_len(limbs: &[Limb]) -> usize {
        let mut len = limbs.len();
        while len > 0 {
            let index = len.wrapping_sub(1);
            // SAFETY: len is positive and never exceeds limbs.len().
            if unsafe { *limbs.get_unchecked(index) } != 0 {
                break;
            }
            len = index;
        }
        len
    }

    /// Copy a polynomial part into an evaluation buffer and clear its guard.
    pub fn copy_part(dst: &mut [Limb], src: &[Limb]) {
        assert!(
            src.len() < dst.len(),
            "evaluation part must leave at least one guard limb"
        );
        let (body, guard) = dst.split_at_mut(src.len());
        body.copy_from_slice(src);
        guard.fill(0);
    }

    /// Compare two operands of different widths as if the narrower were zero-padded.
    ///
    /// An unbalanced Toom split evaluates parts that do not share a width, so the
    /// orientation test that keeps a difference nonnegative has to compare across
    /// widths. Any nonzero limb above the narrower operand decides the comparison
    /// outright; otherwise the shared prefixes are compared most-significant first.
    pub fn compare_with_zero_extension(wide: &[Limb], narrow: &[Limb]) -> Ordering {
        debug_assert!(
            narrow.len() <= wide.len(),
            "zero-extended operand exceeds comparison width"
        );
        let (shared, extension) = wide.split_at(narrow.len());
        if extension.iter().any(|limb| *limb != 0) {
            Ordering::Greater
        } else {
            shared.iter().rev().cmp(narrow.iter().rev())
        }
    }

    /// Zero the destination spans the endpoint products do not write.
    ///
    /// Both endpoint products overwrite their complete ranges. Only the gap between
    /// them and the tail above the exact infinity product must start at zero, before
    /// the overlapping middle coefficients are added into the destination.
    pub fn clear_destination_between_endpoints(
        dst: &mut [Limb],
        low_product_len: usize,
        high_offset: usize,
        high_product_len: usize,
    ) {
        let (before_high, high_and_after) = dst.split_at_mut(high_offset);
        let (_, middle_gap) = before_high.split_at_mut(low_product_len);
        middle_gap.fill(0);
        let (_, trailing_gap) = high_and_after.split_at_mut(high_product_len);
        trailing_gap.fill(0);
    }
}

// ─── Guarded fixed-width addition and subtraction ────────────────────────────

/// An add-and-multiply-by-a-limb backend, selected once per algorithm.
///
/// Adds `scalar * src[..len]` into `dst[..len]` and returns the limb that
/// escapes the top. Passed by value so an evaluator resolves the architecture
/// backend once and reuses it across every point it evaluates, rather than
/// re-selecting inside each call.
///
/// # Safety
///
/// `dst` and `src` must both be valid for `len` limbs, and `dst` must be
/// writable for that span.
pub type AddMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;

impl SharedEval {
    #[allow(
        clippy::inline_always,
        reason = "Critical for recursive reconstruction hot paths"
    )]
    /// Add `src` into `dst` at a limb offset, and report the carry frontier.
    ///
    /// The reconstruction primitive: interpolated coefficients land at increasing
    /// radix offsets and overlap, so each is added rather than written. The return
    /// value is one past the highest limb this call modified, which lets a caller
    /// adding a run of coefficients bound its later work instead of rescanning the
    /// whole destination.
    ///
    /// A carry escaping the added span is propagated through the destination's
    /// guard; the assertion fires only if the buffer was sized without one.
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
        // The contract bounds this sum by `dst.len()`, hence it cannot wrap.
        let carry_start = shift_limbs.wrapping_add(src.len());
        if carry != 0 {
            for index in carry_start..dst.len() {
                // SAFETY: index is produced by a range ending at dst.len().
                let limb = unsafe { dst.get_unchecked_mut(index) };
                let (sum, overflow) = limb.overflowing_add(1);
                *limb = sum;
                if !overflow {
                    return index.wrapping_add(1);
                }
            }
            assert_eq!(carry, 0, "reconstruction buffer dropped a final carry");
            return dst.len();
        }
        carry_start
    }

    /// Add one polynomial part to a guarded fixed-width evaluation.
    ///
    /// Kept as a named operation rather than inlined at its twenty call sites: the
    /// carry frontier the general routine reports is meaningless at zero shift, and
    /// naming the discard once is what keeps those sites readable.
    ///
    /// # Safety
    ///
    /// If `src` is nonempty, `dst.len() >= src.len()`.
    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook evaluation")]
    #[inline(always)]
    pub unsafe fn add_part(dst: &mut [Limb], src: &[Limb]) {
        // SAFETY: the caller guarantees the source fits at offset zero.
        let _ = unsafe { Self::fused_add_shifted_in_place(dst, src, 0) };
    }

    /// Add one interpolated coefficient at a radix-limb offset.
    pub fn add_coefficient_in_place(dst: &mut [Limb], coefficient: &[Limb], shift: usize) {
        let active = Self::active_len(coefficient);
        let available = dst.len().saturating_sub(shift);
        debug_assert!(
            active <= available,
            "interpolated coefficient exceeds the full product width"
        );
        let used_len = min(active, available);
        if used_len == 0 {
            return;
        }
        let (active_span, _) = coefficient.split_at(used_len);
        // SAFETY: `used_len > 0` implies `available > 0`, hence `shift < dst.len()`.
        // Also `active_span.len() == used_len <= available == dst.len() - shift`.
        let _ = unsafe { Self::fused_add_shifted_in_place(dst, active_span, shift) };
    }

    /// Replace an even/odd evaluation pair with their sum and absolute difference.
    ///
    /// A Toom operand evaluated at `+x` and `-x` differs only in the sign of its odd
    /// half, so both points come from one even accumulator and one odd accumulator:
    /// `even + odd` and `even - odd`. Only the second can go negative, so it is
    /// stored as a magnitude and its sign returned.
    ///
    /// The comparison picks the subtraction orientation whose result is nonnegative
    /// *before* the pass, which is what lets the pass itself run on the fused
    /// add-and-subtract kernel instead of branching per limb.
    pub fn sum_and_absolute_difference(even_sum: &mut [Limb], odd_difference: &mut [Limb]) -> bool {
        assert_eq!(
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

    /// The second half of [`Self::sum_and_absolute_difference`], for callers that already
    /// know which orientation is nonnegative.
    ///
    /// A tier evaluating a whole schedule of points derives the sign once, while
    /// forming the even and odd accumulators, and would otherwise pay the comparison
    /// here a second time. `odd_is_larger` must be what the comparison in
    /// [`Self::sum_and_absolute_difference`] would have returned for this pair.
    pub fn apply_sum_and_absolute_difference(
        even_sum: &mut [Limb],
        odd_difference: &mut [Limb],
        odd_is_larger: bool,
    ) {
        assert_eq!(
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
    /// The unfused counterpart of [`Self::apply_sum_and_absolute_difference`], for targets
    /// without a combined add-and-subtract kernel. Those form `E + O` first, multiply
    /// that sum, and only then need `|E - O|` — which is recoverable from the sum and
    /// the odd half alone, in one pass, rather than by evaluating the point again.
    ///
    /// `even_sum` holds `E + O` and `odd` holds `O`. On return `even_sum` holds
    /// `|E - O|` and `odd` is clobbered. `odd_is_larger` carries the same meaning as
    /// in [`Self::apply_sum_and_absolute_difference`].
    pub fn overwrite_sum_with_absolute_difference(
        even_sum: &mut [Limb],
        odd: &mut [Limb],
        odd_is_larger: bool,
    ) {
        assert_eq!(
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
            even_sum.copy_from_slice(odd);
        } else {
            // E - O = (E + O) - 2O, in a single multiply-and-subtract pass.
            Self::sub_mul_word_in_place(even_sum, odd, 2);
        }
    }

    /// Double a fixed-width evaluation whose retained guard proves no overflow.
    pub fn double_evaluation_in_place(value: &mut [Limb]) {
        let mut carry = 0;
        for limb in value {
            let (doubled, overflow_a) = limb.overflowing_add(*limb);
            let (with_carry, overflow_b) = doubled.overflowing_add(carry);
            *limb = with_carry;
            carry = Limb::from(overflow_a | overflow_b);
        }
        debug_assert_eq!(carry, 0, "evaluation exceeded its retained guard limb");
    }

    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook evaluation")]
    /// Add `scalar * src` into `dst`, selecting the backend for this one call.
    ///
    /// The convenience form of [`Self::add_mul_word_with_kernel_in_place`], for callers
    /// that scale a single value. An evaluator running many points should resolve
    /// the kernel once and pass it instead.
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
        assert!(
            src.len() <= dst.len(),
            "scalar-product source exceeds evaluation destination"
        );
        // SAFETY: the release check above proves both pointer spans cover
        // `src.len()` initialized limbs. Rust's borrows make them disjoint.
        let mut carry = unsafe { kernel(dst.as_mut_ptr(), src.as_ptr(), src.len(), scalar) };
        if carry != 0 {
            let (_, suffix) = dst.split_at_mut(src.len());
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

    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
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
        assert!(
            src.len() <= dst.len(),
            "scalar-product source exceeds interpolation destination"
        );
        // SAFETY: the release check above proves both pointer spans cover
        // `src.len()` initialized limbs. Rust's borrows make them disjoint.
        let (carry, initial_borrow) = unsafe {
            ArchKernels::sub_mul_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src.len(), scalar)
        };
        let mut borrow = initial_borrow.wrapping_add(carry);
        if borrow != 0 {
            let (_, suffix) = dst.split_at_mut(src.len());
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

    #[allow(
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
            let (_, suffix) = dst.split_at_mut(shared_len);
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

    #[allow(
        clippy::inline_always,
        reason = "Critical for fixed-width interpolation"
    )]
    #[inline(always)]
    fn sub_full_slices_with_borrow_in_place(dst: &mut [Limb], src: &[Limb], mut borrow: Limb) {
        let shared_len = min(dst.len(), src.len());
        let (dst_shared, dst_suffix) = dst.split_at_mut(shared_len);
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

    #[allow(
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
        let (src1_suffix, src2_suffix) = (
            if shared_len <= src1.len() {
                // SAFETY: shared_len <= src1.len() is true.
                unsafe { src1.get_unchecked(shared_len..) }
            } else {
                &[]
            },
            if shared_len <= src2.len() {
                // SAFETY: shared_len <= src2.len() is true.
                unsafe { src2.get_unchecked(shared_len..) }
            } else {
                &[]
            },
        );
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src1_suffix, borrow1);
        Self::sub_full_slices_with_borrow_in_place(dst_suffix, src2_suffix, borrow2);
    }

    /// Replace `dst` with `positive-dst` modulo its fixed width.
    pub fn reverse_difference_in_place(dst: &mut [Limb], positive: &[Limb]) {
        assert_eq!(
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
