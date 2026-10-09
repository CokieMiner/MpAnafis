//! Seven-point interpolation for the Toom-Cook 4 tier.
//!
//! # References
//!
//! - Bodrato, M., & Zanoni, A. (2007). Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices. *Proceedings of ISSAC '07*, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.

#![expect(
    unsafe_code,
    reason = "Retained coefficient guards and the complete four-part product bound disjoint reconstruction chunks and carry positions"
)]

use super::{Addition, ArchKernels, Limb, SharedEval, Toom4};

/// Middle evaluated products and signs consumed by interpolation.
pub struct MiddleValues<'buffer> {
    pub neg_two: &'buffer mut [Limb],
    pub one: &'buffer mut [Limb],
    pub neg_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub neg_two_negative: bool,
    pub neg_one_negative: bool,
}

impl Toom4 {
    /// Interpolate using disjoint constant and infinity endpoint products.
    pub fn interpolate_with_endpoints(zero: &[Limb], infinity: &[Limb], middle: MiddleValues<'_>) {
        let MiddleValues {
            neg_two,
            one,
            neg_one,
            two,
            half,
            neg_two_negative,
            neg_one_negative,
        } = middle;
        // These are exact identities for coefficients c0..c6 of the degree-six
        // product polynomial. All /2 and /4 steps below operate on values already
        // proven nonnegative by paired +/- evaluations. `half - 65*one` and the
        // later `neg_two` intermediate may be negative; they remain fixed-width
        // two's-complement values and are divided only by odd 9 or 15, for which
        // multiplication by the inverse modulo B^n equals exact signed division.
        // Each subsequent even division occurs only after the formula has restored
        // a nonnegative coefficient combination.
        debug_assert_eq!(
            half.len(),
            two.len(),
            "fixed-width addition requires equal widths"
        );
        let _ = Addition::add_slice_in_place(half, two);
        if neg_two_negative {
            SharedEval::exact_half_sum_in_place(neg_two, two);
        } else {
            SharedEval::exact_half_reverse_difference_in_place(neg_two, two);
        }

        SharedEval::sub_two_full_slices_in_place(two, zero, neg_two);
        SharedEval::exact_div4_in_place(two);
        SharedEval::sub_mul_word_in_place(two, infinity, 16);

        if neg_one_negative {
            SharedEval::exact_half_sum_in_place(neg_one, one);
        } else {
            SharedEval::exact_half_reverse_difference_in_place(neg_one, one);
        }
        SharedEval::sub_full_slices_in_place(one, neg_one);

        SharedEval::sub_mul_word_in_place(half, one, 65);
        SharedEval::sub_two_full_slices_in_place(one, infinity, zero);
        SharedEval::add_mul_word_in_place(half, one, 45);
        SharedEval::exact_div2_in_place(half);

        SharedEval::sub_full_slices_in_place(two, one);
        SharedEval::exact_div_radix_minus_one_in_place::<3>(two);
        SharedEval::sub_full_slices_in_place(one, two);

        SharedEval::reverse_difference_in_place(neg_two, half);

        SharedEval::exact_sub_mul_word_odd_in_place(half, neg_one, 8, 9);
        SharedEval::sub_full_slices_in_place(neg_one, half);

        SharedEval::exact_div_radix_minus_one_in_place::<15>(neg_two);
        debug_assert_eq!(
            neg_two.len(),
            half.len(),
            "fixed-width addition requires equal widths"
        );
        let _ = Addition::add_slice_in_place(neg_two, half);
        SharedEval::exact_div2_in_place(neg_two);
        SharedEval::sub_full_slices_in_place(half, neg_two);
    }
    /// Add the four coefficients surrounding an in-place quadratic coefficient.
    pub fn reconstruct_around_quadratic(
        dst: &mut [Limb],
        split_len: usize,
        first: &[Limb],
        third: &mut [Limb],
        fourth: &mut [Limb],
        fifth: &mut [Limb],
    ) {
        // c0, c2, and c6 already occupy their final ranges. Add c1 normally, then
        // fold each lower overlap and carry into the next coefficient's high half.
        // This writes c3..c5 in one radix chunk per pass instead of adding three
        // complete, mutually overlapping 2m-limb coefficients.
        SharedEval::add_coefficient_in_place(dst, first, split_len);
        // SAFETY: four-part admission gives dst.len()>=6m+2. The three
        // destination chunks occupy [3m,6m); third is a disjoint initialized
        // (2m+1)-limb coefficient buffer. All offsets fit the full product width.
        let (third_chunk, fourth_chunk, fifth_chunk, endpoint, third_low, third_after_low) = unsafe {
            let third_and_after = dst.split_at_mut_unchecked(split_len.unchecked_mul(3)).1;
            let (third_chunk, after_third) = third_and_after.split_at_mut_unchecked(split_len);
            let (fourth_chunk, after_fourth) = after_third.split_at_mut_unchecked(split_len);
            let (fifth_chunk, endpoint) = after_fourth.split_at_mut_unchecked(split_len);
            let (third_low, third_after_low) = third.split_at_mut_unchecked(split_len);
            (
                third_chunk,
                fourth_chunk,
                fifth_chunk,
                endpoint,
                third_low,
                third_after_low,
            )
        };
        let carry_from_third = Addition::add_slice_in_place(third_chunk, third_low);
        let overlap_carry = Addition::add_slice_in_place(third_after_low, fourth_chunk);
        // Fold the escaping carries through the c3 high half: `carry_from_third`
        // lands at the bottom (c3[m]), `overlap_carry` lands one past the
        // just-written overlap (c3[2m]). Both propagate so the middle and guard
        // slices below already carry them forward.
        Self::add_guard_carry(third_after_low, 0, carry_from_third);
        Self::add_guard_carry(third_after_low, split_len, overlap_carry);
        // SAFETY: third_after_low has m+1 limbs and fourth has 2m+1 limbs.
        let ((third_middle, third_guard), (fourth_low, fourth_after_low)) = unsafe {
            (
                third_after_low.split_at_mut_unchecked(split_len),
                fourth.split_at_mut_unchecked(split_len),
            )
        };
        // SAFETY: all three slices span exactly split_len disjoint limbs.
        let carry_from_fourth = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                fourth_chunk.as_mut_ptr(),
                third_middle.as_ptr(),
                fourth_low.as_ptr(),
                split_len,
            )
        };
        let guard_carry = Addition::add_slice_in_place(fourth_after_low, third_guard);
        // `guard_carry` exits fourth_after_low[0] and must land at index one,
        // then propagate; `carry_from_fourth` enters at the bottom.
        Self::add_guard_carry(fourth_after_low, 1, guard_carry);
        Self::add_guard_carry(fourth_after_low, 0, carry_from_fourth);
        // SAFETY: fourth_after_low has m+1 limbs and fifth has 2m+1 limbs.
        let ((fourth_middle, fourth_guard), (fifth_low, fifth_after_low)) = unsafe {
            (
                fourth_after_low.split_at_mut_unchecked(split_len),
                fifth.split_at_mut_unchecked(split_len),
            )
        };
        // SAFETY: all three slices span exactly split_len disjoint limbs.
        let carry_from_fifth = unsafe {
            ArchKernels::add_limbs_3_unchecked(
                fifth_chunk.as_mut_ptr(),
                fourth_middle.as_ptr(),
                fifth_low.as_ptr(),
                split_len,
            )
        };
        let final_guard_carry = Addition::add_slice_in_place(fifth_after_low, fourth_guard);
        Self::add_guard_carry(fifth_after_low, 1, final_guard_carry);
        Self::add_guard_carry(fifth_after_low, 0, carry_from_fifth);
        SharedEval::add_coefficient_in_place(endpoint, fifth_after_low, 0);
    }
    /// Propagates a binary carry from index zero, one, or m in an m+1-limb guard.
    fn add_guard_carry(value: &mut [Limb], start: usize, mut carry: Limb) {
        if carry == 0 {
            return;
        }
        // SAFETY: m>=1, value.len()=m+1, and every caller uses start in {0,1,m}.
        let (_, suffix) = unsafe { value.split_at_mut_unchecked(start) };
        for limb in suffix {
            let (sum, overflow) = limb.overflowing_add(carry);
            *limb = sum;
            if !overflow {
                carry = 0;
                break;
            }
        }
        debug_assert_eq!(carry, 0, "interpolation guard dropped a carry");
    }
}
