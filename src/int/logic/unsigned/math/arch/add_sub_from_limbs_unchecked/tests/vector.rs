//! Experimental AVX2 arithmetic checked against the scalar recurrence.
//!
//! Four raw lane results supply generate/propagate masks. A binary prefix
//! computes the incoming carry or borrow for each packed correction.

#![expect(
    clippy::cast_ptr_alignment,
    reason = "unaligned AVX2 load/store intrinsics require the typed pointer cast"
)]
#![expect(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    reason = "AVX2 masks and carry states contain at most four bits; conversions to i64 or the native 64-bit Limb are exact"
)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_pd, _mm256_cmpeq_epi64,
    _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_set_epi64x,
    _mm256_set1_epi64x, _mm256_srlv_epi64, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use super::{Limb, prefix::lane_carries};

/// Replace `sum` with `sum + source` and write `sum_original - source` to
/// `difference`, returning the final carry and borrow.
///
/// # Safety
///
/// Nonempty spans must be aligned and cover `len` limbs in live allocations
/// of at most `isize::MAX` bytes. `sum` and `source` must be initialized;
/// `sum` and `difference` must be writable. `sum` must be disjoint from both
/// other spans. `difference` and `source` must be disjoint or exactly identical.
/// The executing CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut index = 0_usize;
    let mut carry = 0_u32;
    let mut borrow = 0_u32;

    // SAFETY: index starts at zero and advances by four only when len-index
    // is at least four. Thus index <= len, both unchecked operations are exact,
    // and every vector access lies within its live span. Both inputs are loaded
    // before either store, preserving the permitted exact alias. The caller
    // proves AVX2 support for every intrinsic in this block.
    unsafe {
        let sign_bit = _mm256_set1_epi64x(i64::MIN);
        let maximum = _mm256_set1_epi64x(-1);
        let one = _mm256_set1_epi64x(1);
        let lane_shifts = _mm256_set_epi64x(3, 2, 1, 0);
        let zero = _mm256_set1_epi64x(0);

        while len.unchecked_sub(index) >= 4 {
            let left = _mm256_loadu_si256(sum.add(index).cast::<__m256i>());
            let right = _mm256_loadu_si256(source.add(index).cast::<__m256i>());
            let raw_sum = _mm256_add_epi64(left, right);
            let raw_difference = _mm256_sub_epi64(left, right);

            // XORing the sign bit turns an unsigned ordering into a signed
            // ordering, which is the comparison AVX2 provides for i64 lanes.
            let left_ordered = _mm256_xor_si256(left, sign_bit);
            let right_ordered = _mm256_xor_si256(right, sign_bit);
            let sum_ordered = _mm256_xor_si256(raw_sum, sign_bit);
            let sum_generates = _mm256_cmpgt_epi64(left_ordered, sum_ordered);
            let difference_generates = _mm256_cmpgt_epi64(right_ordered, left_ordered);
            let sum_propagates = _mm256_cmpeq_epi64(raw_sum, maximum);
            let difference_propagates = _mm256_cmpeq_epi64(raw_difference, zero);

            let sum_generate_bits = _mm256_movemask_pd(_mm256_castsi256_pd(sum_generates)) as u32;
            let sum_propagate_bits = _mm256_movemask_pd(_mm256_castsi256_pd(sum_propagates)) as u32;
            let difference_generate_bits =
                _mm256_movemask_pd(_mm256_castsi256_pd(difference_generates)) as u32;
            let difference_propagate_bits =
                _mm256_movemask_pd(_mm256_castsi256_pd(difference_propagates)) as u32;

            let (sum_inputs, next_carry) =
                lane_carries(sum_generate_bits, sum_propagate_bits, carry);
            let (difference_inputs, next_borrow) =
                lane_carries(difference_generate_bits, difference_propagate_bits, borrow);
            // Broadcast the four-bit input mask, shift the bit for lane `i`
            // into bit zero, and mask it to one.
            let sum_correction = _mm256_and_si256(
                _mm256_srlv_epi64(_mm256_set1_epi64x(i64::from(sum_inputs)), lane_shifts),
                one,
            );
            let difference_correction = _mm256_and_si256(
                _mm256_srlv_epi64(
                    _mm256_set1_epi64x(i64::from(difference_inputs)),
                    lane_shifts,
                ),
                one,
            );
            let corrected_sum = _mm256_add_epi64(raw_sum, sum_correction);
            let corrected_difference = _mm256_sub_epi64(raw_difference, difference_correction);
            _mm256_storeu_si256(sum.add(index).cast::<__m256i>(), corrected_sum);
            _mm256_storeu_si256(
                difference.add(index).cast::<__m256i>(),
                corrected_difference,
            );
            carry = next_carry;
            borrow = next_borrow;
            index = index.unchecked_add(4);
        }
    }

    for tail_index in index..len {
        // SAFETY: tail_index < len bounds both initialized source reads.
        // Loading the source before either store preserves the permitted alias.
        let (left, right) = unsafe { (*sum.add(tail_index), *source.add(tail_index)) };
        let (raw_sum, sum_overflow) = left.overflowing_add(right);
        let (corrected_sum, carry_overflow) = raw_sum.overflowing_add(carry as Limb);
        let (raw_difference, difference_underflow) = left.overflowing_sub(right);
        let (corrected_difference, borrow_underflow) =
            raw_difference.overflowing_sub(borrow as Limb);
        // SAFETY: the same tail bounds prove both stores valid; source was
        // loaded before the stores, so exact `difference == source` aliasing is safe.
        unsafe {
            *sum.add(tail_index) = corrected_sum;
            *difference.add(tail_index) = corrected_difference;
        }
        carry = u32::from(sum_overflow | carry_overflow);
        borrow = u32::from(difference_underflow | borrow_underflow);
    }

    (carry as Limb, borrow as Limb)
}
