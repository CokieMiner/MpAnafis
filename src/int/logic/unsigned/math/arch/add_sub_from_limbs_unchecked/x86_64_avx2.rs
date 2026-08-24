//! AVX2 shared-source simultaneous addition and subtraction.
//!
//! AVX2 has no carry-chain instruction for packed 64-bit lanes.  This kernel
//! therefore computes four raw lanes at once, derives each lane's generate and
//! propagate bits, and applies the four carry/borrow inputs as one packed
//! correction.  The short scalar prefix calculation is what preserves the
//! exact multi-limb semantics at every vector boundary.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "unaligned AVX2 load/store intrinsics require the typed pointer cast"
)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "AVX2 movemask returns four boolean lane bits, which fit exactly in u8"
)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64,
    _mm256_loadu_si256, _mm256_movemask_pd, _mm256_set_epi64x, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use super::Limb;

/// Replace `sum` with `sum + source` and write `sum_original - source` to
/// `difference`, returning the final carry and borrow.
///
/// # Safety
///
/// The caller must provide `len` readable/writable limbs for each pointer,
/// keep `sum` disjoint from `source`, and either keep `difference` disjoint
/// from both inputs or make it exactly alias `source`.  The runtime selector
/// proves AVX2 before this function is installed.
#[target_feature(enable = "avx2")]
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut index = 0_usize;
    let mut carry = 0_u8;
    let mut borrow = 0_u8;

    // SAFETY: the caller's span and aliasing contract covers every load/store;
    // `index + 4 <= len` proves all vector offsets are in bounds, and the
    // runtime selector proves AVX2 for every intrinsic in this block.
    unsafe {
        let sign_bit = _mm256_set1_epi64x(i64::MIN);
        let maximum = _mm256_set1_epi64x(-1);
        let zero = _mm256_set1_epi64x(0);

        while len.wrapping_sub(index) >= 4 {
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

            let sum_generate_bits = _mm256_movemask_pd(_mm256_castsi256_pd(sum_generates)) as u8;
            let sum_propagate_bits = _mm256_movemask_pd(_mm256_castsi256_pd(sum_propagates)) as u8;
            let difference_generate_bits =
                _mm256_movemask_pd(_mm256_castsi256_pd(difference_generates)) as u8;
            let difference_propagate_bits =
                _mm256_movemask_pd(_mm256_castsi256_pd(difference_propagates)) as u8;

            let (sum_inputs, next_carry) =
                lane_carries(sum_generate_bits, sum_propagate_bits, carry);
            let (difference_inputs, next_borrow) =
                lane_carries(difference_generate_bits, difference_propagate_bits, borrow);
            let sum_correction = _mm256_set_epi64x(
                i64::from((sum_inputs >> 3) & 1),
                i64::from((sum_inputs >> 2) & 1),
                i64::from((sum_inputs >> 1) & 1),
                i64::from(sum_inputs & 1),
            );
            let difference_correction = _mm256_set_epi64x(
                i64::from((difference_inputs >> 3) & 1),
                i64::from((difference_inputs >> 2) & 1),
                i64::from((difference_inputs >> 1) & 1),
                i64::from(difference_inputs & 1),
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
            index = index.wrapping_add(4);
        }
    }

    while index < len {
        // SAFETY: the vector loop leaves `index <= len`, and this tail advances
        // only while `index < len`; the aliasing contract permits the source
        // load before either destination store.
        let (left, right) = unsafe { (*sum.add(index), *source.add(index)) };
        let (raw_sum, sum_overflow) = left.overflowing_add(right);
        let (corrected_sum, carry_overflow) = raw_sum.overflowing_add(Limb::from(carry));
        let (raw_difference, difference_underflow) = left.overflowing_sub(right);
        let (corrected_difference, borrow_underflow) =
            raw_difference.overflowing_sub(Limb::from(borrow));
        // SAFETY: the same tail bounds prove both stores valid; source was
        // loaded before the stores, so exact `difference == source` aliasing is safe.
        unsafe {
            *sum.add(index) = corrected_sum;
            *difference.add(index) = corrected_difference;
        }
        carry = u8::from(sum_overflow | carry_overflow);
        borrow = u8::from(difference_underflow | borrow_underflow);
        index = index.wrapping_add(1);
    }

    (Limb::from(carry), Limb::from(borrow))
}

/// Return the input carry for each of four lanes and the carry leaving lane 3.
///
/// `generate` and `propagate` use one bit per lane.  For addition, generate is
/// `raw < left` and propagate is `raw == MAX`; for subtraction they are `left <
/// right` and `raw == 0`, respectively.  The recurrence is identical for both.
#[inline]
const fn lane_carries(generate: u8, propagate: u8, incoming: u8) -> (u8, u8) {
    let mut inputs = 0_u8;
    let mut carry = incoming;
    let mut lane = 0_u8;
    while lane < 4 {
        inputs |= carry << lane;
        let generated = (generate >> lane) & 1;
        let passes = (propagate >> lane) & 1;
        carry = generated | (passes & carry);
        lane = lane.wrapping_add(1);
    }
    (inputs, carry)
}
