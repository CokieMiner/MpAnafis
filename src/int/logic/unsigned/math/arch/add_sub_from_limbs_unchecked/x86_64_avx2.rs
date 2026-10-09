//! Experimental AVX2 shared-source simultaneous addition and subtraction.
//!
//! This provider is compiled only for direct tests; production selects ADX or
//! scalar code. Native measurements on Ryzen AI 7 350 show no advantage over
//! scalar at the tested sizes. Performance on AVX2 hosts without ADX remains
//! unmeasured; `docs/int/kernel-matrix.md` records the admission requirements.
//!
//! AVX2 has no carry-chain instruction for packed 64-bit lanes.  This kernel
//! therefore computes four raw lanes at once, derives each lane's generate and
//! propagate bits, and applies the four carry/borrow inputs as one packed
//! correction.  The short scalar prefix calculation is what preserves the
//! exact multi-limb semantics at every vector boundary.

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

use super::Limb;

/// Replace `sum` with `sum + source` and write `sum_original - source` to
/// `difference`, returning the final carry and borrow.
///
/// # Safety
///
/// The caller must provide `len` readable/writable limbs for each pointer,
/// keep `sum` disjoint from `source`, and either keep `difference` disjoint
/// from both inputs or make it exactly alias `source`. The caller must verify
/// AVX2 support before invoking this experimental provider directly.
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

    // SAFETY: the caller's span and aliasing contract covers every load/store;
    // `index + 4 <= len` proves all vector offsets are in bounds, and the
    // caller proves AVX2 support for every intrinsic in this block.
    unsafe {
        let sign_bit = _mm256_set1_epi64x(i64::MIN);
        let maximum = _mm256_set1_epi64x(-1);
        let one = _mm256_set1_epi64x(1);
        let lane_shifts = _mm256_set_epi64x(3, 2, 1, 0);
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
            // into bit zero, and mask it to one.  This stays entirely in three
            // packed instructions per carry chain instead of synthesizing two
            // vectors with scalar lane insertions.
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
            index = index.wrapping_add(4);
        }
    }

    while index < len {
        // SAFETY: the vector loop leaves `index <= len`, and this tail advances
        // only while `index < len`; the aliasing contract permits the source
        // load before either destination store.
        let (left, right) = unsafe { (*sum.add(index), *source.add(index)) };
        let (raw_sum, sum_overflow) = left.overflowing_add(right);
        let (corrected_sum, carry_overflow) = raw_sum.overflowing_add(carry as Limb);
        let (raw_difference, difference_underflow) = left.overflowing_sub(right);
        let (corrected_difference, borrow_underflow) =
            raw_difference.overflowing_sub(borrow as Limb);
        // SAFETY: the same tail bounds prove both stores valid; source was
        // loaded before the stores, so exact `difference == source` aliasing is safe.
        unsafe {
            *sum.add(index) = corrected_sum;
            *difference.add(index) = corrected_difference;
        }
        carry = u32::from(sum_overflow | carry_overflow);
        borrow = u32::from(difference_underflow | borrow_underflow);
        index = index.wrapping_add(1);
    }

    (carry as Limb, borrow as Limb)
}

/// Return the input carry for each of four lanes and the carry leaving lane 3.
///
/// `generate` and `propagate` use one bit per lane.  For addition, generate is
/// `raw < left` and propagate is `raw == MAX`; for subtraction they are `left <
/// right` and `raw == 0`, respectively.  The recurrence is identical for both.
#[inline]
pub const fn lane_carries(generate: u32, propagate: u32, incoming: u32) -> (u32, u32) {
    // Parallel-prefix carry over four bits.  After the distance-one combine,
    // each bit describes a two-lane group; the distance-two combine extends
    // that to every preceding lane.  A group propagates the external carry
    // exactly when every lane in the group propagates it.
    let distance_one_generate = generate | (propagate & generate.wrapping_shl(1));
    let distance_one_propagate = propagate & (propagate.wrapping_shl(1) | 0b0001);
    let prefix_generate =
        distance_one_generate | (distance_one_propagate & distance_one_generate.wrapping_shl(2));
    let prefix_propagate =
        distance_one_propagate & (distance_one_propagate.wrapping_shl(2) | 0b0011);
    let incoming_mask = 0_u32.wrapping_sub(incoming);
    let outputs = (prefix_generate | (prefix_propagate & incoming_mask)) & 0b1111;
    let inputs = incoming | (outputs.wrapping_shl(1) & 0b1110);
    (inputs, (outputs >> 3) & 1)
}
