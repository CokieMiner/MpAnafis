//! AVX2 packing of four 64-bit limbs into sixteen 16-bit NTT digits.

#![allow(
    unsafe_code,
    reason = "AVX2 intrinsics and validated raw spans require localized unsafe code"
)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The narrowed values are masked 16-bit digits"
)]

use core::arch::x86_64::{
    _mm256_and_si256, _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_set1_epi32,
    _mm256_srli_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi32, _mm256_unpacklo_epi32,
};

use super::NttDigitsKernels;

pub fn ntt_digits_u32() -> NttDigitsKernels {
    NttDigitsKernels {
        pack_16: limbs_to_digits_16_avx2,
    }
}

/// Packs complete groups of four limbs, then uses the scalar backend for a
/// partial group.  The input and output are little-endian by limb/digit.
///
/// # Safety
/// `limbs` is readable for `len` `u64`s and `dst` is writable for `dst_len`
/// `u32`s.  The caller must execute this function only when AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn limbs_to_digits_16_avx2(
    dst: *mut u32,
    limbs: *const u64,
    len: usize,
    dst_len: usize,
) -> usize {
    let mask = _mm256_set1_epi32(i32::from(u16::MAX));
    let mut index = 0_usize;
    let mut count = 0_usize;
    while index.wrapping_add(4) <= len && count.wrapping_add(16) <= dst_len {
        // SAFETY: the loop bounds provide four readable limbs and sixteen
        // writable digits; unaligned access is intentional for slice spans.
        let values = unsafe { _mm256_loadu_si256(limbs.add(index).cast()) };
        let low_half = _mm256_and_si256(values, mask);
        let high_half = _mm256_and_si256(_mm256_srli_epi32(values, 16), mask);
        let first_halves = _mm256_unpacklo_epi32(low_half, high_half);
        let second_halves = _mm256_unpackhi_epi32(low_half, high_half);
        // `unpack*` interleaves independently inside each 128-bit lane.
        // Recombine those lanes to preserve the limb-major output order.
        let first = _mm256_permute2x128_si256(first_halves, second_halves, 0x20);
        let second = _mm256_permute2x128_si256(first_halves, second_halves, 0x31);
        // SAFETY: the loop bounds provide sixteen writable output digits.
        unsafe {
            _mm256_storeu_si256(dst.add(count).cast(), first);
            _mm256_storeu_si256(dst.add(count.wrapping_add(8)).cast(), second);
        }
        index = index.wrapping_add(4);
        count = count.wrapping_add(16);
    }

    while index < len && count < dst_len {
        // SAFETY: `index < len` guarantees a readable input limb.
        let limb = unsafe { *limbs.add(index) };
        for shift in [0_u32, 16, 32, 48] {
            if count == dst_len {
                break;
            }
            // SAFETY: `count < dst_len` guarantees a writable output slot.
            unsafe {
                *dst.add(count) = (limb >> shift) as u32 & u32::from(u16::MAX);
            }
            count = count.wrapping_add(1);
        }
        index = index.wrapping_add(1);
    }

    while count != 0 {
        // SAFETY: `count` is a number of initialized output slots.
        let is_zero = unsafe { *dst.add(count.wrapping_sub(1)) == 0 };
        if !is_zero {
            break;
        }
        count = count.wrapping_sub(1);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{super::scalar::limbs_to_digits_16_scalar, limbs_to_digits_16_avx2};

    #[test]
    fn forced_avx2_matches_scalar_for_vector_and_tail_lengths() {
        let limbs = [
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
            0x0001_0203_0405_0607,
            0x8899_aabb_ccdd_eeff,
            0x1357_9bdf_2468_ace0,
            0xffff_0000_aaaa_5555,
            0xdead_beef_cafe_babe,
            0x8000_0000_0000_0001,
        ];
        for len in [4_usize, 5, 8] {
            let mut expected = [0_u32; 32];
            let mut actual = [0_u32; 32];
            // SAFETY: both arrays cover complete input and output spans.
            let expected_len = unsafe {
                limbs_to_digits_16_scalar(
                    expected.as_mut_ptr(),
                    limbs.as_ptr(),
                    len,
                    expected.len(),
                )
            };
            // SAFETY: this test is compiled only with the AVX2 backend.
            let actual_len = unsafe {
                limbs_to_digits_16_avx2(actual.as_mut_ptr(), limbs.as_ptr(), len, actual.len())
            };
            assert_eq!(actual_len, expected_len, "digit count for {len} limbs");
            assert_eq!(actual.get(..actual_len), expected.get(..expected_len));
        }
    }
}
