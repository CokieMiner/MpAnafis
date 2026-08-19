//! AArch64 NEON provider for 16-bit NTT digit packing.

use core::arch::aarch64::{
    uint32x4_t, uint64x2_t, vandq_u32, vdupq_n_u32, vld1q_u64, vreinterpretq_u32_u64, vshrq_n_u32,
    vst1q_u32, vzip1q_u32, vzip2q_u32,
};

use super::NttDigitsKernels;

/// Packs two 64-bit limbs into eight little-endian 16-bit digits.
#[inline]
unsafe fn pack_two(values: uint64x2_t, dst: *mut u32) {
    // SAFETY: the caller executes this helper only on AArch64 with NEON,
    // and `values` is a valid register value rather than a memory span.
    let (first, second): (uint32x4_t, uint32x4_t) = unsafe {
        let words = vreinterpretq_u32_u64(values);
        let mask = vdupq_n_u32(u32::from(u16::MAX));
        let low = vandq_u32(words, mask);
        let high = vandq_u32(vshrq_n_u32(words, 16), mask);
        // The input dwords are [l0.0, l0.1, l1.0, l1.1]. Interleaving the
        // low/high halves yields [d0,d1,d2,d3] and [d4,d5,d6,d7].
        (vzip1q_u32(low, high), vzip2q_u32(low, high))
    };
    // SAFETY: the caller provides eight writable output slots.
    unsafe {
        vst1q_u32(dst, first);
        vst1q_u32(dst.add(4), second);
    }
}

/// Packs up to `dst_len` digits from `len` 64-bit limbs.
///
/// # Safety
/// `limbs` is readable for `len` `u64`s and `dst` is writable for `dst_len`
/// `u32`s. The caller executes this function only on AArch64 NEON hardware.
pub unsafe fn pack_16(dst: *mut u32, limbs: *const u64, len: usize, dst_len: usize) -> usize {
    let mut index = 0_usize;
    let mut count = 0_usize;
    while index.wrapping_add(2) <= len && count.wrapping_add(8) <= dst_len {
        // SAFETY: loop bounds provide two readable limbs and eight outputs.
        let values = unsafe { vld1q_u64(limbs.add(index)) };
        // SAFETY: loop bounds provide eight writable outputs.
        unsafe { pack_two(values, dst.add(count)) };
        index = index.wrapping_add(2);
        count = count.wrapping_add(8);
    }
    while index < len && count < dst_len {
        // SAFETY: `index < len` guarantees a readable limb.
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

pub fn ntt_digits_u32() -> NttDigitsKernels {
    NttDigitsKernels { pack_16 }
}
